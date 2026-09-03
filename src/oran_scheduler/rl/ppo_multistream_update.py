from dataclasses import (
    dataclass,
    fields,
    is_dataclass,
    replace,
)
from typing import Literal

import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
)
from oran_scheduler.rl.ppo_agent_augmentation import (
    PPOAugmentedUpdateData,
    build_candidate_augmented_ppo_update,
)
from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_multistream_buffer import (
    PPOMultiStreamTransitionBuffer,
)
from oran_scheduler.rl.ppo_rollout import (
    PPORolloutBatch,
)
from oran_scheduler.rl.ppo_tti_trajectory import (
    prepare_ppo_trajectory_with_gae,
)


PPOMultiStreamUpdateBoundaryMode = Literal[
    "require_exact",
    "use_all_when_reached",
]


@dataclass(frozen=True)
class PPOMultiStreamUpdateConfig:
    """
    Centralized PPO update-boundary interpretation.

    PAPER-SPECIFIED:
        M = 128 experiences per PPO update.

    UNRESOLVED PAPER / REPRODUCTION ISSUE:
        Under our current joint-layer interpretation:

            21 cells
            x 4 user-slot decisions
            =
            84 transitions / synchronized TTI.

        Therefore:

            after one TTI:
                84

            after two TTIs:
                168

        Exact M=128 does not align naturally with
        whole synchronized TTIs.

    Modes:

        "require_exact":
            Preserve literal exact-M semantics.

            - fewer than M:
                no update yet

            - exactly M:
                prepare update

            - greater than M:
                raise rather than silently decide
                what to do with overflow.

        "use_all_when_reached":
            Treat M as the minimum update threshold.

            Once >= M transitions have been
            collected under the same behavior
            policy, use ALL of them for one update.

            This avoids carrying stale old-policy
            samples across an optimizer update, but
            it is an OPEN-REPRODUCTION choice.
    """

    boundary_mode: (
        PPOMultiStreamUpdateBoundaryMode
    )

    def __post_init__(self) -> None:
        if self.boundary_mode not in (
            "require_exact",
            "use_all_when_reached",
        ):
            raise ValueError(
                "Unsupported multi-stream PPO "
                "boundary mode."
            )


@dataclass(frozen=True)
class PPOPreparedStreamDiagnostics:
    """
    Diagnostics for one independent trajectory
    stream used in one centralized update.
    """

    stream_id: int

    num_transitions: int

    bootstrap_value: torch.Tensor

    mean_td_residual: torch.Tensor

    mean_advantage: torch.Tensor

    mean_target_return: torch.Tensor


@dataclass(frozen=True)
class PPOMultiStreamPreparedUpdate:
    """
    Fully prepared centralized PPO optimization data.

    real_batch:
        Concatenation of the REAL trajectory rows
        after each stream has independently undergone
        GAE.

        No permutation copies are present here.

    optimization:
        Final optimizer data after optional candidate
        permutation augmentation.

    IMPORTANT:
        `num_real_transitions` describes real
        environment transitions.

        `num_optimizer_samples` may be larger due to
        candidate-order augmentation.
    """

    real_batch: PPORolloutBatch

    real_advantage: torch.Tensor

    real_target_return: torch.Tensor

    optimization: PPOAugmentedUpdateData

    num_real_transitions: int

    num_optimizer_samples: int

    num_transitions_by_stream: tuple[
        int,
        ...
    ]

    stream_diagnostics: tuple[
        PPOPreparedStreamDiagnostics,
        ...
    ]

    boundary_mode: (
        PPOMultiStreamUpdateBoundaryMode
    )


def _concatenate_rollout_batches(
    *,
    batches: tuple[
        PPORolloutBatch,
        ...
    ],
    batch_sizes: tuple[
        int,
        ...
    ],
) -> PPORolloutBatch:
    """
    Concatenate independent prepared rollout batches
    along their transition dimension.

    Every tensor field in PPORolloutBatch is expected
    to have transition as dimension 0.

    Using dataclass introspection keeps this helper
    aligned with the current rollout structure
    without rebuilding PPORolloutBatch manually.
    """

    if len(
        batches
    ) == 0:
        raise ValueError(
            "At least one rollout batch is required."
        )

    if len(
        batches
    ) != len(
        batch_sizes
    ):
        raise ValueError(
            "batches and batch_sizes must have the "
            "same length."
        )

    first = batches[0]

    if not is_dataclass(
        first
    ):
        raise TypeError(
            "PPORolloutBatch must remain a "
            "dataclass."
        )

    replacements = {}

    for field_info in fields(
        first
    ):
        values = tuple(
            getattr(
                batch,
                field_info.name,
            )
            for batch
            in batches
        )

        if not all(
            isinstance(
                value,
                torch.Tensor,
            )
            for value
            in values
        ):
            raise TypeError(
                "Every PPORolloutBatch field must "
                "be a torch.Tensor for centralized "
                "concatenation."
            )

        for value, expected_size in zip(
            values,
            batch_sizes,
        ):
            if value.ndim < 1:
                raise ValueError(
                    f"PPORolloutBatch field "
                    f"'{field_info.name}' does not "
                    "have a transition dimension."
                )

            if int(
                value.shape[0]
            ) != expected_size:
                raise ValueError(
                    f"PPORolloutBatch field "
                    f"'{field_info.name}' has an "
                    "unexpected transition count."
                )

        replacements[
            field_info.name
        ] = torch.cat(
            values,
            dim=0,
        )

    return replace(
        first,
        **replacements,
    )


def _validate_update_boundary(
    *,
    buffer: PPOMultiStreamTransitionBuffer,
    config: PPOMultiStreamUpdateConfig,
) -> bool:
    """
    Return True when an update should be prepared.

    Return False when more collection is required.

    Raise when the configured exact-boundary
    interpretation has been violated.
    """

    num_transitions = len(
        buffer
    )

    update_size = (
        buffer
        .config
        .update_size
    )

    if num_transitions < update_size:
        return False

    if (
        config.boundary_mode
        == "require_exact"
    ):
        if num_transitions > update_size:
            raise RuntimeError(
                "Centralized PPO collection crossed "
                "the exact update boundary. The "
                "configured 'require_exact' policy "
                "does not permit silently dropping, "
                "carrying, or consuming overflow."
            )

        return True

    if (
        config.boundary_mode
        == "use_all_when_reached"
    ):
        return True

    raise RuntimeError(
        "Unexpected multi-stream boundary mode."
    )


def prepare_multistream_ppo_update_if_ready(
    *,
    buffer: PPOMultiStreamTransitionBuffer,
    actor: OneLDSPPOActor,
    critic: nn.Module,
    gae_config: PPOGAEConfig,
    update_config: PPOMultiStreamUpdateConfig,
    candidate_augmentation_config: (
        PPOCandidateAugmentationConfig | None
    ) = None,
    candidate_augmentation_generator: (
        torch.Generator | None
    ) = None,
) -> (
    PPOMultiStreamPreparedUpdate
    | None
):
    """
    Prepare one centralized PPO optimization batch.

    Sequence:

        independent cell trajectories
                ↓
        GAE separately per cell
                ↓
        concatenate optimizer rows
                ↓
        candidate-order augmentation
                ↓
        centralized PPO batch

    IMPORTANT:
        This function does NOT call optimizer.step()
        and does NOT clear the multi-stream buffer.

        The behavior actor therefore remains
        unchanged while candidate-augmented old
        log-probabilities are reconstructed.
    """

    is_ready = _validate_update_boundary(
        buffer=buffer,
        config=update_config,
    )

    if not is_ready:
        return None

    prepared_batches: list[
        PPORolloutBatch
    ] = []

    advantages: list[
        torch.Tensor
    ] = []

    target_returns: list[
        torch.Tensor
    ] = []

    batch_sizes: list[
        int
    ] = []

    stream_diagnostics: list[
        PPOPreparedStreamDiagnostics
    ] = []

    num_transitions_by_stream: list[
        int
    ] = []

    for stream_id in range(
        buffer.config.num_streams
    ):
        transitions = (
            buffer
            .transitions_for_stream(
                stream_id
            )
        )

        num_stream_transitions = len(
            transitions
        )

        num_transitions_by_stream.append(
            num_stream_transitions
        )

        if num_stream_transitions == 0:
            continue

        #
        # CRITICAL:
        #
        # GAE is calculated only inside THIS cell's
        # trajectory.
        #
        prepared = (
            prepare_ppo_trajectory_with_gae(
                transitions=transitions,
                critic=critic,
                config=gae_config,
            )
        )

        prepared_batches.append(
            prepared.batch
        )

        advantages.append(
            prepared.gae.advantage
        )

        target_returns.append(
            prepared.gae.target_return
        )

        batch_sizes.append(
            num_stream_transitions
        )

        stream_diagnostics.append(
            PPOPreparedStreamDiagnostics(
                stream_id=stream_id,
                num_transitions=(
                    num_stream_transitions
                ),
                bootstrap_value=(
                    prepared
                    .bootstrap_value
                    .detach()
                    .clone()
                ),
                mean_td_residual=(
                    prepared
                    .gae
                    .td_residual
                    .mean()
                    .detach()
                    .clone()
                ),
                mean_advantage=(
                    prepared
                    .gae
                    .advantage
                    .mean()
                    .detach()
                    .clone()
                ),
                mean_target_return=(
                    prepared
                    .gae
                    .target_return
                    .mean()
                    .detach()
                    .clone()
                ),
            )
        )

    if len(
        prepared_batches
    ) == 0:
        raise RuntimeError(
            "Centralized PPO buffer reports that an "
            "update is ready but contains no stream "
            "transitions."
        )

    real_batch = (
        _concatenate_rollout_batches(
            batches=tuple(
                prepared_batches
            ),
            batch_sizes=tuple(
                batch_sizes
            ),
        )
    )

    real_advantage = torch.cat(
        advantages,
        dim=0,
    )

    real_target_return = torch.cat(
        target_returns,
        dim=0,
    )

    num_real_transitions = int(
        real_batch.states.shape[0]
    )

    if num_real_transitions != len(
        buffer
    ):
        raise RuntimeError(
            "Centralized batch transition count "
            "does not match the multi-stream "
            "buffer."
        )

    #
    # Candidate permutation happens only AFTER each
    # real trajectory has produced its GAE labels.
    #
    optimization = (
        build_candidate_augmented_ppo_update(
            actor=actor,
            batch=real_batch,
            advantage=real_advantage,
            target_return=(
                real_target_return
            ),
            config=(
                candidate_augmentation_config
            ),
            generator=(
                candidate_augmentation_generator
            ),
        )
    )

    return PPOMultiStreamPreparedUpdate(
        real_batch=real_batch,
        real_advantage=real_advantage,
        real_target_return=(
            real_target_return
        ),
        optimization=optimization,
        num_real_transitions=(
            num_real_transitions
        ),
        num_optimizer_samples=(
            optimization
            .num_optimizer_samples
        ),
        num_transitions_by_stream=tuple(
            num_transitions_by_stream
        ),
        stream_diagnostics=tuple(
            stream_diagnostics
        ),
        boundary_mode=(
            update_config.boundary_mode
        ),
    )


