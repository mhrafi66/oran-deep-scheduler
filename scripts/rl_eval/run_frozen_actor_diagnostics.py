from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)
from oran_scheduler.rl.ppo_checkpoint import (
    load_ppo_model_checkpoint,
)


# ============================================================
# FIXED 1LDS DIMENSIONS
# ============================================================

NUM_CANDIDATES = 10
NUM_RBGS = 18

FEATURES_PER_CANDIDATE = (
    5
    + 2 * NUM_RBGS
)

NUM_ACTIONS = (
    NUM_CANDIDATES
    + 1
)

STATE_SIZE = (
    NUM_CANDIDATES
    * FEATURES_PER_CANDIDATE
)


# ============================================================
# LOCKED EVALUATION PLAN
#
# 5 independent controlled state/perturbation seeds
# x
# 3 evaluation families
#
# = 15 Slurm tasks
# ============================================================

EVALUATIONS = (
    "permutation",
    "perturbation",
    "ablation",
)

NUM_REPLICATES = 5

BASE_SEED = 202609230


# Fixed local perturbation magnitudes.
PERTURBATION_EPSILONS = (
    0.01,
    0.02,
    0.05,
    0.10,
)


# Per-candidate feature layout:
#
# 0       past throughput
# 1       rank
# 2       allocated RBG count
# 3       buffer
# 4       wideband CQI
# 5:23    subband CQI
# 23:41   cross-correlation
#
FEATURE_GROUPS = {
    "past_throughput": (
        0,
        1,
    ),

    "rank": (
        1,
        2,
    ),

    "allocated_rbg_count": (
        2,
        3,
    ),

    "buffer": (
        3,
        4,
    ),

    "wideband_cqi": (
        4,
        5,
    ),

    "subband_cqi": (
        5,
        5 + NUM_RBGS,
    ),

    "cross_correlation": (
        5 + NUM_RBGS,
        5 + 2 * NUM_RBGS,
    ),
}


CONFIDENCE_BINS = (
    (
        "lt_0p002",
        0.0,
        0.002,
    ),

    (
        "0p002_0p005",
        0.002,
        0.005,
    ),

    (
        "0p005_0p010",
        0.005,
        0.010,
    ),

    (
        "0p010_0p020",
        0.010,
        0.020,
    ),

    (
        "ge_0p020",
        0.020,
        None,
    ),
)


# ============================================================
# ARGUMENTS / TASK MAPPING
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-id",
        type=int,
        default=int(
            os.getenv(
                "RL_EVAL_TASK_ID",
                "0",
            )
        ),
    )

    return parser.parse_args()


def task_spec(
    task_id: int,
) -> tuple[
    str,
    int,
    int,
]:

    total_tasks = (
        len(EVALUATIONS)
        * NUM_REPLICATES
    )

    if (
        task_id < 0
        or task_id >= total_tasks
    ):
        raise ValueError(
            "task-id must be in "
            f"[0, {total_tasks - 1}]."
        )

    evaluation_index = (
        task_id
        // NUM_REPLICATES
    )

    replicate_index = (
        task_id
        % NUM_REPLICATES
    )

    evaluation = (
        EVALUATIONS[
            evaluation_index
        ]
    )

    seed = (
        BASE_SEED
        + replicate_index
    )

    return (
        evaluation,
        replicate_index,
        seed,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def resolve_checkpoint() -> Path:

    default_path = (
        Path.home()
        / "oran-deep-scheduler"
        / "experiments"
        / "checkpoints"
        / "paper_1lds_ppo_500tti.pt"
    )

    path = Path(
        os.getenv(
            "PPO_CHECKPOINT",
            str(default_path),
        )
    ).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            "Missing trained PPO checkpoint: "
            f"{path}"
        )

    return path


# ============================================================
# CONTROLLED NORMALIZED STATE BANK
#
# This intentionally follows the state distribution used by
# the repo's earlier actor diagnostics, but the FINAL trained
# checkpoint is used here.
#
# These are policy-function diagnostics, not end-to-end
# network KPI experiments.
# ============================================================

def build_states(
    *,
    num_states: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:

    segments = torch.zeros(
        (
            num_states,
            NUM_CANDIDATES,
            FEATURES_PER_CANDIDATE,
        ),
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # Past average throughput
    # --------------------------------------------------------

    segments[
        :,
        :,
        0,
    ] = (
        0.05
        + 0.90
        * torch.rand(
            (
                num_states,
                NUM_CANDIDATES,
            ),
            device=device,
            generator=generator,
        )
    )


    # --------------------------------------------------------
    # Normalized rank: rank 1 or rank 2
    # --------------------------------------------------------

    rank = torch.randint(
        low=1,
        high=3,
        size=(
            num_states,
            NUM_CANDIDATES,
        ),
        device=device,
        generator=generator,
    )

    segments[
        :,
        :,
        1,
    ] = (
        rank.float()
        / 2.0
    )


    # --------------------------------------------------------
    # Normalized allocated RBG count
    # --------------------------------------------------------

    allocated = torch.randint(
        low=0,
        high=NUM_RBGS + 1,
        size=(
            num_states,
            NUM_CANDIDATES,
        ),
        device=device,
        generator=generator,
    )

    segments[
        :,
        :,
        2,
    ] = (
        allocated.float()
        / float(NUM_RBGS)
    )


    # --------------------------------------------------------
    # Buffer status
    #
    # Mix finite buffers and full-buffer proxy = 1.
    # --------------------------------------------------------

    finite_buffer = torch.rand(
        (
            num_states,
            NUM_CANDIDATES,
        ),
        device=device,
        generator=generator,
    )

    full_buffer_mask = (
        torch.rand(
            (
                num_states,
                NUM_CANDIDATES,
            ),
            device=device,
            generator=generator,
        )
        < 0.5
    )

    segments[
        :,
        :,
        3,
    ] = torch.where(
        full_buffer_mask,
        torch.ones_like(
            finite_buffer
        ),
        finite_buffer,
    )


    # --------------------------------------------------------
    # Wideband CQI
    # --------------------------------------------------------

    wideband_cqi = torch.rand(
        (
            num_states,
            NUM_CANDIDATES,
        ),
        device=device,
        generator=generator,
    )

    segments[
        :,
        :,
        4,
    ] = wideband_cqi


    # --------------------------------------------------------
    # Subband CQI
    # --------------------------------------------------------

    subband_cqi = (
        wideband_cqi.unsqueeze(
            -1
        )
        + 0.15
        * torch.randn(
            (
                num_states,
                NUM_CANDIDATES,
                NUM_RBGS,
            ),
            device=device,
            generator=generator,
        )
    ).clamp(
        min=0.0,
        max=1.0,
    )

    segments[
        :,
        :,
        5:
        5 + NUM_RBGS,
    ] = subband_cqi


    # --------------------------------------------------------
    # Max precoder cross-correlation
    # --------------------------------------------------------

    segments[
        :,
        :,
        5 + NUM_RBGS:
        5 + 2 * NUM_RBGS,
    ] = torch.rand(
        (
            num_states,
            NUM_CANDIDATES,
            NUM_RBGS,
        ),
        device=device,
        generator=generator,
    )


    return segments.reshape(
        num_states,
        STATE_SIZE,
    )


def build_action_mask(
    *,
    num_states: int,
    device: torch.device,
) -> torch.Tensor:

    # All ten candidates are deliberately valid in this
    # controlled policy-function diagnostic.

    return torch.ones(
        (
            num_states,
            NUM_RBGS,
            NUM_ACTIONS,
        ),
        dtype=torch.bool,
        device=device,
    )


# ============================================================
# POLICY OUTPUTS
# ============================================================

@torch.no_grad()
def policy_outputs(
    *,
    actor: OneLDSPPOActor,
    states: torch.Tensor,
    action_mask: torch.Tensor,
) -> dict[
    str,
    torch.Tensor,
]:

    (
        distribution,
        _,
    ) = actor.build_distribution(
        state=states,
        action_mask=action_mask,
    )

    probabilities = (
        distribution.probs
    )

    top2 = probabilities.topk(
        k=2,
        dim=-1,
    ).values

    return {
        "probabilities":
            probabilities,

        "actions":
            probabilities.argmax(
                dim=-1
            ),

        "margin":
            (
                top2[
                    ...,
                    0,
                ]
                -
                top2[
                    ...,
                    1,
                ]
            ),

        "normalized_entropy":
            (
                distribution.entropy()
                / math.log(
                    NUM_ACTIONS
                )
            ),
    }


def compare_policies(
    *,
    reference: dict[
        str,
        torch.Tensor,
    ],
    changed: dict[
        str,
        torch.Tensor,
    ],
) -> dict[
    str,
    float,
]:

    p = (
        reference[
            "probabilities"
        ]
    )

    q = (
        changed[
            "probabilities"
        ]
    )

    tiny = torch.finfo(
        p.dtype
    ).tiny

    p_safe = p.clamp_min(
        tiny
    )

    q_safe = q.clamp_min(
        tiny
    )

    mixture = (
        0.5
        * (
            p_safe
            + q_safe
        )
    )

    jsd = (
        0.5
        * (
            p_safe
            * (
                p_safe.log()
                - mixture.log()
            )
        ).sum(
            dim=-1
        )
        +
        0.5
        * (
            q_safe
            * (
                q_safe.log()
                - mixture.log()
            )
        ).sum(
            dim=-1
        )
    )

    agreement = (
        reference[
            "actions"
        ]
        ==
        changed[
            "actions"
        ]
    )

    total_variation = (
        0.5
        * (
            p
            - q
        )
        .abs()
        .sum(
            dim=-1
        )
    )

    max_probability_shift = (
        (
            p
            - q
        )
        .abs()
        .amax(
            dim=-1
        )
    )

    return {
        "mean_tv":
            float(
                total_variation
                .mean()
                .item()
            ),

        "mean_jsd":
            float(
                jsd
                .mean()
                .item()
            ),

        "mean_max_prob_shift":
            float(
                max_probability_shift
                .mean()
                .item()
            ),

        "rbg_agreement_pct":
            float(
                100.0
                * agreement
                .float()
                .mean()
                .item()
            ),

        "whole_18rbg_agreement_pct":
            float(
                100.0
                * agreement
                .all(
                    dim=-1
                )
                .float()
                .mean()
                .item()
            ),
    }


# ============================================================
# CSV
# ============================================================

def write_csv(
    path: Path,
    rows: list[
        dict[
            str,
            object,
        ]
    ],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(
                    key
                )
                seen.add(
                    key
                )

    with path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# EXPERIMENT 1
# CANDIDATE PERMUTATION EQUIVARIANCE
# ============================================================

def evaluate_permutation(
    *,
    actor: OneLDSPPOActor,
    states: torch.Tensor,
    action_mask: torch.Tensor,
    reference: dict[
        str,
        torch.Tensor,
    ],
    repeats: int,
    generator: torch.Generator,
) -> list[
    dict[
        str,
        object,
    ]
]:

    metric_sums = {}

    num_states = (
        states.shape[
            0
        ]
    )

    segments = states.reshape(
        num_states,
        NUM_CANDIDATES,
        FEATURES_PER_CANDIDATE,
    )

    for _ in range(
        repeats
    ):

        # One independent candidate permutation
        # for every state.

        permutation = (
            torch.rand(
                (
                    num_states,
                    NUM_CANDIDATES,
                ),
                device=states.device,
                generator=generator,
            )
            .argsort(
                dim=-1
            )
        )

        gather_index = (
            permutation
            .unsqueeze(
                -1
            )
            .expand(
                num_states,
                NUM_CANDIDATES,
                FEATURES_PER_CANDIDATE,
            )
        )

        permuted_states = (
            torch.gather(
                segments,
                dim=1,
                index=gather_index,
            )
            .reshape(
                num_states,
                STATE_SIZE,
            )
        )

        changed = policy_outputs(
            actor=actor,
            states=permuted_states,
            action_mask=action_mask,
        )


        # ----------------------------------------------------
        # Restore candidate probability indices back to
        # ORIGINAL PHYSICAL CANDIDATE identity.
        #
        # NO-ALLOCATION remains the final action.
        # ----------------------------------------------------

        restored_candidate_probs = (
            torch.empty_like(
                changed[
                    "probabilities"
                ][
                    :,
                    :,
                    :NUM_CANDIDATES,
                ]
            )
        )

        scatter_index = (
            permutation
            .unsqueeze(
                1
            )
            .expand(
                num_states,
                NUM_RBGS,
                NUM_CANDIDATES,
            )
        )

        restored_candidate_probs.scatter_(
            dim=2,
            index=scatter_index,
            src=(
                changed[
                    "probabilities"
                ][
                    :,
                    :,
                    :NUM_CANDIDATES,
                ]
            ),
        )

        restored_probs = torch.cat(
            (
                restored_candidate_probs,

                changed[
                    "probabilities"
                ][
                    :,
                    :,
                    NUM_CANDIDATES:,
                ],
            ),
            dim=-1,
        )

        changed_restored = {
            **changed,

            "probabilities":
                restored_probs,

            "actions":
                restored_probs.argmax(
                    dim=-1
                ),
        }

        metrics = compare_policies(
            reference=reference,
            changed=changed_restored,
        )

        for key, value in (
            metrics.items()
        ):
            metric_sums[
                key
            ] = (
                metric_sums.get(
                    key,
                    0.0,
                )
                + value
            )


    return [
        {
            "condition":
                "candidate_permutation",

            "repeats":
                repeats,

            "mean_reference_margin":
                float(
                    reference[
                        "margin"
                    ]
                    .mean()
                    .item()
                ),

            "mean_reference_entropy":
                float(
                    reference[
                        "normalized_entropy"
                    ]
                    .mean()
                    .item()
                ),

            **{
                key:
                    value
                    / repeats

                for key, value
                in metric_sums.items()
            },
        }
    ]


# ============================================================
# EXPERIMENT 2
# LOCAL OBSERVATION PERTURBATION
# ============================================================

def build_continuous_feature_mask(
    *,
    device: torch.device,
) -> torch.Tensor:

    single_candidate = torch.zeros(
        FEATURES_PER_CANDIDATE,
        dtype=torch.bool,
        device=device,
    )

    # Continuous features.
    #
    # Intentionally NOT perturb:
    #     rank
    #     allocated-RBG count
    #
    # because those are discrete structural variables.

    single_candidate[
        0
    ] = True

    single_candidate[
        3
    ] = True

    single_candidate[
        4
    ] = True

    single_candidate[
        5:
        5 + 2 * NUM_RBGS
    ] = True

    return (
        single_candidate
        .repeat(
            NUM_CANDIDATES
        )
        .view(
            1,
            -1,
        )
    )


def evaluate_perturbation(
    *,
    actor: OneLDSPPOActor,
    states: torch.Tensor,
    action_mask: torch.Tensor,
    reference: dict[
        str,
        torch.Tensor,
    ],
    repeats: int,
    generator: torch.Generator,
) -> tuple[
    list[
        dict[
            str,
            object,
        ]
    ],
    list[
        dict[
            str,
            object,
        ]
    ],
]:

    rows = []

    confidence_rows = []

    continuous_mask = (
        build_continuous_feature_mask(
            device=states.device,
        )
    )


    for epsilon in (
        PERTURBATION_EPSILONS
    ):

        metric_sums = {}

        flip_counts = (
            torch.zeros_like(
                reference[
                    "margin"
                ]
            )
        )


        for _ in range(
            repeats
        ):

            noise = (
                (
                    2.0
                    * torch.rand(
                        states.shape,
                        device=states.device,
                        generator=generator,
                    )
                    - 1.0
                )
                * epsilon
            )

            changed_states = (
                torch.where(
                    continuous_mask,

                    (
                        states
                        + noise
                    ).clamp(
                        min=0.0,
                        max=1.0,
                    ),

                    states,
                )
            )

            changed = policy_outputs(
                actor=actor,
                states=changed_states,
                action_mask=action_mask,
            )

            metrics = compare_policies(
                reference=reference,
                changed=changed,
            )

            for key, value in (
                metrics.items()
            ):

                metric_sums[
                    key
                ] = (
                    metric_sums.get(
                        key,
                        0.0,
                    )
                    + value
                )

            flip_counts += (
                reference[
                    "actions"
                ]
                !=
                changed[
                    "actions"
                ]
            ).float()


        rows.append(
            {
                "condition":
                    (
                        "linf_uniform_"
                        f"eps_{epsilon:.3f}"
                    ),

                "epsilon":
                    epsilon,

                "repeats":
                    repeats,

                "mean_reference_margin":
                    float(
                        reference[
                            "margin"
                        ]
                        .mean()
                        .item()
                    ),

                "mean_reference_entropy":
                    float(
                        reference[
                            "normalized_entropy"
                        ]
                        .mean()
                        .item()
                    ),

                **{
                    key:
                        value
                        / repeats

                    for key, value
                    in metric_sums.items()
                },
            }
        )


        # ----------------------------------------------------
        # DERIVED ANALYSIS:
        #
        # Does low clean-policy confidence predict
        # perturbation-induced action instability?
        #
        # This costs no additional experiment.
        # ----------------------------------------------------

        for (
            label,
            lower,
            upper,
        ) in CONFIDENCE_BINS:

            if upper is None:

                bin_mask = (
                    reference[
                        "margin"
                    ]
                    >= lower
                )

            else:

                bin_mask = (
                    (
                        reference[
                            "margin"
                        ]
                        >= lower
                    )
                    &
                    (
                        reference[
                            "margin"
                        ]
                        < upper
                    )
                )


            count = int(
                bin_mask
                .sum()
                .item()
            )


            if count == 0:

                flip_rate = (
                    float(
                        "nan"
                    )
                )

            else:

                total_flips = float(
                    flip_counts[
                        bin_mask
                    ]
                    .sum()
                    .item()
                )

                flip_rate = (
                    100.0
                    * total_flips
                    / (
                        count
                        * repeats
                    )
                )


            confidence_rows.append(
                {
                    "epsilon":
                        epsilon,

                    "confidence_bin":
                        label,

                    "num_decisions":
                        count,

                    "flip_rate_pct":
                        flip_rate,
                }
            )


    return (
        rows,
        confidence_rows,
    )


# ============================================================
# EXPERIMENT 3
# FEATURE-GROUP ABLATION
#
# IMPORTANT:
# Mean-replacement sensitivity is a diagnostic.
# Do NOT call it causal feature importance.
# ============================================================

def evaluate_ablation(
    *,
    actor: OneLDSPPOActor,
    states: torch.Tensor,
    action_mask: torch.Tensor,
    reference: dict[
        str,
        torch.Tensor,
    ],
) -> list[
    dict[
        str,
        object,
    ]
]:

    num_states = (
        states.shape[
            0
        ]
    )

    segments = states.reshape(
        num_states,
        NUM_CANDIDATES,
        FEATURES_PER_CANDIDATE,
    )

    neutral = (
        segments.mean(
            dim=(
                0,
                1,
            ),
            keepdim=True,
        )
    )

    rows = []


    for (
        group_name,
        (
            start,
            end,
        ),
    ) in FEATURE_GROUPS.items():

        ablated = (
            segments.clone()
        )

        ablated[
            :,
            :,
            start:end,
        ] = neutral[
            :,
            :,
            start:end,
        ]

        changed = policy_outputs(
            actor=actor,

            states=(
                ablated.reshape(
                    num_states,
                    STATE_SIZE,
                )
            ),

            action_mask=action_mask,
        )

        rows.append(
            {
                "condition":
                    group_name,

                "ablation":
                    "global_mean_replacement",

                "mean_reference_margin":
                    float(
                        reference[
                            "margin"
                        ]
                        .mean()
                        .item()
                    ),

                "mean_reference_entropy":
                    float(
                        reference[
                            "normalized_entropy"
                        ]
                        .mean()
                        .item()
                    ),

                **compare_policies(
                    reference=reference,
                    changed=changed,
                ),
            }
        )


    return rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parsed = parse_args()

    (
        evaluation,
        replicate_index,
        seed,
    ) = task_spec(
        parsed.task_id
    )


    device = torch.device(
        os.getenv(
            "RL_EVAL_DEVICE",
            "cuda:0",
        )
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )


    num_states = int(
        os.getenv(
            "RL_EVAL_NUM_STATES",
            "8192",
        )
    )

    repeats = int(
        os.getenv(
            "RL_EVAL_REPEATS",
            "8",
        )
    )


    run_id = os.getenv(
        "RL_EVAL_RUN_ID",
        "manual",
    )

    output_root = Path(
        os.getenv(
            "RL_EVAL_OUTPUT_ROOT",

            str(
                Path.home()
                / "oran_experiments"
                / "rl_eval"
            ),
        )
    )

    run_dir = (
        output_root
        / run_id
    )

    checkpoint_path = (
        resolve_checkpoint()
    )


    # --------------------------------------------------------
    # Reproducible controlled state/perturbation generator
    # --------------------------------------------------------

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        seed
    )


    # --------------------------------------------------------
    # Frozen final actor
    # --------------------------------------------------------

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig()
    ).to(
        device
    )

    checkpoint = (
        load_ppo_model_checkpoint(
            path=checkpoint_path,
            actor=actor,
            map_location=device,
        )
    )

    actor.eval()


    checkpoint_tti = (
        checkpoint.get(
            "tti_index"
        )
    )

    if (
        checkpoint_tti is not None
        and int(
            checkpoint_tti
        ) < 400
    ):
        raise RuntimeError(
            "Refusing to evaluate an early "
            "smoke checkpoint. "
            f"tti_index={checkpoint_tti}"
        )


    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats(
            device
        )

        torch.cuda.synchronize(
            device
        )


    start_time = (
        time.perf_counter()
    )


    states = build_states(
        num_states=num_states,
        device=device,
        generator=generator,
    )

    action_mask = (
        build_action_mask(
            num_states=num_states,
            device=device,
        )
    )

    reference = policy_outputs(
        actor=actor,
        states=states,
        action_mask=action_mask,
    )


    confidence_rows = []


    if evaluation == "permutation":

        result_rows = (
            evaluate_permutation(
                actor=actor,
                states=states,
                action_mask=action_mask,
                reference=reference,
                repeats=repeats,
                generator=generator,
            )
        )


    elif evaluation == "perturbation":

        (
            result_rows,
            confidence_rows,
        ) = evaluate_perturbation(
            actor=actor,
            states=states,
            action_mask=action_mask,
            reference=reference,
            repeats=repeats,
            generator=generator,
        )


    elif evaluation == "ablation":

        result_rows = (
            evaluate_ablation(
                actor=actor,
                states=states,
                action_mask=action_mask,
                reference=reference,
            )
        )


    else:

        raise AssertionError(
            evaluation
        )


    if device.type == "cuda":

        torch.cuda.synchronize(
            device
        )

        peak_gpu_mib = (
            torch.cuda
            .max_memory_allocated(
                device
            )
            / (
                1024.0
                ** 2
            )
        )

    else:

        peak_gpu_mib = 0.0


    elapsed = (
        time.perf_counter()
        - start_time
    )


    actor_norm = math.sqrt(
        sum(
            float(
                parameter
                .detach()
                .pow(
                    2
                )
                .sum()
                .item()
            )

            for parameter
            in actor.parameters()
        )
    )


    common = {
        "run_id":
            run_id,

        "task_id":
            parsed.task_id,

        "evaluation":
            evaluation,

        "replicate":
            replicate_index,

        "seed":
            seed,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_tti":
            checkpoint_tti,

        "ppo_updates":
            checkpoint.get(
                "num_ppo_updates"
            ),

        "actor_norm":
            actor_norm,

        "device":
            str(
                device
            ),

        "num_states":
            num_states,

        "elapsed_s":
            elapsed,

        "peak_gpu_mib":
            peak_gpu_mib,
    }


    result_rows = [
        {
            **common,
            **row,
        }

        for row
        in result_rows
    ]


    output_path = (
        run_dir
        / (
            f"task_{parsed.task_id:02d}_"
            f"{evaluation}_"
            f"seed_{seed}.csv"
        )
    )

    write_csv(
        output_path,
        result_rows,
    )


    if confidence_rows:

        confidence_rows = [
            {
                **common,
                **row,
            }

            for row
            in confidence_rows
        ]

        confidence_path = (
            run_dir
            / (
                f"task_{parsed.task_id:02d}_"
                f"{evaluation}_"
                f"seed_{seed}_"
                "confidence.csv"
            )
        )

        write_csv(
            confidence_path,
            confidence_rows,
        )

    else:

        confidence_path = None


    print(
        "=" * 72
    )

    print(
        "FROZEN PPO ACTOR RL EVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Run ID:          {run_id}"
    )

    print(
        f"Task:            {parsed.task_id}"
    )

    print(
        f"Evaluation:      {evaluation}"
    )

    print(
        f"Replicate:       {replicate_index}"
    )

    print(
        f"Seed:            {seed}"
    )

    print(
        f"Checkpoint:      {checkpoint_path}"
    )

    print(
        f"Checkpoint TTI:  {checkpoint_tti}"
    )

    print(
        "PPO updates:     "
        f"{checkpoint.get('num_ppo_updates')}"
    )

    print(
        f"Actor norm:      {actor_norm:.8f}"
    )

    print(
        f"States:          {num_states}"
    )

    print(
        f"Elapsed:         {elapsed:.3f} s"
    )

    print(
        f"Peak GPU MiB:    {peak_gpu_mib:.1f}"
    )

    print(
        f"Output:          {output_path}"
    )

    if confidence_path is not None:

        print(
            f"Confidence:      {confidence_path}"
        )


    print(
        "=" * 72
    )

    print(
        "RL_ACTOR_EVAL_PASS"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
