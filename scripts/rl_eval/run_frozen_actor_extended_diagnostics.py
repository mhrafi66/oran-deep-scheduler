from __future__ import annotations

import argparse
import csv
import importlib.util
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
# Reuse the state-generation / policy-comparison machinery
# from RL evaluation Set 1.
# ============================================================

BASE_PATH = Path(__file__).with_name(
    "run_frozen_actor_diagnostics.py"
)

spec = importlib.util.spec_from_file_location(
    "rl_eval_base",
    BASE_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not load {BASE_PATH}"
    )

base = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    base
)


K = base.NUM_CANDIDATES
M = base.NUM_RBGS
F = base.FEATURES_PER_CANDIDATE
A = base.NUM_ACTIONS
S = base.STATE_SIZE


# ============================================================
# LOCKED SECOND RL CAMPAIGN
#
# 5 experiment families
# x 5 independent seeds
# =
# 25 Slurm tasks
# ============================================================

EVALUATIONS = (
    "adversarial",
    "distribution_shift",
    "action_mask",
    "temporal_smoothness",
    "symmetry_bias",
)

N_REPLICATES = 5

BASE_SEED = 202609240


ADVERSARIAL_EPSILONS = (
    0.01,
    0.02,
    0.05,
    0.10,
)


TEMPORAL_SIGMAS = (
    0.0025,
    0.0050,
    0.0100,
    0.0200,
)


VALID_CANDIDATE_COUNTS = (
    10,
    8,
    6,
    4,
    2,
)


FEATURE_GROUPS = {
    "past_throughput":
        (
            0,
            1,
        ),

    "rank":
        (
            1,
            2,
        ),

    "allocated_rbg_count":
        (
            2,
            3,
        ),

    "buffer":
        (
            3,
            4,
        ),

    "wideband_cqi":
        (
            4,
            5,
        ),

    "subband_cqi":
        (
            5,
            5 + M,
        ),

    "cross_correlation":
        (
            5 + M,
            5 + 2 * M,
        ),
}


# ============================================================
# CLI / TASK MAPPING
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-id",
        type=int,
        default=int(
            os.getenv(
                "RL_EVAL2_TASK_ID",
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

    total = (
        len(EVALUATIONS)
        * N_REPLICATES
    )

    if not 0 <= task_id < total:
        raise ValueError(
            f"task-id must be in [0, {total - 1}]"
        )

    family = EVALUATIONS[
        task_id
        // N_REPLICATES
    ]

    replicate = (
        task_id
        % N_REPLICATES
    )

    seed = (
        BASE_SEED
        + replicate
    )

    return (
        family,
        replicate,
        seed,
    )


# ============================================================
# BASIC HELPERS
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

    fields = []

    seen = set()

    for row in rows:

        for key in row:

            if key not in seen:

                fields.append(
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
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def l2_norm(
    module: torch.nn.Module,
) -> float:

    return math.sqrt(
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
            in module.parameters()
        )
    )


def load_actor(
    *,
    device: torch.device,
):

    checkpoint_path = (
        base.resolve_checkpoint()
    )

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

    return (
        actor,
        checkpoint,
        checkpoint_path,
    )


def entropy(
    probabilities: torch.Tensor,
) -> torch.Tensor:

    probabilities = (
        probabilities.clamp_min(
            torch.finfo(
                probabilities.dtype
            ).tiny
        )
    )

    return -(
        probabilities
        * probabilities.log()
    ).sum(
        dim=-1
    )


def continuous_mask(
    *,
    device: torch.device,
) -> torch.Tensor:

    return (
        base
        .build_continuous_feature_mask(
            device=device,
        )
    )


def compare(
    reference,
    changed,
) -> dict[
    str,
    float,
]:

    metrics = (
        base.compare_policies(
            reference=reference,
            changed=changed,
        )
    )

    return {
        **metrics,

        "rbg_flip_pct":
            (
                100.0
                - metrics[
                    "rbg_agreement_pct"
                ]
            ),

        "whole_schedule_change_pct":
            (
                100.0
                - metrics[
                    "whole_18rbg_agreement_pct"
                ]
            ),
    }


# ============================================================
# EXPERIMENT 1
#
# WHITE-BOX BOUNDED OBSERVATION ATTACK
#
# Objective:
# reduce the probability assigned to the clean greedy action.
#
# Only continuous normalized state variables are attacked.
#
# Rank and allocated-RBG count remain unchanged because they
# are discrete structural state components.
# ============================================================

def reference_action_gradient(
    *,
    actor,
    states,
    action_mask,
    reference_actions,
):

    attacked = (
        states
        .detach()
        .clone()
        .requires_grad_(
            True
        )
    )

    (
        distribution,
        _,
    ) = actor.build_distribution(
        state=attacked,
        action_mask=action_mask,
    )

    selected_probability = (
        distribution
        .probs
        .gather(
            dim=-1,
            index=(
                reference_actions
                .unsqueeze(
                    -1
                )
            ),
        )
        .squeeze(
            -1
        )
        .clamp_min(
            1.0e-8
        )
    )

    loss = -(
        selected_probability.log()
    ).mean()

    gradient = torch.autograd.grad(
        loss,
        attacked,
        retain_graph=False,
        create_graph=False,
    )[0]

    return (
        float(
            loss.item()
        ),
        gradient.detach(),
    )


def evaluate_adversarial(
    *,
    actor,
    states,
    action_mask,
    reference,
    pgd_steps,
):

    cont_mask = continuous_mask(
        device=states.device,
    )

    float_mask = (
        cont_mask.to(
            states.dtype
        )
    )

    (
        clean_loss,
        clean_gradient,
    ) = reference_action_gradient(
        actor=actor,
        states=states,
        action_mask=action_mask,
        reference_actions=(
            reference[
                "actions"
            ]
        ),
    )


    # --------------------------------------------------------
    # Local gradient sensitivity by feature family.
    #
    # This is a diagnostic sensitivity metric.
    # It is NOT claimed to be causal feature importance.
    # --------------------------------------------------------

    gradient_segments = (
        clean_gradient
        .abs()
        .reshape(
            states.shape[
                0
            ],
            K,
            F,
        )
    )

    gradient_summary = {
        (
            f"grad_abs_{name}"
        ):
            float(
                gradient_segments[
                    :,
                    :,
                    start:end,
                ]
                .mean()
                .item()
            )

        for (
            name,
            (
                start,
                end,
            ),
        )
        in FEATURE_GROUPS.items()
    }


    rows = []


    for epsilon in (
        ADVERSARIAL_EPSILONS
    ):

        attacked = (
            states
            .detach()
            .clone()
        )

        step_size = (
            epsilon
            / max(
                1,
                pgd_steps // 2,
            )
        )


        for _ in range(
            pgd_steps
        ):

            attacked = (
                attacked
                .detach()
                .requires_grad_(
                    True
                )
            )

            (
                distribution,
                _,
            ) = (
                actor
                .build_distribution(
                    state=attacked,
                    action_mask=action_mask,
                )
            )

            selected_probability = (
                distribution
                .probs
                .gather(
                    dim=-1,
                    index=(
                        reference[
                            "actions"
                        ]
                        .unsqueeze(
                            -1
                        )
                    ),
                )
                .squeeze(
                    -1
                )
                .clamp_min(
                    1.0e-8
                )
            )

            attack_loss = -(
                selected_probability
                .log()
            ).mean()

            gradient = (
                torch.autograd.grad(
                    attack_loss,
                    attacked,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            )

            proposal = (
                attacked
                + step_size
                * gradient.sign()
                * float_mask
            )

            delta = (
                proposal
                - states
            ).clamp(
                min=-epsilon,
                max=epsilon,
            )

            attacked = (
                torch.where(
                    cont_mask,

                    (
                        states
                        + delta
                    ).clamp(
                        min=0.0,
                        max=1.0,
                    ),

                    states,
                )
                .detach()
            )


        changed = (
            base.policy_outputs(
                actor=actor,
                states=attacked,
                action_mask=action_mask,
            )
        )

        rows.append(
            {
                "condition":
                    (
                        "pgd_reference_action_"
                        f"eps_{epsilon:.3f}"
                    ),

                "epsilon":
                    epsilon,

                "pgd_steps":
                    pgd_steps,

                "clean_reference_action_nll":
                    clean_loss,

                **compare(
                    reference,
                    changed,
                ),

                **gradient_summary,
            }
        )


    return rows


# ============================================================
# EXPERIMENT 2
#
# CONTROLLED DISTRIBUTION / COMPOSITION SHIFTS
#
# These measure actor behavior under shifted state
# distributions.
#
# They are NOT end-to-end network-performance claims.
# ============================================================

def shifted_state(
    *,
    segments,
    profile,
):

    changed = (
        segments.clone()
    )

    wideband = 4

    subband_start = 5

    subband_end = (
        5
        + M
    )

    correlation_start = (
        5
        + M
    )

    correlation_end = (
        5
        + 2 * M
    )


    if profile == "cqi_low":

        changed[
            :,
            :,
            wideband,
        ] *= 0.25

        changed[
            :,
            :,
            subband_start:
            subband_end,
        ] *= 0.25


    elif profile == "cqi_high":

        changed[
            :,
            :,
            wideband,
        ] = (
            0.75
            + 0.25
            * changed[
                :,
                :,
                wideband,
            ]
        )

        changed[
            :,
            :,
            subband_start:
            subband_end,
        ] = (
            0.75
            + 0.25
            * changed[
                :,
                :,
                subband_start:
                subband_end,
            ]
        )


    elif profile == "spatial_corr_high":

        changed[
            :,
            :,
            correlation_start:
            correlation_end,
        ] = (
            0.75
            + 0.25
            * changed[
                :,
                :,
                correlation_start:
                correlation_end,
            ]
        )


    elif profile == "buffers_near_empty":

        changed[
            :,
            :,
            3,
        ] *= 0.05


    elif profile == "buffers_full":

        changed[
            :,
            :,
            3,
        ] = 1.0


    elif profile == "rank_all_1":

        changed[
            :,
            :,
            1,
        ] = 0.5


    elif profile == "rank_all_2":

        changed[
            :,
            :,
            1,
        ] = 1.0


    elif profile == "history_low":

        changed[
            :,
            :,
            0,
        ] = (
            0.05
            + 0.15
            * changed[
                :,
                :,
                0,
            ]
        )


    elif profile == "cqi_history_conflict":

        changed[
            :,
            :,
            wideband,
        ] = (
            1.0
            - changed[
                :,
                :,
                wideband,
            ]
        )

        changed[
            :,
            :,
            subband_start:
            subband_end,
        ] = (
            1.0
            - changed[
                :,
                :,
                subband_start:
                subband_end,
            ]
        )


    elif profile == "joint_hard_shift":

        changed[
            :,
            :,
            wideband,
        ] *= 0.25

        changed[
            :,
            :,
            subband_start:
            subband_end,
        ] *= 0.25

        changed[
            :,
            :,
            correlation_start:
            correlation_end,
        ] = (
            0.75
            + 0.25
            * changed[
                :,
                :,
                correlation_start:
                correlation_end,
            ]
        )

        changed[
            :,
            :,
            3,
        ] = 1.0

        changed[
            :,
            :,
            1,
        ] = 0.5


    else:

        raise ValueError(
            profile
        )


    return changed


def evaluate_distribution_shift(
    *,
    actor,
    states,
    action_mask,
    reference,
):

    profiles = (
        "cqi_low",
        "cqi_high",
        "spatial_corr_high",
        "buffers_near_empty",
        "buffers_full",
        "rank_all_1",
        "rank_all_2",
        "history_low",
        "cqi_history_conflict",
        "joint_hard_shift",
    )


    segments = states.reshape(
        states.shape[
            0
        ],
        K,
        F,
    )


    rows = []


    for profile in profiles:

        changed_state = (
            shifted_state(
                segments=segments,
                profile=profile,
            )
            .reshape(
                states.shape[
                    0
                ],
                S,
            )
        )

        changed = (
            base.policy_outputs(
                actor=actor,
                states=changed_state,
                action_mask=action_mask,
            )
        )


        rows.append(
            {
                "condition":
                    profile,

                "changed_mean_margin":
                    float(
                        changed[
                            "margin"
                        ]
                        .mean()
                        .item()
                    ),

                "changed_mean_normalized_entropy":
                    float(
                        changed[
                            "normalized_entropy"
                        ]
                        .mean()
                        .item()
                    ),

                "changed_no_allocation_argmax_pct":
                    float(
                        100.0
                        * (
                            changed[
                                "actions"
                            ]
                            == K
                        )
                        .float()
                        .mean()
                        .item()
                    ),

                **compare(
                    reference,
                    changed,
                ),
            }
        )


    return rows


# ============================================================
# EXPERIMENT 3
#
# STATE-DEPENDENT ACTION-MASK / CANDIDATE-SET ROBUSTNESS
# ============================================================

def evaluate_action_mask(
    *,
    actor,
    states,
    reference,
    generator,
):

    num_states = states.shape[
        0
    ]

    rows = []


    for valid_count in (
        VALID_CANDIDATE_COUNTS
    ):

        random_order = (
            torch.rand(
                (
                    num_states,
                    K,
                ),
                device=states.device,
                generator=generator,
            )
            .argsort(
                dim=-1
            )
        )


        candidate_valid = (
            torch.zeros(
                (
                    num_states,
                    K,
                ),
                dtype=torch.bool,
                device=states.device,
            )
        )

        candidate_valid.scatter_(
            dim=1,
            index=(
                random_order[
                    :,
                    :valid_count,
                ]
            ),
            value=True,
        )


        mask = torch.zeros(
            (
                num_states,
                M,
                A,
            ),
            dtype=torch.bool,
            device=states.device,
        )


        mask[
            :,
            :,
            :K,
        ] = (
            candidate_valid
            .unsqueeze(
                1
            )
            .expand(
                num_states,
                M,
                K,
            )
        )


        # No-allocation remains valid.
        mask[
            :,
            :,
            K,
        ] = True


        changed = (
            base.policy_outputs(
                actor=actor,
                states=states,
                action_mask=mask,
            )
        )

        probabilities = (
            changed[
                "probabilities"
            ]
        )


        invalid_mass = (
            probabilities
            .masked_fill(
                mask,
                0.0,
            )
            .sum(
                dim=-1
            )
        )


        reference_valid = (
            mask
            .gather(
                dim=-1,
                index=(
                    reference[
                        "actions"
                    ]
                    .unsqueeze(
                        -1
                    )
                ),
            )
            .squeeze(
                -1
            )
        )


        same_action = (
            changed[
                "actions"
            ]
            ==
            reference[
                "actions"
            ]
        )


        if bool(
            reference_valid.any()
        ):

            conditional_agreement = (
                float(
                    100.0
                    * same_action[
                        reference_valid
                    ]
                    .float()
                    .mean()
                    .item()
                )
            )

        else:

            conditional_agreement = (
                float(
                    "nan"
                )
            )


        support_entropy = (
            entropy(
                probabilities
            )
            / math.log(
                valid_count
                + 1
            )
        )


        rows.append(
            {
                "condition":
                    (
                        "valid_candidates_"
                        f"{valid_count}"
                    ),

                "valid_candidates":
                    valid_count,

                "reference_action_still_valid_pct":
                    float(
                        100.0
                        * reference_valid
                        .float()
                        .mean()
                        .item()
                    ),

                "agreement_given_reference_valid_pct":
                    conditional_agreement,

                "masked_invalid_probability_mass_max":
                    float(
                        invalid_mass
                        .max()
                        .item()
                    ),

                "support_normalized_entropy_mean":
                    float(
                        support_entropy
                        .mean()
                        .item()
                    ),

                "no_allocation_argmax_pct":
                    float(
                        100.0
                        * (
                            changed[
                                "actions"
                            ]
                            == K
                        )
                        .float()
                        .mean()
                        .item()
                    ),

                "no_allocation_probability_mean":
                    float(
                        probabilities[
                            :,
                            :,
                            K,
                        ]
                        .mean()
                        .item()
                    ),
            }
        )


    return rows


# ============================================================
# EXPERIMENT 4
#
# TEMPORAL SMOOTHNESS / ACTION CHATTERING
#
# Nearby consecutive observations are generated by a small
# random walk over continuous features.
# ============================================================

def evaluate_temporal(
    *,
    actor,
    states,
    action_mask,
    generator,
    trajectory_length,
    trajectory_count,
):

    count = min(
        trajectory_count,
        states.shape[
            0
        ],
    )

    initial_states = (
        states[
            :count
        ]
        .detach()
        .clone()
    )

    action_mask = (
        action_mask[
            :count
        ]
    )

    cont_mask = continuous_mask(
        device=states.device,
    )

    initial_output = (
        base.policy_outputs(
            actor=actor,
            states=initial_states,
            action_mask=action_mask,
        )
    )


    rows = []


    for sigma in (
        TEMPORAL_SIGMAS
    ):

        current_state = (
            initial_states.clone()
        )

        previous_output = (
            initial_output
        )

        total_tv = 0.0
        total_jsd = 0.0
        total_flip = 0.0
        total_schedule_change = 0.0


        for _ in range(
            trajectory_length
            - 1
        ):

            noise = (
                sigma
                * torch.randn(
                    current_state.shape,
                    device=states.device,
                    generator=generator,
                )
            )

            current_state = (
                torch.where(
                    cont_mask,

                    (
                        current_state
                        + noise
                    ).clamp(
                        min=0.0,
                        max=1.0,
                    ),

                    current_state,
                )
            )


            current_output = (
                base.policy_outputs(
                    actor=actor,
                    states=current_state,
                    action_mask=action_mask,
                )
            )


            metrics = compare(
                previous_output,
                current_output,
            )


            total_tv += (
                metrics[
                    "mean_tv"
                ]
            )

            total_jsd += (
                metrics[
                    "mean_jsd"
                ]
            )

            total_flip += (
                metrics[
                    "rbg_flip_pct"
                ]
            )

            total_schedule_change += (
                metrics[
                    "whole_schedule_change_pct"
                ]
            )


            previous_output = (
                current_output
            )


        denominator = float(
            trajectory_length
            - 1
        )


        final_metrics = compare(
            initial_output,
            previous_output,
        )


        rows.append(
            {
                "condition":
                    (
                        "gaussian_random_walk_"
                        f"sigma_{sigma:.4f}"
                    ),

                "sigma":
                    sigma,

                "trajectory_count":
                    count,

                "trajectory_length":
                    trajectory_length,

                "mean_adjacent_tv":
                    (
                        total_tv
                        / denominator
                    ),

                "mean_adjacent_jsd":
                    (
                        total_jsd
                        / denominator
                    ),

                "mean_adjacent_rbg_flip_pct":
                    (
                        total_flip
                        / denominator
                    ),

                "mean_adjacent_whole_schedule_change_pct":
                    (
                        total_schedule_change
                        / denominator
                    ),

                "final_vs_initial_tv":
                    (
                        final_metrics[
                            "mean_tv"
                        ]
                    ),

                "final_vs_initial_jsd":
                    (
                        final_metrics[
                            "mean_jsd"
                        ]
                    ),

                "final_vs_initial_rbg_flip_pct":
                    (
                        final_metrics[
                            "rbg_flip_pct"
                        ]
                    ),
            }
        )


    return rows


# ============================================================
# EXPERIMENT 5
#
# EXACT CANDIDATE-SYMMETRY / POSITIONAL BIAS
#
# All ten candidate segments are identical.
#
# If candidate position carries no learned bias, conditional
# candidate probabilities should approach a uniform vector.
# ============================================================

def evaluate_symmetry(
    *,
    actor,
    states,
    action_mask,
):

    num_states = states.shape[
        0
    ]


    segments = states.reshape(
        num_states,
        K,
        F,
    )


    prototype = (
        segments[
            :,
            0:1,
            :,
        ]
    )


    symmetric_state = (
        prototype
        .expand(
            num_states,
            K,
            F,
        )
        .clone()
        .reshape(
            num_states,
            S,
        )
    )


    output = (
        base.policy_outputs(
            actor=actor,
            states=symmetric_state,
            action_mask=action_mask,
        )
    )


    probabilities = (
        output[
            "probabilities"
        ]
    )


    candidate_probabilities = (
        probabilities[
            :,
            :,
            :K,
        ]
    )


    candidate_conditional = (
        candidate_probabilities
        / candidate_probabilities
        .sum(
            dim=-1,
            keepdim=True,
        )
        .clamp_min(
            1.0e-8
        )
    )


    conditional_entropy = (
        entropy(
            candidate_conditional
        )
        / math.log(
            K
        )
    )


    coefficient_of_variation = (
        candidate_probabilities
        .std(
            dim=-1,
            unbiased=False,
        )
        / candidate_probabilities
        .mean(
            dim=-1
        )
        .clamp_min(
            1.0e-8
        )
    )


    max_min_gap = (
        candidate_probabilities
        .amax(
            dim=-1
        )
        -
        candidate_probabilities
        .amin(
            dim=-1
        )
    )


    l1_from_uniform = (
        candidate_conditional
        - (
            1.0
            / K
        )
    ).abs().sum(
        dim=-1
    )


    candidate_actions = (
        output[
            "actions"
        ][
            output[
                "actions"
            ]
            < K
        ]
    )


    if (
        candidate_actions.numel()
        > 0
    ):

        position_shares = (
            torch.bincount(
                candidate_actions,
                minlength=K,
            )
            .float()
        )

        position_shares /= (
            position_shares.sum()
        )

        max_position_share = (
            float(
                100.0
                * position_shares
                .max()
                .item()
            )
        )

        position_entropy = (
            float(
                (
                    entropy(
                        position_shares
                        .unsqueeze(
                            0
                        )
                    )
                    / math.log(
                        K
                    )
                )
                .item()
            )
        )

    else:

        max_position_share = (
            float(
                "nan"
            )
        )

        position_entropy = (
            float(
                "nan"
            )
        )


    return [
        {
            "condition":
                "identical_candidate_segments",

            "candidate_conditional_entropy_mean":
                float(
                    conditional_entropy
                    .mean()
                    .item()
                ),

            "candidate_probability_cv_mean":
                float(
                    coefficient_of_variation
                    .mean()
                    .item()
                ),

            "candidate_probability_max_min_gap_mean":
                float(
                    max_min_gap
                    .mean()
                    .item()
                ),

            "candidate_conditional_l1_from_uniform_mean":
                float(
                    l1_from_uniform
                    .mean()
                    .item()
                ),

            "max_argmax_position_share_pct":
                max_position_share,

            "argmax_position_entropy_normalized":
                position_entropy,

            "no_allocation_argmax_pct":
                float(
                    100.0
                    * (
                        output[
                            "actions"
                        ]
                        == K
                    )
                    .float()
                    .mean()
                    .item()
                ),

            "no_allocation_probability_mean":
                float(
                    probabilities[
                        :,
                        :,
                        K,
                    ]
                    .mean()
                    .item()
                ),
        }
    ]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    config = parse_args()

    (
        family,
        replicate,
        seed,
    ) = task_spec(
        config.task_id
    )


    device = torch.device(
        os.getenv(
            "RL_EVAL2_DEVICE",
            "cuda:0",
        )
    )


    if (
        device.type
        == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )


    num_states = int(
        os.getenv(
            "RL_EVAL2_NUM_STATES",
            "2048",
        )
    )


    pgd_steps = int(
        os.getenv(
            "RL_EVAL2_PGD_STEPS",
            "7",
        )
    )


    trajectory_length = int(
        os.getenv(
            "RL_EVAL2_TRAJECTORY_LENGTH",
            "32",
        )
    )


    trajectory_count = int(
        os.getenv(
            "RL_EVAL2_TRAJECTORY_COUNT",
            "512",
        )
    )


    run_id = os.getenv(
        "RL_EVAL2_RUN_ID",
        "manual",
    )


    output_root = Path(
        os.getenv(
            "RL_EVAL2_OUTPUT_ROOT",

            str(
                Path.home()
                / "oran_experiments"
                / "rl_eval2"
            ),
        )
    )


    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        seed
    )


    (
        actor,
        checkpoint,
        checkpoint_path,
    ) = load_actor(
        device=device
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


    states = base.build_states(
        num_states=num_states,
        device=device,
        generator=generator,
    )


    action_mask = (
        base.build_action_mask(
            num_states=num_states,
            device=device,
        )
    )


    reference = (
        base.policy_outputs(
            actor=actor,
            states=states,
            action_mask=action_mask,
        )
    )


    if family == "adversarial":

        rows = evaluate_adversarial(
            actor=actor,
            states=states,
            action_mask=action_mask,
            reference=reference,
            pgd_steps=pgd_steps,
        )


    elif (
        family
        == "distribution_shift"
    ):

        rows = (
            evaluate_distribution_shift(
                actor=actor,
                states=states,
                action_mask=action_mask,
                reference=reference,
            )
        )


    elif family == "action_mask":

        rows = evaluate_action_mask(
            actor=actor,
            states=states,
            reference=reference,
            generator=generator,
        )


    elif (
        family
        == "temporal_smoothness"
    ):

        rows = evaluate_temporal(
            actor=actor,
            states=states,
            action_mask=action_mask,
            generator=generator,
            trajectory_length=(
                trajectory_length
            ),
            trajectory_count=(
                trajectory_count
            ),
        )


    elif (
        family
        == "symmetry_bias"
    ):

        rows = evaluate_symmetry(
            actor=actor,
            states=states,
            action_mask=action_mask,
        )


    else:

        raise AssertionError(
            family
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


    common = {
        "run_id":
            run_id,

        "task_id":
            config.task_id,

        "evaluation":
            family,

        "replicate":
            replicate,

        "seed":
            seed,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_tti":
            checkpoint.get(
                "tti_index"
            ),

        "ppo_updates":
            checkpoint.get(
                "num_ppo_updates"
            ),

        "actor_norm":
            l2_norm(
                actor
            ),

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


    rows = [
        {
            **common,
            **row,
        }

        for row in rows
    ]


    output_path = (
        output_root
        / run_id
        / (
            f"task_{config.task_id:02d}_"
            f"{family}_"
            f"seed_{seed}.csv"
        )
    )


    write_csv(
        output_path,
        rows,
    )


    print(
        "=" * 72
    )

    print(
        "FROZEN PPO ACTOR RL EVALUATION — SET 2"
    )

    print(
        "=" * 72
    )

    print(
        f"Run ID:          {run_id}"
    )

    print(
        f"Task:            {config.task_id}"
    )

    print(
        f"Evaluation:      {family}"
    )

    print(
        f"Replicate:       {replicate}"
    )

    print(
        f"Seed:            {seed}"
    )

    print(
        f"Checkpoint:      {checkpoint_path}"
    )

    print(
        "Checkpoint TTI:  "
        f"{checkpoint.get('tti_index')}"
    )

    print(
        "PPO updates:     "
        f"{checkpoint.get('num_ppo_updates')}"
    )

    print(
        "Actor norm:      "
        f"{l2_norm(actor):.8f}"
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

    print(
        "=" * 72
    )

    print(
        "RL_ACTOR_EVAL2_PASS"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
