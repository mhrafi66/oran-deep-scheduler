from dataclasses import dataclass

import torch

from oran_scheduler.phy.schedule_evaluator import (
    CellAllocationEvaluationData,
    evaluate_cell_allocation,
)
from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
    PPOScheduleGreedySearchData,
    evaluate_ppo_schedule_greedy_search,
)
from oran_scheduler.rl.ppo_physical_score import (
    CachedPPOPhysicalRBGScorer,
    PPOPhysicalScoreInputs,
)
from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
    PPORewardData,
    PPORewardReduction,
    compute_ppo_reward,
    reduce_ppo_rbg_rewards,
)
from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    CellAllocation,
    validate_cell_allocation,
)

@dataclass(frozen=True)
class PPOPhysicalTTIOutcome:
    """
    Physical and counterfactual-reward information
    for one completed PPO-scheduled TTI.

    allocation_evaluation:
        Real physical evaluation of the actor's final
        CellAllocation.

    candidate_total_target_compliant_rate_bps:
        Shape [candidate].

        Sum over RBGs of the common PHY's
        target-compliant rate for each PF-TDS
        candidate.

        IMPORTANT:
        This is physical service capacity in the
        current reproduction.

        For Full Buffer traffic it can later be used
        directly as delivered throughput.

        For finite-buffer / FTP traffic it must NOT
        automatically be interpreted as delivered
        throughput; the traffic/buffer model must
        determine how many bits were actually served.

    greedy_search:
        Counterfactual PF comparison for every
        [user_layer, RBG] actor decision.

        None only when counterfactual evaluation is
        deliberately skipped during pre-collection
        warm-up.

    num_score_requests:
        Number of score requests issued by the
        counterfactual greedy judge.

    num_unique_phy_evaluations:
        Number of unique physical candidate sets that
        actually required PHY evaluation after cache
        reuse.
    """

    allocation_evaluation: (
        CellAllocationEvaluationData
    )

    candidate_total_target_compliant_rate_bps: (
        torch.Tensor
    )

    greedy_search: (
        PPOScheduleGreedySearchData
        | None
    )

    num_score_requests: int

    num_unique_phy_evaluations: int


@dataclass(frozen=True)
class PPOResolvedTTIReward:
    """
    PPO reward after explicitly selecting the
    throughput population used to calculate G.

    reward_population_name:
        Human-readable reproducibility label such as:

            "candidate_ues"
            "serving_cell_ues"

        This string does not alter the calculation.
        It records which interpretation the caller
        chose.

    reward_data:
        Complete paper-defined reward information,
        including raw [layer, RBG] rewards.

    reduced_reward:
        Shape [layer].

        Scalar reward per joint 1LDS layer decision
        under the selected reduction rule.

        Primary reproduction currently uses "mean",
        which remains PAPER-INFERRED /
        OPEN-REPRODUCTION.
    """

    reward_population_name: str

    reward_data: PPORewardData

    reduced_reward: torch.Tensor


def _validate_physical_tti_inputs(
    *,
    allocation: CellAllocation,
    actions: torch.Tensor,
    past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    physical_inputs: PPOPhysicalScoreInputs,
) -> int:
    """
    Validate the scheduler-side inputs for one
    completed PPO TTI.

    Returns:
        Number of PF-TDS candidate slots.
    """

    candidate_global_ue_indices = (
        physical_inputs
        .candidate_global_ue_indices
    )

    if (
        candidate_global_ue_indices.ndim
        != 1
    ):
        raise ValueError(
            "candidate_global_ue_indices must "
            "have shape [candidate]."
        )

    num_candidates = int(
        candidate_global_ue_indices.shape[0]
    )

    if num_candidates < 1:
        raise ValueError(
            "At least one candidate slot is required."
        )

    validate_cell_allocation(
        allocation=allocation,
        num_candidates=num_candidates,
    )

    if actions.ndim != 2:
        raise ValueError(
            "actions must have shape "
            "[user_layer, RBG]."
        )

    if tuple(
        actions.shape
    ) != tuple(
        allocation
        .candidate_by_user_slot
        .shape
    ):
        raise ValueError(
            "actions and allocation must have the "
            "same [user_layer, RBG] shape."
        )

    if (
        torch.is_floating_point(
            actions
        )
        or actions.dtype
        == torch.bool
    ):
        raise ValueError(
            "actions must use an integer dtype."
        )

    if torch.any(
        actions < 0
    ):
        raise ValueError(
            "Actor actions cannot be negative."
        )

    if torch.any(
        actions > num_candidates
    ):
        raise ValueError(
            "Actor action exceeds the action space."
        )

    expected_allocation = torch.where(
        actions == num_candidates,
        torch.full_like(
            actions,
            fill_value=NO_ALLOCATION,
        ),
        actions,
    )

    if not torch.equal(
        expected_allocation,
        allocation
        .candidate_by_user_slot,
    ):
        raise ValueError(
            "Actor actions do not reconstruct the "
            "supplied final CellAllocation."
        )

    expected_candidate_shape = (
        num_candidates,
    )

    if tuple(
        candidate_valid_mask.shape
    ) != expected_candidate_shape:
        raise ValueError(
            "candidate_valid_mask must have shape "
            "[candidate]."
        )

    if candidate_valid_mask.dtype != torch.bool:
        raise ValueError(
            "candidate_valid_mask must use "
            "torch.bool."
        )

    if tuple(
        past_average_throughput.shape
    ) != expected_candidate_shape:
        raise ValueError(
            "past_average_throughput must have "
            "shape [candidate]."
        )

    if not torch.is_floating_point(
        past_average_throughput
    ):
        raise ValueError(
            "past_average_throughput must use a "
            "floating-point dtype."
        )

    if not torch.isfinite(
        past_average_throughput
    ).all():
        raise ValueError(
            "past_average_throughput contains "
            "non-finite values."
        )

    if torch.any(
        past_average_throughput < 0.0
    ):
        raise ValueError(
            "past_average_throughput cannot be "
            "negative."
        )

    scheduler_device = actions.device

    scheduler_tensors = {
        "allocation": (
            allocation
            .candidate_by_user_slot
        ),
        "candidate_valid_mask": (
            candidate_valid_mask
        ),
        "past_average_throughput": (
            past_average_throughput
        ),
        "candidate_global_ue_indices": (
            candidate_global_ue_indices
        ),
    }

    for name, tensor in (
        scheduler_tensors.items()
    ):
        if tensor.device != scheduler_device:
            raise ValueError(
                f"{name} is on the wrong device."
            )

    physical_tensors = {
        "h_freq": physical_inputs.h_freq,
        "recommended_rank": (
            physical_inputs
            .recommended_rank
        ),
        "rx_combiners": (
            physical_inputs
            .rx_combiners
        ),
    }

    for name, tensor in (
        physical_tensors.items()
    ):
        if tensor.device != scheduler_device:
            raise ValueError(
                f"{name} is on the wrong device."
            )

    valid_global_ue_indices = (
        candidate_global_ue_indices[
            candidate_valid_mask
        ]
    )

    if (
        valid_global_ue_indices.numel()
        > 0
        and torch.any(
            valid_global_ue_indices < 0
        )
    ):
        raise ValueError(
            "A valid candidate cannot have a "
            "negative global UE index."
        )

    if (
        torch.unique(
            valid_global_ue_indices
        ).numel()
        != valid_global_ue_indices.numel()
    ):
        raise ValueError(
            "Valid PF-TDS candidates must refer "
            "to distinct global UEs."
        )

    candidate_physical_ue_indices = (
        physical_inputs
        .candidate_physical_ue_indices
    )

    if (
        candidate_physical_ue_indices
        is not None
    ):
        if tuple(
            candidate_physical_ue_indices
            .shape
        ) != tuple(
            candidate_global_ue_indices
            .shape
        ):
            raise ValueError(
                "Physical candidate UE mapping must "
                "have shape [candidate]."
            )

        if (
            candidate_physical_ue_indices
            .device
            != scheduler_device
        ):
            raise ValueError(
                "Physical candidate UE mapping is "
                "on the wrong device."
            )

        valid_physical_ue_indices = (
            candidate_physical_ue_indices[
                candidate_valid_mask
            ]
        )

        if (
            valid_physical_ue_indices
            .numel()
            > 0
            and torch.any(
                valid_physical_ue_indices
                < 0
            )
        ):
            raise ValueError(
                "A valid candidate cannot have a "
                "negative physical UE index."
            )

        if (
            valid_physical_ue_indices
            .numel()
            > 0
            and torch.any(
                valid_physical_ue_indices
                >= physical_inputs
                .h_freq
                .shape[1]
            )
        ):
            raise ValueError(
                "A valid candidate physical UE "
                "index exceeds the stored PHY "
                "tensor."
            )

        if (
            torch.unique(
                valid_physical_ue_indices
            ).numel()
            != valid_physical_ue_indices
            .numel()
        ):
            raise ValueError(
                "Valid candidates must map to "
                "distinct physical UE entries."
            )

    selected_actor_actions = (
        actions[
            actions
            != num_candidates
        ]
    )

    if (
        selected_actor_actions.numel()
        > 0
    ):
        selected_validity = (
            candidate_valid_mask[
                selected_actor_actions
            ]
        )

        if not torch.all(
            selected_validity
        ):
            raise ValueError(
                "Actor actions select an invalid "
                "or padded candidate."
            )

    return num_candidates


def evaluate_ppo_physical_tti(
    *,
    allocation: CellAllocation,
    actions: torch.Tensor,
    past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    physical_inputs: PPOPhysicalScoreInputs,
    greedy_config: PPOGreedySearchConfig,
    physical_scorer: (
        CachedPPOPhysicalRBGScorer
        | None
    ) = None,
    run_counterfactual_greedy: bool = True,
) -> PPOPhysicalTTIOutcome:
    """
    Evaluate the physical outcome of one complete PPO
    TTI and run the PPO counterfactual greedy judge.

    This function deliberately does NOT:

        - update PF throughput history
        - update traffic buffers
        - choose the UE population used for G
        - attach transitions to the rollout buffer

    Those operations belong to the temporal/environment
    layer surrounding this physical outcome.
    """

    num_candidates = (
        _validate_physical_tti_inputs(
            allocation=allocation,
            actions=actions,
            past_average_throughput=(
                past_average_throughput
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
            physical_inputs=(
                physical_inputs
            ),
        )
    )

    #
    # Actual actor schedule:
    #
    # CellAllocation
    #     ->
    # RZF
    #     ->
    # MRC
    #     ->
    # SINR
    #     ->
    # MCS/TBLER
    #     ->
    # target-compliant rate
    #
    allocation_evaluation = (
        evaluate_cell_allocation(
            allocation=allocation,
            candidate_global_ue_indices=(
                physical_inputs
                .candidate_global_ue_indices
            ),

            candidate_physical_ue_indices=(
                physical_inputs
                .candidate_physical_ue_indices
            ),
            h_freq=physical_inputs.h_freq,
            serving_cell_index=(
                physical_inputs
                .serving_cell_index
            ),
            recommended_rank=(
                physical_inputs
                .recommended_rank
            ),
            rx_combiners=(
                physical_inputs
                .rx_combiners
            ),
            csi_subcarrier_index=(
                physical_inputs
                .csi_subcarrier_index
            ),
            subcarriers_per_rbg=(
                physical_inputs
                .subcarriers_per_rbg
            ),
            tx_power_per_subcarrier_w=(
                physical_inputs
                .tx_power_per_subcarrier_w
            ),
            noise_power_per_subcarrier_w=(
                physical_inputs
                .noise_power_per_subcarrier_w
            ),
            link_adaptation_config=(
                physical_inputs
                .link_adaptation_config
            ),
            rate_config=(
                physical_inputs
                .rate_config
            ),
            batch_index=(
                physical_inputs
                .batch_index
            ),
        )
    )

    candidate_total_rate = (
        allocation_evaluation
        .candidate_target_compliant_rate_bps
        .sum(
            dim=1
        )
    )

    if tuple(
        candidate_total_rate.shape
    ) != (
        num_candidates,
    ):
        raise RuntimeError(
            "Physical evaluator returned an "
            "unexpected candidate-rate shape."
        )


    # ----------------------------------------------------------
    # WARM-UP FAST PATH
    #
    # PAPER-SPECIFIED:
    #     PPO experience collection starts only
    #     after the initial 100 TTIs.
    #
    # During those TTIs the actual actor schedule
    # must still pass through the real PHY because
    # traffic delivery and PF history must evolve.
    #
    # But the counterfactual greedy reward judge is
    # training-only work and produces no usable PPO
    # sample during warm-up.
    # ----------------------------------------------------------

    if not run_counterfactual_greedy:
        if physical_scorer is not None:
            raise ValueError(
                "physical_scorer must be None when "
                "counterfactual greedy evaluation "
                "is disabled."
            )

        return PPOPhysicalTTIOutcome(
            allocation_evaluation=(
                allocation_evaluation
            ),
            candidate_total_target_compliant_rate_bps=(
                candidate_total_rate
                .detach()
                .clone()
            ),
            greedy_search=None,
            num_score_requests=0,
            num_unique_phy_evaluations=0,
        )

    # ----------------------------------------------------------
    # COUNTERFACTUAL TRAINING JUDGE
    #
    # A caller may provide a scorer already used by
    # Teacher 2. Reusing it preserves the exact same
    # PHY calculations while avoiding duplicate
    # candidate-set evaluations.
    # ----------------------------------------------------------

    if physical_scorer is None:
        physical_scorer = (
            CachedPPOPhysicalRBGScorer(
                physical_inputs
            )
        )

    elif (
        physical_scorer.inputs
        is not physical_inputs
    ):
        raise ValueError(
            "A shared PPO physical scorer must have "
            "been constructed from the exact same "
            "physical_inputs object."
        )

    score_requests_before = (
        physical_scorer
        .num_score_requests
    )

    unique_evaluations_before = (
        physical_scorer
        .num_unique_phy_evaluations
    )

    greedy_search = (
        evaluate_ppo_schedule_greedy_search(
            actions=actions,
            num_candidates=num_candidates,
            past_average_throughput=(
                past_average_throughput
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
            score_rbg=physical_scorer,
            config=greedy_config,
        )
    )

    #
    # Internal consistency check:
    # replaying the actor actions in the reward judge
    # must reconstruct the exact same final schedule.
    #
    if not torch.equal(
        greedy_search
        .allocation
        .candidate_by_user_slot,
        allocation
        .candidate_by_user_slot,
    ):
        raise RuntimeError(
            "PPO greedy-search replay did not "
            "reconstruct the actor allocation."
        )

    score_requests_after = (
        physical_scorer
        .num_score_requests
    )

    unique_evaluations_after = (
        physical_scorer
        .num_unique_phy_evaluations
    )

    return PPOPhysicalTTIOutcome(
        allocation_evaluation=(
            allocation_evaluation
        ),
        candidate_total_target_compliant_rate_bps=(
            candidate_total_rate
            .detach()
            .clone()
        ),
        greedy_search=greedy_search,
        num_score_requests=(
            score_requests_after
            - score_requests_before
        ),
        num_unique_phy_evaluations=(
            unique_evaluations_after
            - unique_evaluations_before
        ),
    )


def resolve_ppo_tti_reward(
    *,
    physical_outcome: PPOPhysicalTTIOutcome,
    reward_throughput_bps: torch.Tensor,
    reward_valid_ue_mask: torch.Tensor,
    reward_population_name: str,
    reward_config: PPORewardConfig,
    reward_reduction: PPORewardReduction = (
        "mean"
    ),
) -> PPOResolvedTTIReward:
    """
    Resolve the v3 PPO reward after the caller has
    explicitly selected the throughput population
    entering geometric mean G.

    PAPER-SPECIFIED:
        first layer:
            r_m,l = P * v_m

        later layers:
            r_m,l = k * v_m

        P = G / G_max

        k = 0.2

    PAPER-AMBIGUOUS / REPRODUCTION CHOICE:
        Exact public definition of the UE population
        over which G is evaluated in the training
        reward.

    Therefore this function intentionally receives
    reward_throughput_bps and reward_valid_ue_mask
    from the environment/orchestration layer instead
    of silently selecting a population.
    """

    if not isinstance(
        reward_population_name,
        str,
    ):
        raise TypeError(
            "reward_population_name must be a string."
        )

    if not reward_population_name.strip():
        raise ValueError(
            "reward_population_name cannot be empty."
        )

    greedy_search = (
        physical_outcome
        .greedy_search
    )

    if greedy_search is None:
        raise ValueError(
            "PPO reward cannot be resolved without "
            "the counterfactual greedy search."
        )

    reward_reference_device = (
        greedy_search
        .better_allocation_exists
        .device
    )

    if (
        reward_throughput_bps.device
        != reward_reference_device
    ):
        raise ValueError(
            "reward_throughput_bps is on the wrong "
            "device."
        )

    if (
        reward_valid_ue_mask.device
        != reward_reference_device
    ):
        raise ValueError(
            "reward_valid_ue_mask is on the wrong "
            "device."
        )

    reward_data = compute_ppo_reward(
        throughput_bps=(
            reward_throughput_bps
        ),
        valid_ue_mask=(
            reward_valid_ue_mask
        ),
        better_allocation_exists=(
            greedy_search
            .better_allocation_exists
        ),
        config=reward_config,
    )

    reduced_reward = (
        reduce_ppo_rbg_rewards(
            reward_data
            .reward_by_layer_rbg,
            reduction=reward_reduction,
        )
    )

    return PPOResolvedTTIReward(
        reward_population_name=(
            reward_population_name
        ),
        reward_data=reward_data,
        reduced_reward=(
            reduced_reward
        ),
    )

