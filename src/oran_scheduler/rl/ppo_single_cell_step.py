from collections.abc import Callable
from dataclasses import dataclass

import torch

from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)
from oran_scheduler.rl.ppo_physical_score import (
    PPOPhysicalScoreInputs,
)
from oran_scheduler.rl.ppo_physical_tti import (
    PPOPhysicalTTIOutcome,
    PPOResolvedTTIReward,
    evaluate_ppo_physical_tti,
    resolve_ppo_tti_reward,
)
from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
    PPORewardReduction,
)
from oran_scheduler.rl.ppo_training_controller import (
    OneLDSPPOTrainingController,
)
from oran_scheduler.schedulers.one_lds_loop import (
    OneLDSScheduleResult,
    run_1lds_user_slot_loop,
)
from oran_scheduler.schedulers.throughput_history import (
    CellThroughputHistoryUpdate,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
    OneLDSCellTTIStateManager,
    PreparedOneLDSCellTTI,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)


PPOPhysicalInputsBuilder = Callable[
    [
        PreparedOneLDSCellTTI,
    ],
    PPOPhysicalScoreInputs,
]


@dataclass(frozen=True)
class FullBufferPPOCellTTIStepResult:
    """
    Result of one complete single-cell Full-Buffer
    PPO scheduling TTI.

    prepared:
        PF-TDS result and candidate-level 1LDS
        scheduler inputs.

    schedule:
        Complete actor scheduling result.

    physical_outcome:
        Real MU-MIMO PHY outcome plus PPO
        counterfactual greedy search.

    reward:
        Raw [layer, RBG] PPO reward plus the
        scalar-per-layer reduction.

    history_update:
        PF average-throughput state after applying
        realized Full-Buffer service.
    """

    prepared: PreparedOneLDSCellTTI

    schedule: OneLDSScheduleResult

    physical_outcome: PPOPhysicalTTIOutcome

    reward: PPOResolvedTTIReward

    history_update: CellThroughputHistoryUpdate


def _validate_physical_inputs_match_tds(
    *,
    prepared: PreparedOneLDSCellTTI,
    physical_inputs: PPOPhysicalScoreInputs,
) -> None:
    """
    Ensure that the physical evaluator and scheduler
    refer to exactly the same PF-TDS candidate slots.
    """

    if not torch.equal(
        physical_inputs
        .candidate_global_ue_indices,
        prepared
        .candidate_global_ue_indices,
    ):
        raise ValueError(
            "PPO physical inputs do not match the "
            "PF-TDS candidate ordering."
        )

    if (
        physical_inputs
        .candidate_global_ue_indices
        .device
        != prepared
        .candidate_global_ue_indices
        .device
    ):
        raise ValueError(
            "Physical inputs and PF-TDS candidates "
            "must be on the same device."
        )


def _validate_step_dimensions(
    *,
    prepared: PreparedOneLDSCellTTI,
    state_config: OneLDSStateConfig,
    training_controller: (
        OneLDSPPOTrainingController
    ),
) -> None:
    num_candidates = int(
        prepared
        .candidate_global_ue_indices
        .shape[0]
    )

    num_rbgs = int(
        prepared
        .decision_inputs
        .subband_cqi
        .shape[1]
    )

    if (
        state_config.num_candidates
        != num_candidates
    ):
        raise ValueError(
            "OneLDSStateConfig.num_candidates does "
            "not match PF-TDS output."
        )

    if (
        state_config.num_rbgs
        != num_rbgs
    ):
        raise ValueError(
            "OneLDSStateConfig.num_rbgs does not "
            "match the TTI observation."
        )

    if (
        training_controller
        .config
        .num_user_slots
        < 1
    ):
        raise ValueError(
            "PPO controller must have at least one "
            "user slot."
        )


def run_full_buffer_ppo_cell_tti_step(
    *,
    tti_index: int,
    observation: OneLDSCellTTIObservation,
    state_manager: OneLDSCellTTIStateManager,
    training_controller: (
        OneLDSPPOTrainingController
    ),
    state_config: OneLDSStateConfig,
    physical_inputs_builder: (
        PPOPhysicalInputsBuilder
    ),
    greedy_config: PPOGreedySearchConfig,
    reward_config: PPORewardConfig,
    reward_reduction: PPORewardReduction = "mean",
    device: str | torch.device = "cuda:0",
) -> FullBufferPPOCellTTIStepResult:
    """
    Run one complete Full-Buffer PPO scheduling TTI.

    Execution order:

        persistent PF history
            ->
        PF TDS
            ->
        candidate-level 1LDS state
            ->
        PPO actor user-slot loop
            ->
        final allocation
            ->
        real MU-MIMO PHY
            ->
        counterfactual PPO greedy judge
            ->
        PPO reward
            ->
        PPO trajectory finish
            ->
        PF throughput-history update

    TEMPORARY SCAFFOLDING:
        The physical target-compliant candidate rate
        is interpreted as actually delivered
        throughput.

        This is appropriate for this Full-Buffer
        integration path but must NOT later be reused
        blindly for finite-buffer FTP traffic.
    """

    prepared = state_manager.prepare_tti(
        tti_index=tti_index,
        observation=observation,
    )

    _validate_step_dimensions(
        prepared=prepared,
        state_config=state_config,
        training_controller=(
            training_controller
        ),
    )

    #
    # Construct this before actor interaction so an
    # invalid PHY/candidate mapping fails before we
    # begin recording PPO actions for the TTI.
    #
    physical_inputs = (
        physical_inputs_builder(
            prepared
        )
    )

    _validate_physical_inputs_match_tds(
        prepared=prepared,
        physical_inputs=physical_inputs,
    )

    schedule = run_1lds_user_slot_loop(
        num_user_slots=(
            training_controller
            .config
            .num_user_slots
        ),
        inputs=(
            prepared.decision_inputs
        ),
        state_config=state_config,
        action_policy=(
            training_controller
            .make_action_policy(
                tti_index=tti_index,
            )
        ),
        device=device,
    )

    physical_outcome = (
        evaluate_ppo_physical_tti(
            allocation=(
                schedule.allocation
            ),
            actions=schedule.actions,
            past_average_throughput=(
                prepared
                .decision_inputs
                .past_average_throughput
            ),
            candidate_valid_mask=(
                prepared
                .candidate_valid_mask
            ),
            physical_inputs=(
                physical_inputs
            ),
            greedy_config=(
                greedy_config
            ),
        )
    )

    #
    # TEMPORARY FULL-BUFFER CHOICE:
    #
    # Every valid TDS candidate participates in the
    # reward-population geometric mean, and the
    # common PHY capacity is treated as delivered
    # throughput.
    #
    reward = resolve_ppo_tti_reward(
        physical_outcome=(
            physical_outcome
        ),
        reward_throughput_bps=(
            physical_outcome
            .candidate_total_target_compliant_rate_bps
        ),
        reward_valid_ue_mask=(
            prepared
            .candidate_valid_mask
        ),
        reward_population_name=(
            "pf_tds_candidates_full_buffer_capacity"
        ),
        reward_config=reward_config,
        reward_reduction=(
            reward_reduction
        ),
    )

    expected_reward_shape = (
        training_controller
        .config
        .num_user_slots,
        state_config.num_rbgs,
    )

    if tuple(
        reward
        .reward_data
        .reward_by_layer_rbg
        .shape
    ) != expected_reward_shape:
        raise RuntimeError(
            "PPO reward does not match the "
            "[user_slot, RBG] scheduling shape."
        )

    if tuple(
        reward.reduced_reward.shape
    ) != (
        training_controller
        .config
        .num_user_slots,
    ):
        raise RuntimeError(
            "Reduced PPO reward must have shape "
            "[user_slot]."
        )

    #
    # This stores rewards for the TTI and leaves the
    # final layer waiting for next TTI's slot-0 state.
    #
    training_controller.finish_tti(
        reward_by_rbg=(
            reward
            .reward_data
            .reward_by_layer_rbg
        ),
        reduced_reward=(
            reward.reduced_reward
        ),
    )

    #
    # Full Buffer:
    #
    # offered service == actually delivered service.
    #
    history_update = (
        state_manager.complete_tti(
            candidate_delivered_rate_bps=(
                physical_outcome
                .candidate_total_target_compliant_rate_bps
            )
        )
    )

    return FullBufferPPOCellTTIStepResult(
        prepared=prepared,
        schedule=schedule,
        physical_outcome=(
            physical_outcome
        ),
        reward=reward,
        history_update=(
            history_update
        ),
    )


