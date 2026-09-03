from dataclasses import dataclass
from typing import Literal

import torch

from typing import (
    Literal,
    TypeAlias,
)

from oran_scheduler.rl.ppo_multicell_rollout import (
    OneLDSPPOMultiCellRolloutController,
)
from oran_scheduler.schedulers.allocation import (
    build_empty_cell_allocation,
)
from oran_scheduler.state.one_lds_decision import (
    OneLDSDecisionData,
    build_1lds_decision_data,
)

from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)
from oran_scheduler.rl.ppo_physical_score import (
    CachedPPOPhysicalRBGScorer,
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
from oran_scheduler.rl.ppo_single_cell_step import (
    PPOPhysicalInputsBuilder,
)
from oran_scheduler.rl.ppo_training_controller import (
    OneLDSPPOTrainingController,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_expert_collection import (
    PPOPFExpertTTILabels,
    commit_ppo_pf_expert_tti_labels,
    generate_ppo_pf_expert_tti_labels,
)
from oran_scheduler.rl.ppo_pf_expert import (
    PPOPFExpertConfig,
)

from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
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
from oran_scheduler.simulator.traffic import (
    TrafficBufferManager,
    TrafficServiceResult,
    TrafficTTIStart,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)

from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
    TDSEligibilityData,
    build_tds_eligibility,
)

PPOTrafficRewardPopulation = Literal[
    "candidates",
    "serving_ues",
]

PPOCellRolloutController: TypeAlias = (
    OneLDSPPOTrainingController
    | OneLDSPPOMultiCellRolloutController
)


@dataclass(frozen=True)
class TrafficAwarePPOCellTTIStepResult:
    """
    One complete traffic-aware PPO cell TTI.

    traffic_start:
        Arrivals and scheduler-visible queues at
        the beginning of the TTI.

    scheduler_observation:
        Original radio observation with dl_buffer
        replaced by the real traffic-buffer state.

    prepared:
        PF-TDS candidate selection and candidate
        1LDS inputs.

    schedule:
        PPO actor decisions and final allocation.

    physical_outcome:
        Real MU-MIMO service capacity and PPO
        counterfactual greedy-search result.

    serving_offered_capacity_bps:
        PHY service capacity scattered back into
        persistent serving-UE order.

    traffic_service:
        Actual traffic delivery after queue
        limitation.

    candidate_delivered_rate_bps:
        Actual delivered rates gathered back into
        PF-TDS candidate order.

    reward:
        PPO reward calculated using ACTUAL delivered
        throughput rather than raw PHY capacity.

    history_update:
        Next-TTI PF throughput history.
    """

    traffic_start: TrafficTTIStart

    scheduler_observation: (
        OneLDSCellTTIObservation
    )

    tds_eligibility: TDSEligibilityData

    prepared: PreparedOneLDSCellTTI

    schedule: OneLDSScheduleResult

    physical_outcome: PPOPhysicalTTIOutcome

    serving_offered_capacity_bps: (
        torch.Tensor
    )

    traffic_service: TrafficServiceResult

    candidate_delivered_rate_bps: (
        torch.Tensor
    )

    reward: PPOResolvedTTIReward

    expert_labels: (
        PPOPFExpertTTILabels | None
    )

    num_expert_demonstrations_added: int

    history_update: CellThroughputHistoryUpdate



@dataclass(frozen=True)
class PreparedTrafficAwarePPOCellTTI:
    """
    Scheduler inputs prepared BEFORE actor actions
    are sampled.

    This is the synchronization barrier used by
    centralized multi-cell PPO.

    first_decision:
        Exact empty-allocation slot-0 decision state.

        This is the state that will later be seen by
        the actor at user slot 0.
    """

    tti_index: int

    num_user_slots: int

    traffic_start: TrafficTTIStart

    scheduler_observation: (
        OneLDSCellTTIObservation
    )

    tds_eligibility: TDSEligibilityData

    prepared: PreparedOneLDSCellTTI

    physical_inputs: PPOPhysicalScoreInputs

    first_decision: OneLDSDecisionData



def _build_traffic_aware_observation(
    *,
    observation: OneLDSCellTTIObservation,
    buffer_for_scheduler_bits: torch.Tensor,
) -> OneLDSCellTTIObservation:
    """
    Replace the observation's temporary DL-buffer
    feature with the actual queue state for this TTI.
    """

    if tuple(
        buffer_for_scheduler_bits.shape
    ) != tuple(
        observation.dl_buffer.shape
    ):
        raise ValueError(
            "Traffic buffer and scheduler serving-UE "
            "layout must have the same shape."
        )

    if (
        buffer_for_scheduler_bits.device
        != observation.dl_buffer.device
    ):
        raise ValueError(
            "Traffic buffer and scheduler observation "
            "must be on the same device."
        )

    if not torch.is_floating_point(
        buffer_for_scheduler_bits
    ):
        raise ValueError(
            "Scheduler traffic buffer must use a "
            "floating-point dtype."
        )

    if not torch.isfinite(
        buffer_for_scheduler_bits
    ).all():
        raise ValueError(
            "Scheduler traffic buffer contains "
            "non-finite values."
        )

    if torch.any(
        buffer_for_scheduler_bits < 0.0
    ):
        raise ValueError(
            "Scheduler traffic buffer cannot be "
            "negative."
        )

    return OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            observation
            .serving_global_ue_indices
        ),
        serving_ue_valid_mask=(
            observation
            .serving_ue_valid_mask
        ),
        td_instantaneous_rate_bps=(
            observation
            .td_instantaneous_rate_bps
        ),
        rank=observation.rank,
        dl_buffer=(
            buffer_for_scheduler_bits
        ),
        wideband_cqi=(
            observation.wideband_cqi
        ),
        subband_cqi=(
            observation.subband_cqi
        ),
        precoder_directions=(
            observation.precoder_directions
        ),
    )



def _scatter_candidate_values_to_serving(
    *,
    candidate_values: torch.Tensor,
    candidate_serving_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    num_serving_ues: int,
) -> torch.Tensor:
    """
    Convert:

        [candidate]

    into persistent:

        [serving_ue]

    order.

    Serving UEs outside the current TDS candidate
    set receive zero offered service.
    """

    if candidate_values.ndim != 1:
        raise ValueError(
            "candidate_values must have shape "
            "[candidate]."
        )

    expected_shape = (
        candidate_values.shape
    )

    if tuple(
        candidate_serving_indices.shape
    ) != tuple(
        expected_shape
    ):
        raise ValueError(
            "candidate_serving_indices has the "
            "wrong shape."
        )

    if tuple(
        candidate_valid_mask.shape
    ) != tuple(
        expected_shape
    ):
        raise ValueError(
            "candidate_valid_mask has the wrong "
            "shape."
        )

    if candidate_valid_mask.dtype != torch.bool:
        raise ValueError(
            "candidate_valid_mask must use "
            "torch.bool."
        )

    device = candidate_values.device

    if (
        candidate_serving_indices.device
        != device
        or candidate_valid_mask.device
        != device
    ):
        raise ValueError(
            "Candidate tensors must share one "
            "device."
        )

    serving_values = torch.zeros(
        num_serving_ues,
        dtype=candidate_values.dtype,
        device=device,
    )

    valid_indices = (
        candidate_serving_indices[
            candidate_valid_mask
        ]
    )

    valid_values = (
        candidate_values[
            candidate_valid_mask
        ]
    )

    if valid_indices.numel() > 0:
        if torch.any(
            valid_indices < 0
        ):
            raise ValueError(
                "Valid candidate serving indices "
                "cannot be negative."
            )

        if torch.any(
            valid_indices >= num_serving_ues
        ):
            raise ValueError(
                "Candidate serving index exceeds "
                "the serving layout."
            )

        if (
            torch.unique(
                valid_indices
            ).numel()
            != valid_indices.numel()
        ):
            raise ValueError(
                "PF-TDS candidate serving indices "
                "must be unique."
            )

        serving_values[
            valid_indices
        ] = valid_values

    return serving_values



def _gather_serving_values_to_candidates(
    *,
    serving_values: torch.Tensor,
    candidate_serving_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert persistent serving-UE values back into
    current PF-TDS candidate order.

    Input:
        [serving_ue]

    Output:
        [candidate]
    """

    if serving_values.ndim != 1:
        raise ValueError(
            "serving_values must have shape "
            "[serving_ue]."
        )

    gathered = serving_values[
        candidate_serving_indices
    ]

    return torch.where(
        candidate_valid_mask,
        gathered,
        torch.zeros_like(
            gathered
        ),
    )


def _validate_physical_inputs(
    *,
    prepared: PreparedOneLDSCellTTI,
    physical_inputs: PPOPhysicalScoreInputs,
) -> None:
    if not torch.equal(
        physical_inputs
        .candidate_global_ue_indices,
        prepared
        .candidate_global_ue_indices,
    ):
        raise ValueError(
            "Physical inputs do not match the "
            "PF-TDS candidate ordering."
        )


def prepare_traffic_aware_ppo_cell_tti(
    *,
    tti_index: int,
    observation: OneLDSCellTTIObservation,
    traffic_manager: TrafficBufferManager,
    tds_eligibility_config: (
        TDSBufferEligibilityConfig
    ),
    state_manager: OneLDSCellTTIStateManager,
    state_config: OneLDSStateConfig,
    physical_inputs_builder: (
        PPOPhysicalInputsBuilder
    ),
    num_user_slots: int,
    packet_arrivals: (
        torch.Tensor | None
    ) = None,
    device: str | torch.device = "cuda:0",
) -> PreparedTrafficAwarePPOCellTTI:
    """
    Prepare one cell up to, but NOT including, the
    actor's first scheduling decision.

    Sequence:

        traffic arrivals
            ->
        scheduler-visible queue
            ->
        TDS eligibility
            ->
        PF TDS
            ->
        candidate PHY inputs
            ->
        exact slot-0 RL state

    No PPO action is sampled here.
    """

    if tti_index < 0:
        raise ValueError(
            "tti_index must be non-negative."
        )

    if num_user_slots <= 0:
        raise ValueError(
            "num_user_slots must be positive."
        )

    num_serving_ues = int(
        observation
        .serving_global_ue_indices
        .shape[
            0
        ]
    )

    if tuple(
        traffic_manager
        .current_buffer_bits
        .shape
    ) != (
        num_serving_ues,
    ):
        raise ValueError(
            "Traffic manager and scheduler serving "
            "layout have different UE counts."
        )

    traffic_start = (
        traffic_manager.begin_tti(
            tti_index=tti_index,
            packet_arrivals=(
                packet_arrivals
            ),
        )
    )

    scheduler_observation = (
        _build_traffic_aware_observation(
            observation=observation,
            buffer_for_scheduler_bits=(
                traffic_start
                .buffer_for_scheduler_bits
            ),
        )
    )

    tds_eligibility = build_tds_eligibility(
        serving_ue_valid_mask=(
            scheduler_observation
            .serving_ue_valid_mask
        ),
        dl_buffer_bits=(
            scheduler_observation
            .dl_buffer
        ),
        full_buffer_mask=(
            traffic_manager
            .full_buffer_mask
        ),
        config=(
            tds_eligibility_config
        ),
    )

    prepared = state_manager.prepare_tti(
        tti_index=tti_index,
        observation=(
            scheduler_observation
        ),
        tds_eligible_mask=(
            tds_eligibility
            .eligible_mask
        ),
    )

    physical_inputs = (
        physical_inputs_builder(
            prepared
        )
    )

    _validate_physical_inputs(
        prepared=prepared,
        physical_inputs=physical_inputs,
    )

    #
    # Empty allocation = exact state BEFORE the
    # first 1LDS action.
    #
    empty_allocation = (
        build_empty_cell_allocation(
            num_user_slots=(
                num_user_slots
            ),
            num_rbgs=(
                state_config.num_rbgs
            ),
            device=device,
        )
    )

    first_decision = (
        build_1lds_decision_data(
            allocation=(
                empty_allocation
            ),
            user_slot_index=0,
            inputs=(
                prepared
                .decision_inputs
            ),
            config=state_config,
        )
    )

    return PreparedTrafficAwarePPOCellTTI(
        tti_index=tti_index,
        num_user_slots=(
            num_user_slots
        ),
        traffic_start=traffic_start,
        scheduler_observation=(
            scheduler_observation
        ),
        tds_eligibility=(
            tds_eligibility
        ),
        prepared=prepared,
        physical_inputs=(
            physical_inputs
        ),
        first_decision=(
            first_decision
        ),
    )


def _resolve_traffic_reward(
    *,
    physical_outcome: PPOPhysicalTTIOutcome,
    candidate_delivered_rate_bps: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    serving_delivered_rate_bps: torch.Tensor,
    serving_valid_mask: torch.Tensor,
    reward_population: (
        PPOTrafficRewardPopulation
    ),
    reward_config: PPORewardConfig,
    reward_reduction: PPORewardReduction,
) -> PPOResolvedTTIReward:
    """
    Keep the public-paper ambiguity around the
    geometric-mean UE population explicit.
    """

    if reward_population == "candidates":
        return resolve_ppo_tti_reward(
            physical_outcome=(
                physical_outcome
            ),
            reward_throughput_bps=(
                candidate_delivered_rate_bps
            ),
            reward_valid_ue_mask=(
                candidate_valid_mask
            ),
            reward_population_name=(
                "pf_tds_candidates_actual_delivery"
            ),
            reward_config=reward_config,
            reward_reduction=(
                reward_reduction
            ),
        )

    if reward_population == "serving_ues":
        return resolve_ppo_tti_reward(
            physical_outcome=(
                physical_outcome
            ),
            reward_throughput_bps=(
                serving_delivered_rate_bps
            ),
            reward_valid_ue_mask=(
                serving_valid_mask
            ),
            reward_population_name=(
                "serving_ues_actual_delivery"
            ),
            reward_config=reward_config,
            reward_reduction=(
                reward_reduction
            ),
        )

    raise ValueError(
        "reward_population must be "
        "'candidates' or 'serving_ues'."
    )


def run_traffic_aware_ppo_cell_tti_step(
    *,
    tti_index: int,
    collect_experience: bool = True,
    observation: OneLDSCellTTIObservation,
    traffic_manager: TrafficBufferManager,
    tds_eligibility_config: (
        TDSBufferEligibilityConfig
    ),
    state_manager: OneLDSCellTTIStateManager,
    training_controller: (
        PPOCellRolloutController
    ),
    state_config: OneLDSStateConfig,
    physical_inputs_builder: (
        PPOPhysicalInputsBuilder
    ),
    greedy_config: PPOGreedySearchConfig,
    reward_config: PPORewardConfig,
    reward_population: (
        PPOTrafficRewardPopulation
    ),
    expert_buffer: (
        PPOExpertDemonstrationBuffer | None
    ) = None,
    pf_expert_config: (
        PPOPFExpertConfig | None
    ) = None,
    candidate_augmentation_config: (
        PPOCandidateAugmentationConfig | None
    ) = None,
    candidate_augmentation_generator: (
        torch.Generator | None
    ) = None,

    preparation: (
        PreparedTrafficAwarePPOCellTTI | None
    ) = None,
    reward_reduction: PPORewardReduction = "mean",
    packet_arrivals: torch.Tensor | None = None,
    device: str | torch.device = "cuda:0",
) -> TrafficAwarePPOCellTTIStepResult:
    """
    Execute one mixed-traffic PPO scheduling TTI.

    Sequence:

        traffic arrivals
            ->
        real DL-buffer state
            ->
        PF TDS
            ->
        PPO 1LDS scheduling
            ->
        real MU-MIMO PHY service capacity
            ->
        traffic queue limitation
            ->
        actual delivered throughput
            ->
        PPO reward
            ->
        PF history update

    PAPER-SPECIFIED:
        - mixed Full Buffer / FTP3 training traffic
        - DL-buffer feature
        - throughput-dependent PPO reward
        - throughput history feeds future scheduler state

    OPEN-REPRODUCTION:
        - exact G reward population
        - PF averaging constant
        - buffer normalization
        - exact empty-buffer PF-TDS filtering
    """

    if not collect_experience:
        #
        # Warm-up is only safe before an on-policy
        # trajectory has begun.
        #
        # We must never stop collecting in the
        # middle of a PPO trajectory because the
        # unresolved last-layer transition would
        # lose its next state.
        #
        if (
            training_controller
            .has_unresolved_tti_boundary
        ):
            raise RuntimeError(
                "Cannot disable PPO experience "
                "collection while a TTI boundary "
                "transition is unresolved."
            )

        if (
            training_controller
            .agent_buffer_size
            != 0
        ):
            raise RuntimeError(
                "Cannot disable PPO experience "
                "collection after on-policy samples "
                "have entered the agent buffer."
            )

    expert_buffer_enabled = (
        expert_buffer is not None
    )

    expert_config_enabled = (
        pf_expert_config is not None
    )

    if (
        expert_buffer_enabled
        != expert_config_enabled
    ):
        raise ValueError(
            "expert_buffer and pf_expert_config "
            "must either both be provided or both "
            "be None."
        )


    if preparation is None:
        preparation = (
            prepare_traffic_aware_ppo_cell_tti(
                tti_index=tti_index,
                observation=observation,
                traffic_manager=(
                    traffic_manager
                ),
                tds_eligibility_config=(
                    tds_eligibility_config
                ),
                state_manager=(
                    state_manager
                ),
                state_config=state_config,
                physical_inputs_builder=(
                    physical_inputs_builder
                ),
                num_user_slots=(
                    training_controller
                    .config
                    .num_user_slots
                ),
                packet_arrivals=(
                    packet_arrivals
                ),
                device=device,
            )
        )

    else:
        if (
            preparation.tti_index
            != tti_index
        ):
            raise ValueError(
                "Prepared cell TTI has the wrong "
                "tti_index."
            )

        if (
            preparation.num_user_slots
            != (
                training_controller
                .config
                .num_user_slots
            )
        ):
            raise ValueError(
                "Prepared cell TTI and rollout "
                "controller disagree on the number "
                "of user slots."
            )

    traffic_start = (
        preparation.traffic_start
    )

    scheduler_observation = (
        preparation
        .scheduler_observation
    )

    tds_eligibility = (
        preparation
        .tds_eligibility
    )

    prepared = (
        preparation.prepared
    )

    physical_inputs = (
        preparation.physical_inputs
    )

    num_serving_ues = int(
        scheduler_observation
        .serving_global_ue_indices
        .shape[
            0
        ]
    )

    # ----------------------------------------------------------
    # 3. PPO performs every configured scheduler user slot.
    # ----------------------------------------------------------

    if collect_experience:
        action_policy = (
            training_controller
            .make_action_policy(
                tti_index=tti_index,
            )
        )

    else:
        action_policy = (
            training_controller
            .make_untracked_action_policy()
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
        action_policy=action_policy,
        device=device,
    )

    #
    # The precomputed synchronization state must
    # exactly equal the state actually seen by
    # slot-0 actor inference.
    #
    actual_first_decision = (
        schedule.decisions[
            0
        ]
    )

    if not torch.equal(
        actual_first_decision
        .state_data
        .state,
        preparation
        .first_decision
        .state_data
        .state,
    ):
        raise RuntimeError(
            "Precomputed multi-cell slot-0 state "
            "does not match the actual 1LDS slot-0 "
            "state."
        )

    if not torch.equal(
        actual_first_decision
        .action_mask,
        preparation
        .first_decision
        .action_mask,
    ):
        raise RuntimeError(
            "Precomputed multi-cell slot-0 mask "
            "does not match the actual 1LDS slot-0 "
            "mask."
        )


    # ----------------------------------------------------------
    # Teacher 2: PF expert supervision.
    #
    # PAPER:
    #     Each PPO layer also gets an expert action
    #     label.
    #
    # Our implementation reconstructs each exact
    # pre-action PPO allocation from schedule history.
    #
    # REPRODUCTION CHOICE:
    #     Expert demonstrations are collected only
    #     once normal PPO experience collection has
    #     started. Warm-up TTIs therefore do not fill
    #     D_expert.
    # ----------------------------------------------------------

    expert_labels: (
        PPOPFExpertTTILabels | None
    ) = None

    if (
        collect_experience
        and expert_buffer is not None
    ):
        assert (
            pf_expert_config is not None
        )

        expert_physical_scorer = (
            CachedPPOPhysicalRBGScorer(
                physical_inputs
            )
        )

        expert_labels = (
            generate_ppo_pf_expert_tti_labels(
                schedule=schedule,
                num_candidates=(
                    state_config
                    .num_candidates
                ),
                past_average_throughput=(
                    prepared
                    .decision_inputs
                    .past_average_throughput
                ),
                candidate_valid_mask=(
                    prepared
                    .candidate_valid_mask
                ),
                score_rbg=(
                    expert_physical_scorer
                ),
                config=(
                    pf_expert_config
                ),
            )
        )

    # ----------------------------------------------------------
    # 4. Common PHY computes OFFERED service capacity.
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # 5. Candidate capacity -> persistent serving-UE order.
    # ----------------------------------------------------------

    serving_offered_capacity_bps = (
        _scatter_candidate_values_to_serving(
            candidate_values=(
                physical_outcome
                .candidate_total_target_compliant_rate_bps
            ),
            candidate_serving_indices=(
                prepared
                .candidate_serving_indices
            ),
            candidate_valid_mask=(
                prepared
                .candidate_valid_mask
            ),
            num_serving_ues=(
                num_serving_ues
            ),
        )
    )

    # ----------------------------------------------------------
    # 6. Traffic model converts CAPACITY into DELIVERY.
    # ----------------------------------------------------------

    traffic_service = (
        traffic_manager.apply_service(
            offered_service_capacity_bps=(
                serving_offered_capacity_bps
            )
        )
    )

    candidate_delivered_rate_bps = (
        _gather_serving_values_to_candidates(
            serving_values=(
                traffic_service
                .delivered_rate_bps
            ),
            candidate_serving_indices=(
                prepared
                .candidate_serving_indices
            ),
            candidate_valid_mask=(
                prepared
                .candidate_valid_mask
            ),
        )
    )

    # ----------------------------------------------------------
    # 7. PPO reward uses ACTUAL delivered throughput.
    # ----------------------------------------------------------

    reward = _resolve_traffic_reward(
        physical_outcome=(
            physical_outcome
        ),
        candidate_delivered_rate_bps=(
            candidate_delivered_rate_bps
        ),
        candidate_valid_mask=(
            prepared.candidate_valid_mask
        ),
        serving_delivered_rate_bps=(
            traffic_service
            .delivered_rate_bps
        ),
        serving_valid_mask=(
            scheduler_observation
            .serving_ue_valid_mask
        ),
        reward_population=(
            reward_population
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
            "Traffic-aware PPO reward has the "
            "wrong [user_slot, RBG] shape."
        )

    # ----------------------------------------------------------
    # 8. Attach delayed reward to PPO trajectory.
    # ----------------------------------------------------------

    if collect_experience:
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

    # ----------------------------------------------------------
    # Algorithm 1:
    #
    # After the TTI outcome/reward is available,
    # copy (state, expert action) supervision from
    # the temporary TTI data into D_expert.
    # ----------------------------------------------------------

    num_expert_demonstrations_added = 0

    if (
        collect_experience
        and expert_labels is not None
    ):
        assert expert_buffer is not None

        # num_expert_demonstrations_added = (
        #     commit_ppo_pf_expert_tti_labels(
        #         labels=(
        #             expert_labels
        #         ),
        #         expert_buffer=(
        #             expert_buffer
        #         ),
        #     )
        # )

        num_expert_demonstrations_added = (
            commit_ppo_pf_expert_tti_labels(
                labels=expert_labels,
                expert_buffer=expert_buffer,
                augmentation_config=(
                    candidate_augmentation_config
                ),
                augmentation_generator=(
                    candidate_augmentation_generator
                ),
            )
        )

    # ----------------------------------------------------------
    # 9. Actual delivery updates PF history.
    # ----------------------------------------------------------

    history_update = (
        state_manager.complete_tti(
            candidate_delivered_rate_bps=(
                candidate_delivered_rate_bps
            )
        )
    )

    return TrafficAwarePPOCellTTIStepResult(
        traffic_start=traffic_start,
        scheduler_observation=(
            scheduler_observation
        ),
        tds_eligibility=(
            tds_eligibility
        ),
        prepared=prepared,
        schedule=schedule,
        physical_outcome=(
            physical_outcome
        ),
        serving_offered_capacity_bps=(
            serving_offered_capacity_bps
        ),
        traffic_service=(
            traffic_service
        ),
        candidate_delivered_rate_bps=(
            candidate_delivered_rate_bps
        ),
        reward=reward,
        expert_labels=(
            expert_labels
        ),
        num_expert_demonstrations_added=(
            num_expert_demonstrations_added
        ),
        history_update=(
            history_update
        ),
    )

