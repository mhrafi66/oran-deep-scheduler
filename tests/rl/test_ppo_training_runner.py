import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)
from oran_scheduler.rl.ppo_agent_buffer import (
    PPOAgentBufferConfig,
    PPOAgentUpdateCoordinator,
)
from oran_scheduler.rl.ppo_critic import (
    OneLDSPPOCritic,
    OneLDSPPOCriticConfig,
)
from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
)
from oran_scheduler.rl.ppo_physical_score import (
    PPOPhysicalScoreInputs,
)
from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
)
from oran_scheduler.rl.ppo_training_controller import (
    OneLDSPPOTrainingController,
    OneLDSPPOTrainingControllerConfig,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingRunnerConfig,
    PPOTrainingTTIInputs,
    run_single_cell_ppo_training,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizerConfig,
    create_ppo_optimizers,
)

from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertBufferConfig,
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_pf_expert import (
    PPOPFExpertConfig,
)

from oran_scheduler.rl.ppo_expert_guidance import (
    PPOExpertGuidanceConfig,
)
from oran_scheduler.rl.ppo_expert_training import (
    PPOExpertGuidanceCoordinator,
)

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
    OneLDSCellTTIStateManager,
    PreparedOneLDSCellTTI,
)
from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)
from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
    TrafficBufferManager,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)

from oran_scheduler.rl.ppo_multicell_rollout import (
    OneLDSPPOMultiCellRolloutConfig,
    OneLDSPPOMultiCellRolloutController,
)
from oran_scheduler.rl.ppo_multicell_training_runner import (
    run_multicell_ppo_training,
)
from oran_scheduler.rl.ppo_multistream_buffer import (
    PPOMultiStreamBufferConfig,
    PPOMultiStreamTransitionBuffer,
)
from oran_scheduler.rl.ppo_multistream_training import (
    PPOMultiStreamTrainingCoordinator,
)
from oran_scheduler.rl.ppo_multistream_update import (
    PPOMultiStreamUpdateConfig,
)

from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertBufferConfig,
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_expert_guidance import (
    PPOExpertGuidanceConfig,
)
from oran_scheduler.rl.ppo_expert_training import (
    PPOExpertGuidanceCoordinator,
)
from oran_scheduler.rl.ppo_pf_expert import (
    PPOPFExpertConfig,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def test_training_runner_uses_paper_warmup_boundary():
    config = PPOTrainingRunnerConfig()

    assert (
        config.first_collection_tti_index
        == 100
    )


def build_small_training_system(
    device: torch.device,
):
    state_config = OneLDSStateConfig(
        throughput_normalization_bps=(
            100.0e6
        ),
        buffer_normalization=8000.0,
        subband_cqi_normalization=15.0,
        num_candidates=2,
        num_rbgs=1,
        max_rank=2,
    )

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=(
                state_config.state_size
            ),
            hidden_size=8,
            num_rbgs=1,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=(
                state_config.state_size
            ),
            hidden_size=8,
        )
    ).to(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    controller = (
        OneLDSPPOTrainingController(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            agent_buffer=(
                PPOAgentUpdateCoordinator(
                    config=(
                        PPOAgentBufferConfig(
                            update_size=4,
                        )
                    )
                )
            ),
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            loss_config=PPOLossConfig(
                entropy_coefficient=0.0,
            ),
            config=(
                OneLDSPPOTrainingControllerConfig(
                    num_user_slots=2,
                )
            ),
        )
    )

    state_manager = (
        OneLDSCellTTIStateManager(
            initial_average_throughput_bps=(
                torch.full(
                    (
                        2,
                    ),
                    1.0e6,
                    dtype=torch.float32,
                    device=device,
                )
            ),
            serving_global_ue_indices=(
                torch.tensor(
                    [
                        0,
                        1,
                    ],
                    dtype=torch.long,
                    device=device,
                )
            ),
            serving_ue_valid_mask=(
                torch.ones(
                    2,
                    dtype=torch.bool,
                    device=device,
                )
            ),
            tds_config=PFTimeDomainConfig(
                num_candidates=2,
            ),
            throughput_forgetting_factor=0.5,
        )
    )

    traffic_manager = TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                True,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=1000,
            packet_arrival_rate_per_s=500.0,
            tti_duration_s=0.001,
        ),
        full_buffer_state_bits=8000.0,
        seed=1234,
    )

    return (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    )


def build_input_provider(
    device: torch.device,
):
    h_freq = torch.zeros(
        (
            1,
            2,
            1,
            1,
            2,
            1,
            12,
        ),
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0,
        0,
        0,
        0,
        0,
        0,
        :,
    ] = 1.0

    h_freq[
        0,
        1,
        0,
        0,
        1,
        0,
        :,
    ] = 1.0

    directions = torch.zeros(
        (
            2,
            1,
            1,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        0,
        0,
        0,
        0,
    ] = 1.0

    directions[
        1,
        0,
        0,
        1,
    ] = 1.0

    observation = OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            )
        ),
        serving_ue_valid_mask=torch.ones(
            2,
            dtype=torch.bool,
            device=device,
        ),
        td_instantaneous_rate_bps=(
            torch.tensor(
                [
                    10.0e6,
                    8.0e6,
                ],
                dtype=torch.float32,
                device=device,
            )
        ),
        rank=torch.ones(
            2,
            dtype=torch.long,
            device=device,
        ),
        dl_buffer=torch.zeros(
            2,
            dtype=torch.float32,
            device=device,
        ),
        wideband_cqi=torch.full(
            (
                2,
            ),
            10.0,
            device=device,
        ),
        subband_cqi=torch.full(
            (
                2,
                1,
            ),
            10.0,
            device=device,
        ),
        precoder_directions=(
            directions
        ),
    )

    def provider(
        tti_index: int,
    ) -> PPOTrainingTTIInputs:
        del tti_index

        def physical_builder(
            prepared: PreparedOneLDSCellTTI,
        ) -> PPOPhysicalScoreInputs:
            return PPOPhysicalScoreInputs(
                candidate_global_ue_indices=(
                    prepared
                    .candidate_global_ue_indices
                ),
                h_freq=h_freq,
                serving_cell_index=0,
                recommended_rank=torch.ones(
                    (
                        1,
                        2,
                    ),
                    dtype=torch.long,
                    device=device,
                ),
                rx_combiners=torch.ones(
                    (
                        1,
                        2,
                        1,
                        2,
                        1,
                    ),
                    dtype=torch.complex64,
                    device=device,
                ),
                csi_subcarrier_index=6,
                subcarriers_per_rbg=12,
                tx_power_per_subcarrier_w=2.0,
                noise_power_per_subcarrier_w=(
                    1.0e-3
                ),
                link_adaptation_config=(
                    LinkAdaptationConfig(
                        num_rbgs=1,
                        subcarriers_per_rbg=12,
                        device=str(
                            device
                        ),
                    )
                ),
                rate_config=RateConfig(
                    subcarriers_per_rbg=12,
                    device=str(
                        device
                    ),
                ),
            )

        return PPOTrainingTTIInputs(
            observation=observation,
            physical_inputs_builder=(
                physical_builder
            ),

            #
            # FTP UE 1 receives one packet each TTI.
            #
            packet_arrivals=torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
        )

    return provider


def test_training_runner_warmup_does_not_collect_ppo_samples():
    device = preferred_device()

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    ) = build_small_training_system(
        device
    )

    result = run_single_cell_ppo_training(
        start_tti_index=0,
        num_ttis=2,
        input_provider=(
            build_input_provider(
                device
            )
        ),
        traffic_manager=traffic_manager,
        tds_eligibility_config=(
            TDSBufferEligibilityConfig(
                mode="data_available_only",
            )
        ),
        state_manager=state_manager,
        training_controller=controller,
        state_config=state_config,
        greedy_config=(
            PPOGreedySearchConfig()
        ),
        reward_config=PPORewardConfig(
            geometric_mean_normalizer_bps=(
                100.0e6
            ),
        ),
        reward_population="candidates",
        runner_config=PPOTrainingRunnerConfig(
            first_collection_tti_index=2,
        ),
        device=device,
    )

    assert result.num_warmup_ttis == 2

    assert result.num_collection_ttis == 0

    assert (
        result.first_collected_tti_index
        is None
    )

    assert result.num_ppo_updates == 0

    assert result.final_agent_buffer_size == 0

    assert not (
        result.has_unresolved_tti_boundary
    )

    assert controller.agent_buffer_size == 0

    assert not (
        controller
        .has_unresolved_tti_boundary
    )


def test_training_runner_collects_after_warmup_and_updates():
    device = preferred_device()

    torch.manual_seed(
        1234
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            1234
        )

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    ) = build_small_training_system(
        device
    )

    provider = build_input_provider(
        device
    )

    common_runner_config = (
        PPOTrainingRunnerConfig(
            first_collection_tti_index=2,
        )
    )

    #
    # ----------------------------------------------------------
    # TTIs 0 and 1:
    # warm-up only.
    # ----------------------------------------------------------
    #

    warmup = run_single_cell_ppo_training(
        start_tti_index=0,
        num_ttis=2,
        input_provider=provider,
        traffic_manager=traffic_manager,
        tds_eligibility_config=(
            TDSBufferEligibilityConfig(
                mode="data_available_only",
            )
        ),
        state_manager=state_manager,
        training_controller=controller,
        state_config=state_config,
        greedy_config=(
            PPOGreedySearchConfig()
        ),
        reward_config=PPORewardConfig(
            geometric_mean_normalizer_bps=(
                100.0e6
            ),
        ),
        reward_population="candidates",
        runner_config=(
            common_runner_config
        ),
        device=device,
    )

    assert (
        warmup.final_agent_buffer_size
        == 0
    )

    #
    # ----------------------------------------------------------
    # TTIs 2, 3, 4:
    # experience collection is active.
    #
    # L = 2 and update size = 4.
    #
    # TTI 2 finish:
    #     1 resolved transition
    #
    # TTI 3 start:
    #     resolves previous last transition -> 2
    #
    # TTI 3 finish:
    #     one more -> 3
    #
    # TTI 4 start:
    #     resolves previous last -> 4
    #
    #     PPO UPDATE OCCURS HERE
    #
    # Then TTI 4 is collected under the new actor.
    # ----------------------------------------------------------
    #

    training = run_single_cell_ppo_training(
        start_tti_index=2,
        num_ttis=3,
        input_provider=provider,
        traffic_manager=traffic_manager,
        tds_eligibility_config=(
            TDSBufferEligibilityConfig(
                mode="data_available_only",
            )
        ),
        state_manager=state_manager,
        training_controller=controller,
        state_config=state_config,
        greedy_config=(
            PPOGreedySearchConfig()
        ),
        reward_config=PPORewardConfig(
            geometric_mean_normalizer_bps=(
                100.0e6
            ),
        ),
        reward_population="candidates",
        runner_config=(
            common_runner_config
        ),
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    assert training.num_warmup_ttis == 0

    assert training.num_collection_ttis == 3

    assert (
        training.first_collected_tti_index
        == 2
    )

    assert training.num_ppo_updates == 1

    assert controller.num_updates == 1

    #
    # After the update, TTI 4 begins a new
    # on-policy rollout.
    #
    # With L=2, finishing it leaves:
    #
    # one fully resolved transition
    # +
    # one unresolved final transition.
    #
    assert (
        training.final_agent_buffer_size
        == 1
    )

    assert (
        training
        .has_unresolved_tti_boundary
    )



def test_training_runner_populates_expert_buffer_only_after_warmup():
    device = preferred_device()

    torch.manual_seed(
        1234
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            1234
        )

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    ) = build_small_training_system(
        device
    )

    provider = build_input_provider(
        device
    )

    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=100,
            )
        )
    )

    runner_config = PPOTrainingRunnerConfig(
        first_collection_tti_index=2,
    )

    # ==========================================================
    # TTIs 0 and 1:
    # warm-up.
    # ==========================================================

    warmup = run_single_cell_ppo_training(
        start_tti_index=0,
        num_ttis=2,
        input_provider=provider,
        traffic_manager=traffic_manager,
        tds_eligibility_config=(
            TDSBufferEligibilityConfig(
                mode="data_available_only",
            )
        ),
        state_manager=state_manager,
        training_controller=controller,
        state_config=state_config,
        greedy_config=(
            PPOGreedySearchConfig()
        ),
        reward_config=PPORewardConfig(
            geometric_mean_normalizer_bps=(
                100.0e6
            ),
        ),
        reward_population="candidates",
        runner_config=runner_config,
        expert_buffer=expert_buffer,
        pf_expert_config=(
            PPOPFExpertConfig()
        ),
        device=device,
    )

    #
    # Warm-up evolves the environment but collects
    # neither PPO nor expert training data.
    #
    assert (
        warmup.num_collection_ttis
        == 0
    )

    assert len(
        expert_buffer
    ) == 0

    # ==========================================================
    # TTIs 2, 3, 4:
    # normal experience collection.
    #
    # Test system:
    #     2 user slots / TTI
    #
    # Therefore:
    #
    #     3 TTIs * 2 labels
    #     =
    #     6 expert demonstrations.
    # ==========================================================

    training = run_single_cell_ppo_training(
        start_tti_index=2,
        num_ttis=3,
        input_provider=provider,
        traffic_manager=traffic_manager,
        tds_eligibility_config=(
            TDSBufferEligibilityConfig(
                mode="data_available_only",
            )
        ),
        state_manager=state_manager,
        training_controller=controller,
        state_config=state_config,
        greedy_config=(
            PPOGreedySearchConfig()
        ),
        reward_config=PPORewardConfig(
            geometric_mean_normalizer_bps=(
                100.0e6
            ),
        ),
        reward_population="candidates",
        runner_config=runner_config,
        expert_buffer=expert_buffer,
        pf_expert_config=(
            PPOPFExpertConfig()
        ),
        device=device,
    )

    assert (
        training.num_collection_ttis
        == 3
    )

    assert len(
        expert_buffer
    ) == 6

    assert (
        expert_buffer.num_added_total
        == 6
    )

    #
    # Inspect one Teacher-2 mini-batch.
    #
    batch = expert_buffer.sample(
        batch_size=6,
    )

    assert tuple(
        batch.states.shape
    ) == (
        6,
        state_config.state_size,
    )

    assert tuple(
        batch.expert_actions.shape
    ) == (
        6,
        state_config.num_rbgs,
    )

    assert tuple(
        batch.action_masks.shape
    ) == (
        6,
        state_config.num_rbgs,
        state_config.num_candidates + 1,
    )

    #
    # Every stored expert action must be legal
    # under the mask from that exact state.
    #
    expert_is_legal = (
        batch
        .action_masks
        .gather(
            dim=2,
            index=(
                batch
                .expert_actions
                .unsqueeze(
                    -1
                )
            ),
        )
        .squeeze(
            -1
        )
    )

    assert torch.all(
        expert_is_legal
    )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    assert (
        batch.states.device
        == device
    )



def build_small_expert_training_system(
    device: torch.device,
):
    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=100,
            )
        )
    )

    expert_coordinator = (
        PPOExpertGuidanceCoordinator(
            expert_buffer=expert_buffer,
            guidance_config=(
                PPOExpertGuidanceConfig(
                    #
                    # Test-only small batch.
                    #
                    # b' is publicly unspecified.
                    #
                    batch_size=2,
                    guidance_weight=1.0,
                    divergence_mode="true_jsd",
                )
            ),
        )
    )

    state_config = OneLDSStateConfig(
        throughput_normalization_bps=(
            100.0e6
        ),
        buffer_normalization=8000.0,
        subband_cqi_normalization=15.0,
        num_candidates=2,
        num_rbgs=1,
        max_rank=2,
    )

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=(
                state_config.state_size
            ),
            hidden_size=8,
            num_rbgs=1,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=(
                state_config.state_size
            ),
            hidden_size=8,
        )
    ).to(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    controller = (
        OneLDSPPOTrainingController(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            agent_buffer=(
                PPOAgentUpdateCoordinator(
                    config=(
                        PPOAgentBufferConfig(
                            update_size=4,
                        )
                    )
                )
            ),
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            loss_config=PPOLossConfig(
                entropy_coefficient=0.0,
            ),
            config=(
                OneLDSPPOTrainingControllerConfig(
                    num_user_slots=2,
                )
            ),
            expert_guidance_coordinator=(
                expert_coordinator
            ),
        )
    )

    state_manager = (
        OneLDSCellTTIStateManager(
            initial_average_throughput_bps=(
                torch.full(
                    (
                        2,
                    ),
                    1.0e6,
                    dtype=torch.float32,
                    device=device,
                )
            ),
            serving_global_ue_indices=(
                torch.tensor(
                    [
                        0,
                        1,
                    ],
                    dtype=torch.long,
                    device=device,
                )
            ),
            serving_ue_valid_mask=(
                torch.ones(
                    2,
                    dtype=torch.bool,
                    device=device,
                )
            ),
            tds_config=PFTimeDomainConfig(
                num_candidates=2,
            ),
            throughput_forgetting_factor=0.5,
        )
    )

    traffic_manager = TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                True,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=1000,
            packet_arrival_rate_per_s=500.0,
            tti_duration_s=0.001,
        ),
        full_buffer_state_bits=8000.0,
        seed=1234,
    )

    return (
        state_config,
        controller,
        state_manager,
        traffic_manager,
        expert_buffer,
        expert_coordinator,
    )


def test_training_runner_alternates_ppo_and_expert_updates():
    device = preferred_device()

    torch.manual_seed(
        1234
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            1234
        )

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
        expert_buffer,
        expert_coordinator,
    ) = build_small_expert_training_system(
        device
    )

    provider = build_input_provider(
        device
    )

    runner_config = (
        PPOTrainingRunnerConfig(
            first_collection_tti_index=0,
        )
    )

    result = run_single_cell_ppo_training(
        start_tti_index=0,

        #
        # With two user slots and update_size=4:
        #
        # TTI0:
        #     one resolved + one pending
        #
        # TTI1 starts:
        #     pending becomes resolved
        #
        # TTI1 finishes:
        #     another resolved + pending
        #
        # TTI2 starts:
        #     fourth experience resolves
        #     ->
        #     PPO update
        #     ->
        #     expert update
        #     ->
        #     THEN TTI2 action is sampled
        #
        num_ttis=3,

        input_provider=provider,
        traffic_manager=traffic_manager,

        tds_eligibility_config=(
            TDSBufferEligibilityConfig(
                mode="data_available_only",
            )
        ),

        state_manager=state_manager,

        training_controller=(
            controller
        ),

        state_config=state_config,

        greedy_config=(
            PPOGreedySearchConfig()
        ),

        reward_config=PPORewardConfig(
            geometric_mean_normalizer_bps=(
                100.0e6
            ),
        ),

        reward_population="candidates",

        runner_config=runner_config,

        expert_buffer=expert_buffer,

        pf_expert_config=(
            PPOPFExpertConfig()
        ),

        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    #
    # Teacher 1 updated exactly once.
    #
    assert result.num_ppo_updates == 1

    assert controller.num_updates == 1

    #
    # Teacher 2 must update exactly once as part of
    # the SAME alternating update boundary.
    #
    assert (
        result
        .num_expert_guidance_updates
        == 1
    )

    assert (
        controller
        .num_expert_guidance_updates
        == 1
    )

    assert (
        expert_coordinator
        .num_guidance_updates
        == 1
    )

    assert (
        expert_coordinator
        .last_update
        is not None
    )

    assert (
        expert_coordinator
        .last_update
        .ppo_update_index
        == 1
    )

    #
    # Expert data are persistent.
    #
    # Three TTIs * two layers = six labels.
    #
    assert len(
        expert_buffer
    ) == 6


def test_training_runner_rejects_different_expert_buffers():
    device = preferred_device()

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
        controller_expert_buffer,
        _,
    ) = build_small_expert_training_system(
        device
    )

    different_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=100,
            )
        )
    )

    try:
        run_single_cell_ppo_training(
            start_tti_index=0,
            num_ttis=1,
            input_provider=(
                build_input_provider(
                    device
                )
            ),
            traffic_manager=(
                traffic_manager
            ),
            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode="data_available_only",
                )
            ),
            state_manager=state_manager,
            training_controller=controller,
            state_config=state_config,
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=(
                PPORewardConfig(
                    geometric_mean_normalizer_bps=(
                        100.0e6
                    ),
                )
            ),
            reward_population="candidates",
            runner_config=(
                PPOTrainingRunnerConfig(
                    first_collection_tti_index=0,
                )
            ),

            #
            # WRONG buffer intentionally.
            #
            expert_buffer=(
                different_buffer
            ),

            pf_expert_config=(
                PPOPFExpertConfig()
            ),

            device=device,
        )

    except ValueError as error:
        assert "same buffer" in str(
            error
        )

    else:
        raise AssertionError(
            "Expected mismatched expert buffers "
            "to be rejected."
        )

    assert len(
        controller_expert_buffer
    ) == 0



def test_real_two_cell_centralized_ppo_smoke():
    """
    End-to-end two-cell centralized PPO smoke test.

    Unlike the orchestration-only multi-cell test,
    this uses the existing real traffic-aware cell
    pipeline and its physical-input builder.

    TEST-SCALE configuration:
        - 2 cells
        - 2 1LDS user slots per cell
        - update size = 4
        - collection starts immediately
        - 2 TTIs

    Temporal count:

        TTI 0 finish:

            Cell 0:
                slot 0 resolved
                slot 1 pending

            Cell 1:
                slot 0 resolved
                slot 1 pending

            centralized buffer = 2

        TTI 1 begin:

            Cell 0 previous slot 1 resolves
            Cell 1 previous slot 1 resolves

            centralized buffer = 4

            ->
            one real centralized PPO update

        TTI 1 then runs under the updated policy.
    """

    device = preferred_device()

    if device.type != "cuda":
        #
        # This integration is intended to exercise
        # our GPU-first PHY/training path.
        #
        # If your normal CHPC test allocation has a
        # GPU, this branch should not be taken.
        #
        import pytest

        pytest.skip(
            "Real two-cell PPO smoke test requires "
            "cuda:0."
        )

    torch.manual_seed(
        1234
    )

    torch.cuda.manual_seed_all(
        1234
    )

    # ==========================================================
    # Build TWO independent environment states using the
    # already-established single-cell integration fixture.
    #
    # IMPORTANT:
    #     We are reusing only:
    #
    #         state configuration
    #         PF-history manager
    #         traffic manager
    #         physical input provider
    #
    #     We do NOT use the fixture's local single-cell PPO
    #     controller for learning.
    # ==========================================================

    (
        state_config_0,
        old_controller_0,
        state_manager_0,
        traffic_manager_0,
    ) = build_small_training_system(
        device
    )

    (
        state_config_1,
        old_controller_1,
        state_manager_1,
        traffic_manager_1,
    ) = build_small_training_system(
        device
    )

    #
    # Both cells must expose exactly the same neural-interface
    # dimensions because one shared actor is used.
    #
    assert (
        state_config_0
        == state_config_1
    )

    state_config = state_config_0

    # ==========================================================
    # Shared centralized actor + critic.
    #
    # Reuse the first fixture's networks only as initialized
    # model objects.
    #
    # The old single-cell controllers themselves are NOT used.
    # ==========================================================

    actor = old_controller_0.actor

    critic = old_controller_0.critic

    actor.train()
    critic.train()

    #
    # Create fresh optimizers so this smoke run begins with
    # clean Adam state.
    #
    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    # ==========================================================
    # Centralized transition storage.
    #
    # TEST SCALE:
    #
    #     2 cells
    #     x
    #     2 user slots
    #     =
    #     4 real transitions/update
    #
    # This deliberately aligns exactly for the integration
    # smoke test.
    #
    # It is NOT the paper's M=128 configuration.
    # ==========================================================

    transition_buffer = (
        PPOMultiStreamTransitionBuffer(
            config=(
                PPOMultiStreamBufferConfig(
                    num_streams=2,
                    update_size=4,
                )
            )
        )
    )

    # ==========================================================
    # One trajectory controller PER cell.
    #
    # Same actor.
    # Same critic.
    # Same centralized transition buffer.
    #
    # Different trajectory history.
    # ==========================================================

    rollout_controllers = tuple(
        OneLDSPPOMultiCellRolloutController(
            stream_id=cell_index,
            actor=actor,
            critic=critic,
            transition_buffer=(
                transition_buffer
            ),
            config=(
                OneLDSPPOMultiCellRolloutConfig(
                    num_user_slots=2,
                )
            ),
        )
        for cell_index
        in range(
            2
        )
    )

    # ==========================================================
    # Central learner.
    #
    # No expert guidance yet.
    # No candidate augmentation yet.
    #
    # This isolates the first real multi-cell PPO update.
    # ==========================================================

    centralized_training = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=(
                transition_buffer
            ),
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode="require_exact",
                )
            ),
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            loss_config=PPOLossConfig(
                entropy_coefficient=0.0,
            ),
        )
    )

    # ==========================================================
    # Give each cell its own input-provider instance.
    #
    # The existing provider builds the small real physical
    # inputs used by our established cell-training tests.
    # ==========================================================

    provider_0 = build_input_provider(
        device
    )

    provider_1 = build_input_provider(
        device
    )

    def multicell_input_provider(
        tti_index: int,
        cell_index: int,
    ):
        if cell_index == 0:
            return provider_0(
                tti_index
            )

        if cell_index == 1:
            return provider_1(
                tti_index
            )

        raise ValueError(
            "Unexpected cell_index."
        )

    # ==========================================================
    # Snapshot network parameters BEFORE learning.
    #
    # This lets us prove that an optimizer step genuinely
    # happened rather than merely counting an update.
    # ==========================================================

    actor_before = tuple(
        parameter
        .detach()
        .clone()
        for parameter
        in actor.parameters()
    )

    critic_before = tuple(
        parameter
        .detach()
        .clone()
        for parameter
        in critic.parameters()
    )

    # ==========================================================
    # ACTUAL RUN.
    # ==========================================================

    result = run_multicell_ppo_training(
        start_tti_index=0,

        #
        # TTI 0 collects.
        #
        # TTI 1 boundary resolves the two pending
        # previous-TTI transitions and triggers update #1.
        #
        num_ttis=2,

        input_provider=(
            multicell_input_provider
        ),

        traffic_managers=(
            traffic_manager_0,
            traffic_manager_1,
        ),

        state_managers=(
            state_manager_0,
            state_manager_1,
        ),

        rollout_controllers=(
            rollout_controllers
        ),

        actor=actor,

        critic=critic,

        optimizers=optimizers,

        centralized_training=(
            centralized_training
        ),

        tds_eligibility_config=(
            TDSBufferEligibilityConfig(
                mode="data_available_only",
            )
        ),

        state_config=state_config,

        greedy_config=(
            PPOGreedySearchConfig()
        ),

        reward_config=(
            PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    100.0e6
                ),
            )
        ),

        reward_population="candidates",

        #
        # TEST ONLY:
        # collect immediately.
        #
        # Paper training uses 100 warm-up TTIs.
        #
        runner_config=(
            PPOTrainingRunnerConfig(
                first_collection_tti_index=0,
            )
        ),

        #
        # Isolate vanilla centralized PPO first.
        #
        expert_buffer=None,
        pf_expert_config=None,

        device=device,
    )

    torch.cuda.synchronize(
        device
    )

    # ==========================================================
    # Assertions: did real centralized learning happen?
    # ==========================================================

    assert result.num_cells == 2

    assert (
        result.num_collection_ttis
        == 2
    )

    assert (
        result.num_ppo_updates
        == 1
    )

    assert (
        centralized_training
        .num_updates
        == 1
    )

    #
    # No Teacher-2 update was enabled.
    #
    assert (
        result
        .num_expert_guidance_updates
        == 0
    )

    # ==========================================================
    # The network must ACTUALLY have changed.
    # ==========================================================

    actor_changed = any(
        not torch.equal(
            before,
            after.detach(),
        )
        for before, after
        in zip(
            actor_before,
            actor.parameters(),
        )
    )

    critic_changed = any(
        not torch.equal(
            before,
            after.detach(),
        )
        for before, after
        in zip(
            critic_before,
            critic.parameters(),
        )
    )

    assert actor_changed

    assert critic_changed

    # ==========================================================
    # After update #1:
    #
    # TTI 1 generates:
    #
    #     Cell 0 slot 0 resolved
    #     Cell 1 slot 0 resolved
    #
    # while both final slot-1 decisions remain pending.
    #
    # Hence:
    #
    #     centralized buffer = 2
    # ==========================================================

    assert (
        result
        .final_transition_buffer_size
        == 2
    )

    assert (
        result
        .unresolved_boundary_by_cell
        == (
            True,
            True,
        )
    )

    # ==========================================================
    # Loss diagnostics must also be finite.
    # ==========================================================

    last_update = (
        centralized_training
        .last_update
    )

    assert last_update is not None

    assert torch.isfinite(
        last_update
        .ppo_update
        .actor_loss
    )

    assert torch.isfinite(
        last_update
        .ppo_update
        .critic_loss
    )

    assert torch.isfinite(
        last_update
        .ppo_update
        .mean_probability_ratio
    )


def test_real_two_cell_centralized_ppo_with_expert_and_permutation():
    """
    Real two-cell centralized PPO integration with
    the paper's two additional training mechanisms:

        1. PF expert guidance
        2. candidate-UE permutation augmentation

    TEST-SCALE configuration:

        cells:
            2

        user slots/cell:
            2

        centralized PPO update size:
            4 real transitions

        PPO candidate augmentation:
            original + 1 permutation

        PF-expert augmentation:
            original + 1 permutation

        expert mini-batch:
            2

    Expected first update:

        TTI 0:
            Cell 0:
                2 PF expert layer labels

            Cell 1:
                2 PF expert layer labels

            with original + 1 permutation:

                2 cells
                x 2 layers
                x 2 representations
                =
                8 expert demonstrations

        TTI 1 boundary:
            4 real PPO transitions become ready

            ->
            permutation augmentation

            4 real
            x 2 representations
            =
            8 PPO optimizer samples

            ->
            PPO actor + critic update

            ->
            expert JSD actor-only update

            ->
            TTI 1 actions use the twice-updated actor
    """

    device = preferred_device()

    if device.type != "cuda":
        import pytest

        pytest.skip(
            "Real two-cell expert PPO integration "
            "requires cuda:0."
        )

    torch.manual_seed(
        1234
    )

    torch.cuda.manual_seed_all(
        1234
    )

    # ==========================================================
    # Build two independent real cell environments.
    # ==========================================================

    (
        state_config_0,
        old_controller_0,
        state_manager_0,
        traffic_manager_0,
    ) = build_small_training_system(
        device
    )

    (
        state_config_1,
        old_controller_1,
        state_manager_1,
        traffic_manager_1,
    ) = build_small_training_system(
        device
    )

    assert (
        state_config_0
        == state_config_1
    )

    state_config = state_config_0

    # ==========================================================
    # Shared actor + critic.
    # ==========================================================

    actor = old_controller_0.actor

    critic = old_controller_0.critic

    actor.train()
    critic.train()

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    # ==========================================================
    # Centralized real-transition buffer.
    # ==========================================================

    transition_buffer = (
        PPOMultiStreamTransitionBuffer(
            config=(
                PPOMultiStreamBufferConfig(
                    num_streams=2,
                    update_size=4,
                )
            )
        )
    )

    # ==========================================================
    # Persistent shared expert buffer.
    #
    # This is D_expert.
    # ==========================================================

    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=100,
                sampling_with_replacement=False,
            )
        )
    )

    # ==========================================================
    # Teacher 2.
    #
    # Test-only batch_size=2.
    #
    # The paper's exact b' remains unspecified.
    # ==========================================================

    expert_guidance = (
        PPOExpertGuidanceCoordinator(
            expert_buffer=(
                expert_buffer
            ),
            guidance_config=(
                PPOExpertGuidanceConfig(
                    batch_size=2,

                    #
                    # TEST/REPRODUCTION parameter.
                    #
                    guidance_weight=1.0,

                    #
                    # Primary interpretation of the
                    # paper's JSD description.
                    #
                    divergence_mode="true_jsd",
                )
            ),
        )
    )

    # ==========================================================
    # One trajectory collector per cell.
    # ==========================================================

    rollout_controllers = tuple(
        OneLDSPPOMultiCellRolloutController(
            stream_id=cell_index,
            actor=actor,
            critic=critic,
            transition_buffer=(
                transition_buffer
            ),
            config=(
                OneLDSPPOMultiCellRolloutConfig(
                    num_user_slots=2,
                )
            ),
        )
        for cell_index
        in range(
            2
        )
    )

    # ==========================================================
    # Deterministic augmentation generators.
    #
    # Use separate generators for:
    #
    #     PPO augmentation
    #     expert-buffer augmentation
    #
    # so the two stochastic processes do not
    # accidentally share random-stream state.
    # ==========================================================

    ppo_augmentation_generator = (
        torch.Generator(
            device="cpu"
        )
    )

    ppo_augmentation_generator.manual_seed(
        1111
    )

    expert_augmentation_generator = (
        torch.Generator(
            device="cpu"
        )
    )

    expert_augmentation_generator.manual_seed(
        2222
    )

    augmentation_config = (
        PPOCandidateAugmentationConfig(
            num_permutations=1,
            include_original=True,
        )
    )

    # ==========================================================
    # Central learner.
    #
    # Candidate augmentation belongs HERE for
    # Teacher 1.
    # ==========================================================

    centralized_training = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=(
                transition_buffer
            ),
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode="require_exact",
                )
            ),
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            loss_config=PPOLossConfig(
                entropy_coefficient=0.0,
            ),
            candidate_augmentation_config=(
                augmentation_config
            ),
            candidate_augmentation_generator=(
                ppo_augmentation_generator
            ),
            expert_guidance_coordinator=(
                expert_guidance
            ),
        )
    )

    # ==========================================================
    # Real physical-input providers for the two cells.
    # ==========================================================

    provider_0 = build_input_provider(
        device
    )

    provider_1 = build_input_provider(
        device
    )

    def multicell_input_provider(
        tti_index: int,
        cell_index: int,
    ):
        if cell_index == 0:
            return provider_0(
                tti_index
            )

        if cell_index == 1:
            return provider_1(
                tti_index
            )

        raise ValueError(
            "Unexpected cell_index."
        )

    # ==========================================================
    # Save parameters so we can prove learning occurred.
    # ==========================================================

    actor_before = tuple(
        parameter
        .detach()
        .clone()
        for parameter
        in actor.parameters()
    )

    critic_before = tuple(
        parameter
        .detach()
        .clone()
        for parameter
        in critic.parameters()
    )

    # ==========================================================
    # RUN.
    # ==========================================================

    result = run_multicell_ppo_training(
        start_tti_index=0,

        #
        # TTI 0:
        #     collect
        #
        # TTI 1 boundary:
        #     centralized update
        #
        # TTI 1:
        #     collect under updated policy
        #
        num_ttis=2,

        input_provider=(
            multicell_input_provider
        ),

        traffic_managers=(
            traffic_manager_0,
            traffic_manager_1,
        ),

        state_managers=(
            state_manager_0,
            state_manager_1,
        ),

        rollout_controllers=(
            rollout_controllers
        ),

        actor=actor,

        critic=critic,

        optimizers=optimizers,

        centralized_training=(
            centralized_training
        ),

        tds_eligibility_config=(
            TDSBufferEligibilityConfig(
                mode="data_available_only",
            )
        ),

        state_config=state_config,

        greedy_config=(
            PPOGreedySearchConfig()
        ),

        reward_config=(
            PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    100.0e6
                ),
            )
        ),

        reward_population="candidates",

        #
        # TEST ONLY.
        #
        runner_config=(
            PPOTrainingRunnerConfig(
                first_collection_tti_index=0,
            )
        ),

        #
        # Teacher-2 data generation.
        #
        expert_buffer=(
            expert_buffer
        ),

        pf_expert_config=(
            PPOPFExpertConfig()
        ),

        #
        # Teacher-2 candidate augmentation.
        #
        expert_candidate_augmentation_config=(
            augmentation_config
        ),

        expert_candidate_augmentation_generator=(
            expert_augmentation_generator
        ),

        device=device,
    )

    torch.cuda.synchronize(
        device
    )

    # ==========================================================
    # CENTRAL PPO UPDATE
    # ==========================================================

    assert (
        result.num_ppo_updates
        == 1
    )

    assert (
        centralized_training
        .num_updates
        == 1
    )

    # ==========================================================
    # TEACHER-2 UPDATE
    # ==========================================================

    assert (
        result
        .num_expert_guidance_updates
        == 1
    )

    assert (
        expert_guidance
        .num_guidance_updates
        == 1
    )

    # ==========================================================
    # PPO permutation augmentation.
    #
    # Four REAL transitions:
    #
    #     2 cells x 2 slots
    #
    # each contributes:
    #
    #     original + 1 permutation
    #
    # -> 8 optimizer rows.
    # ==========================================================

    last_update = (
        centralized_training
        .last_update
    )

    assert last_update is not None

    assert (
        last_update.num_real_transitions
        == 4
    )

    assert (
        last_update.num_optimizer_samples
        == 8
    )

    # ==========================================================
    # Expert demonstrations.
    #
    # Per TTI:
    #
    #     2 cells
    #     x 2 expert layer labels
    #     x 2 representations
    #     =
    #     8 demonstrations
    #
    # Two TTIs:
    #
    #     16 demonstrations total.
    #
    # Expert sampling does NOT remove demonstrations.
    # ==========================================================

    assert (
        result
        .num_expert_demonstrations_added
        == 16
    )

    assert len(
        expert_buffer
    ) == 16

    # ==========================================================
    # Both networks really changed.
    # ==========================================================

    actor_changed = any(
        not torch.equal(
            before,
            after.detach(),
        )
        for before, after
        in zip(
            actor_before,
            actor.parameters(),
        )
    )

    critic_changed = any(
        not torch.equal(
            before,
            after.detach(),
        )
        for before, after
        in zip(
            critic_before,
            critic.parameters(),
        )
    )

    assert actor_changed

    assert critic_changed

    # ==========================================================
    # Teacher 2 was attached to the same PPO update.
    # ==========================================================

    assert (
        last_update.expert_update
        is not None
    )

    assert (
        last_update
        .expert_update
        .ppo_update_index
        == 1
    )

    # ==========================================================
    # PPO diagnostics.
    # ==========================================================

    assert torch.isfinite(
        last_update
        .ppo_update
        .actor_loss
    )

    assert torch.isfinite(
        last_update
        .ppo_update
        .critic_loss
    )

    assert torch.isfinite(
        last_update
        .ppo_update
        .mean_probability_ratio
    )

    # ==========================================================
    # Teacher-2 diagnostics.
    # ==========================================================

    expert_loss = (
        last_update
        .expert_update
        .guidance_update
        .loss_data
    )

    assert torch.isfinite(
        expert_loss.raw_loss
    )

    assert torch.isfinite(
        expert_loss.weighted_loss
    )

    # ==========================================================
    # TTI 1 starts a fresh on-policy rollout under
    # the updated actor.
    # ==========================================================

    assert (
        result
        .final_transition_buffer_size
        == 2
    )

    assert (
        result
        .unresolved_boundary_by_cell
        == (
            True,
            True,
        )
    )




