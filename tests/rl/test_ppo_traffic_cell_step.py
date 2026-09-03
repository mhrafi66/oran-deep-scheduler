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
from oran_scheduler.rl.ppo_traffic_cell_step import (
    run_traffic_aware_ppo_cell_tti_step,
)
from oran_scheduler.rl.ppo_training_controller import (
    OneLDSPPOTrainingController,
    OneLDSPPOTrainingControllerConfig,
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
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
    OneLDSCellTTIStateManager,
    PreparedOneLDSCellTTI,
)
from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
    TrafficBufferManager,
)

from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)
from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)

def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def build_three_ue_channel(
    device: torch.device,
) -> torch.Tensor:
    h_freq = torch.zeros(
        (
            1,
            3,
            1,
            1,
            3,
            1,
            24,
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

    h_freq[
        0,
        2,
        0,
        0,
        2,
        0,
        :,
    ] = 1.0

    return h_freq


def build_observation(
    device: torch.device,
) -> OneLDSCellTTIObservation:
    directions = torch.zeros(
        (
            3,
            2,
            1,
            3,
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        0,
        :,
        0,
        0,
    ] = 1.0

    directions[
        1,
        :,
        0,
        1,
    ] = 1.0

    directions[
        2,
        :,
        0,
        2,
    ] = 1.0

    return OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            torch.tensor(
                [
                    0,
                    1,
                    2,
                ],
                dtype=torch.long,
                device=device,
            )
        ),
        serving_ue_valid_mask=(
            torch.ones(
                3,
                dtype=torch.bool,
                device=device,
            )
        ),
        td_instantaneous_rate_bps=(
            torch.tensor(
                [
                    10.0e6,
                    8.0e6,
                    6.0e6,
                ],
                dtype=torch.float32,
                device=device,
            )
        ),
        rank=torch.ones(
            3,
            dtype=torch.long,
            device=device,
        ),

        #
        # This placeholder is intentionally ignored
        # by the traffic-aware step.
        #
        dl_buffer=torch.zeros(
            3,
            dtype=torch.float32,
            device=device,
        ),

        wideband_cqi=torch.full(
            (
                3,
            ),
            10.0,
            dtype=torch.float32,
            device=device,
        ),
        subband_cqi=torch.full(
            (
                3,
                2,
            ),
            10.0,
            dtype=torch.float32,
            device=device,
        ),
        precoder_directions=(
            directions
        ),
    )

def build_system(
    device: torch.device,
):
    state_config = OneLDSStateConfig(
        throughput_normalization_bps=(
            100.0e6
        ),

        #
        # Integration-test normalization only.
        #
        buffer_normalization=4000.0,

        subband_cqi_normalization=15.0,
        num_candidates=2,
        num_rbgs=2,
        max_rank=2,
    )

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=(
                state_config.state_size
            ),
            hidden_size=8,
            num_rbgs=2,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    #
    # Deterministic integration-test actor:
    #
    # candidate 0 preferred;
    # once masked, candidate 1 preferred;
    # then NO ALLOCATION.
    #
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()

        actor.network[-1].bias.copy_(
            torch.tensor(
                [
                    1000.0,
                    0.0,
                    -1000.0,

                    1000.0,
                    0.0,
                    -1000.0,
                ],
                dtype=torch.float32,
                device=device,
            )
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
                            update_size=128,
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
                    num_user_slots=4,
                )
            ),
        )
    )

    state_manager = (
        OneLDSCellTTIStateManager(
            initial_average_throughput_bps=(
                torch.full(
                    (
                        3,
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
                        2,
                    ],
                    dtype=torch.long,
                    device=device,
                )
            ),
            serving_ue_valid_mask=(
                torch.ones(
                    3,
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

    #
    # UE 0 = Full Buffer
    # UE 1 = FTP
    # UE 2 = FTP
    #
    traffic_manager = TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        ftp3_config=FTP3TrafficConfig(
            #
            # 500 bytes = 4000 bits.
            #
            packet_size_bytes=500,
            packet_arrival_rate_per_s=500.0,
            tti_duration_s=0.001,
        ),

        #
        # Scheduler-state proxy only.
        #
        full_buffer_state_bits=4000.0,
        seed=1234,
    )

    return (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    )


def build_physical_input_factory(
    *,
    device: torch.device,
    h_freq: torch.Tensor,
):
    def build_physical_inputs(
        prepared: PreparedOneLDSCellTTI,
    ) -> PPOPhysicalScoreInputs:
        return PPOPhysicalScoreInputs(
            candidate_global_ue_indices=(
                prepared
                .candidate_global_ue_indices
            ),
            h_freq=h_freq,
            serving_cell_index=0,
            recommended_rank=(
                torch.ones(
                    (
                        1,
                        3,
                    ),
                    dtype=torch.long,
                    device=device,
                )
            ),
            rx_combiners=(
                torch.ones(
                    (
                        1,
                        3,
                        2,
                        2,
                        1,
                    ),
                    dtype=torch.complex64,
                    device=device,
                )
            ),
            csi_subcarrier_index=6,
            subcarriers_per_rbg=12,
            tx_power_per_subcarrier_w=2.0,
            noise_power_per_subcarrier_w=(
                1.0e-3
            ),
            link_adaptation_config=(
                LinkAdaptationConfig(
                    num_rbgs=2,
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

    return build_physical_inputs


def test_mixed_traffic_tti_uses_actual_delivery():
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
    ) = build_system(
        device
    )

    observation = build_observation(
        device
    )

    h_freq = build_three_ue_channel(
        device
    )

    physical_builder = (
        build_physical_input_factory(
            device=device,
            h_freq=h_freq,
        )
    )

    result = (
        run_traffic_aware_ppo_cell_tti_step(
            tti_index=0,
            observation=observation,
            traffic_manager=traffic_manager,
            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode="all_serving_ues",
                )
            ),
            state_manager=state_manager,
            training_controller=controller,
            state_config=state_config,
            physical_inputs_builder=(
                physical_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    100.0e6
                ),
            ),
            reward_population="candidates",
            reward_reduction="mean",

            #
            # UE 1 and UE 2 each receive exactly one
            # 4000-bit FTP packet.
            #
            packet_arrivals=torch.tensor(
                [
                    0,
                    1,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            device=device,
        )
    )

    #
    # Equal initial PF histories and rates
    # 10 > 8 > 6 select UE 0 and UE 1.
    #
    torch.testing.assert_close(
        result
        .prepared
        .candidate_global_ue_indices,
        torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    #
    # Real buffer state reached the scheduler.
    #
    torch.testing.assert_close(
        result
        .scheduler_observation
        .dl_buffer,
        torch.tensor(
            [
                4000.0,
                4000.0,
                4000.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # UE 2 was not in the candidate set, therefore
    # it received no physical service.
    #
    torch.testing.assert_close(
        result
        .serving_offered_capacity_bps[
            2
        ],
        torch.tensor(
            0.0,
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # Candidate 1 is FTP UE 1.
    #
    # It has only 4000 bits for a 1 ms TTI, so its
    # actual delivered rate cannot exceed 4 Mbps.
    #
    assert (
        result
        .physical_outcome
        .candidate_total_target_compliant_rate_bps[
            1
        ]
        >= result
        .candidate_delivered_rate_bps[
            1
        ]
    )

    torch.testing.assert_close(
        result
        .candidate_delivered_rate_bps[
            1
        ],
        torch.tensor(
            4.0e6,
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # UE 2's packet remains queued.
    #
    torch.testing.assert_close(
        result
        .traffic_service
        .buffer_after_service_bits[
            2
        ],
        torch.tensor(
            4000.0,
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # FTP UE 1 was completely drained.
    #
    torch.testing.assert_close(
        result
        .traffic_service
        .buffer_after_service_bits[
            1
        ],
        torch.tensor(
            0.0,
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # PF history receives ACTUAL delivered rate,
    # mapped back into persistent serving-UE order.
    #
    # In this test:
    #
    #     candidate 0 -> serving UE 0
    #     candidate 1 -> serving UE 1
    #
    # FTP UE 1 could actually deliver only 4 Mbps.
    #
    torch.testing.assert_close(
        result
        .history_update
        .delivered_rate_bps[
            1
        ],
        torch.tensor(
            4.0e6,
            dtype=torch.float32,
            device=device,
        ),
    )


    #
    # Previous PF history for UE 1:
    #     1 Mbps
    #
    # Actual delivered rate:
    #     4 Mbps
    #
    # forgetting_factor = 0.5
    #
    # R_new
    #     =
    # 0.5 * R_old
    # +
    # 0.5 * delivered
    #
    #     =
    # 0.5 * 1 Mbps
    # +
    # 0.5 * 4 Mbps
    #
    #     =
    # 2.5 Mbps
    #
    torch.testing.assert_close(
        result
        .history_update
        .updated_average_throughput_bps[
            1
        ],
        torch.tensor(
            2.5e6,
            dtype=torch.float32,
            device=device,
        ),
    )

    assert (
        result
        .reward
        .reward_population_name
        ==
        "pf_tds_candidates_actual_delivery"
    )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )


def test_traffic_buffer_changes_next_tti_state():
    device = preferred_device()

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    ) = build_system(
        device
    )

    observation = build_observation(
        device
    )

    physical_builder = (
        build_physical_input_factory(
            device=device,
            h_freq=(
                build_three_ue_channel(
                    device
                )
            ),
        )
    )

    reward_config = PPORewardConfig(
        geometric_mean_normalizer_bps=(
            100.0e6
        ),
    )

    tti_0 = (
        run_traffic_aware_ppo_cell_tti_step(
            tti_index=0,
            observation=observation,
            traffic_manager=traffic_manager,

            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode="all_serving_ues",
                )
            ),
            state_manager=state_manager,
            training_controller=controller,
            state_config=state_config,
            physical_inputs_builder=(
                physical_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=reward_config,
            reward_population="candidates",
            packet_arrivals=torch.tensor(
                [
                    0,
                    1,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            device=device,
        )
    )

    #
    # TTI 0:
    #
    # UE 1's queue drained.
    # UE 2's packet remained.
    #
    torch.testing.assert_close(
        traffic_manager.current_buffer_bits,
        torch.tensor(
            [
                4000.0,
                0.0,
                4000.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    tti_1 = (
        run_traffic_aware_ppo_cell_tti_step(
            tti_index=1,
            observation=observation,
            traffic_manager=traffic_manager,

            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode="all_serving_ues",
                )
            ),
            state_manager=state_manager,
            training_controller=controller,
            state_config=state_config,
            physical_inputs_builder=(
                physical_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=reward_config,
            reward_population="candidates",

            #
            # No new FTP arrivals.
            #
            packet_arrivals=torch.tensor(
                [
                    0,
                    0,
                    0,
                ],
                dtype=torch.long,
                device=device,
            ),
            device=device,
        )
    )

    torch.testing.assert_close(
        tti_1
        .scheduler_observation
        .dl_buffer,
        torch.tensor(
            [
                4000.0,
                0.0,
                4000.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # UE 2 was unserved in TTI 0 while its PF
    # history decayed. It should now enter the
    # candidate set.
    #
    assert torch.any(
        tti_1
        .prepared
        .candidate_global_ue_indices
        == 2
    )

    #
    # The first PPO state genuinely changes across
    # TTIs due to both:
    #
    #     throughput history
    # and
    #     traffic-buffer evolution.
    #
    state_0 = (
        tti_0
        .schedule
        .decisions[0]
        .state_data
        .state
    )

    state_1 = (
        tti_1
        .schedule
        .decisions[0]
        .state_data
        .state
    )

    assert not torch.equal(
        state_0,
        state_1,
    )

    #
    # Same PPO temporal bookkeeping as before:
    #
    # after two complete 4-layer TTIs:
    #
    # 7 transitions are fully resolved;
    # TTI-1 slot 3 awaits TTI-2 slot 0.
    #
    assert (
        controller.agent_buffer_size
        == 7
    )

    assert (
        controller
        .has_unresolved_tti_boundary
    )

def test_serving_ue_reward_population_is_explicit():
    device = preferred_device()

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    ) = build_system(
        device
    )

    observation = build_observation(
        device
    )

    physical_builder = (
        build_physical_input_factory(
            device=device,
            h_freq=(
                build_three_ue_channel(
                    device
                )
            ),
        )
    )

    result = (
        run_traffic_aware_ppo_cell_tti_step(
            tti_index=0,
            observation=observation,
            traffic_manager=traffic_manager,
            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode="all_serving_ues",
                )
            ),
            state_manager=state_manager,
            training_controller=controller,
            state_config=state_config,
            physical_inputs_builder=(
                physical_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    100.0e6
                ),
            ),
            reward_population="serving_ues",
            packet_arrivals=torch.tensor(
                [
                    0,
                    1,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            device=device,
        )
    )

    assert (
        result
        .reward
        .reward_population_name
        ==
        "serving_ues_actual_delivery"
    )

    #
    # UE 2 was not scheduled, so with all serving UEs
    # in the geometric mean, one throughput is zero.
    #
    torch.testing.assert_close(
        result
        .reward
        .reward_data
        .geometric_mean_throughput_bps,
        torch.tensor(
            0.0,
            dtype=torch.float32,
            device=device,
        ),
    )

def test_data_available_policy_excludes_drained_ftp_next_tti():
    device = preferred_device()

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    ) = build_system(
        device
    )

    observation = build_observation(
        device
    )

    physical_builder = (
        build_physical_input_factory(
            device=device,
            h_freq=(
                build_three_ue_channel(
                    device
                )
            ),
        )
    )

    reward_config = PPORewardConfig(
        geometric_mean_normalizer_bps=(
            100.0e6
        ),
    )

    eligibility_config = (
        TDSBufferEligibilityConfig(
            mode="data_available_only",
        )
    )

    # ==========================================================
    # TTI 0
    #
    # UE 0 = Full Buffer
    # UE 1 = FTP + one packet
    # UE 2 = FTP + one packet
    #
    # Initial PF:
    #     UE0 > UE1 > UE2
    #
    # so candidates are UE0, UE1.
    # ==========================================================

    tti_0 = (
        run_traffic_aware_ppo_cell_tti_step(
            tti_index=0,
            observation=observation,
            traffic_manager=traffic_manager,
            tds_eligibility_config=(
                eligibility_config
            ),
            state_manager=state_manager,
            training_controller=controller,
            state_config=state_config,
            physical_inputs_builder=(
                physical_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=reward_config,
            reward_population="candidates",
            packet_arrivals=torch.tensor(
                [
                    0,
                    1,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            device=device,
        )
    )

    torch.testing.assert_close(
        tti_0
        .tds_eligibility
        .eligible_mask,
        torch.tensor(
            [
                True,
                True,
                True,
            ],
            dtype=torch.bool,
            device=device,
        ),
    )

    #
    # UE1 was selected and drained.
    # UE2 remained unserved with its packet queued.
    #
    torch.testing.assert_close(
        traffic_manager.current_buffer_bits,
        torch.tensor(
            [
                4000.0,
                0.0,
                4000.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    # ==========================================================
    # TTI 1
    #
    # No new FTP arrivals.
    #
    # data_available_only should make:
    #
    # UE0 FB        -> eligible
    # UE1 FTP empty -> INELIGIBLE
    # UE2 FTP data  -> eligible
    # ==========================================================

    tti_1 = (
        run_traffic_aware_ppo_cell_tti_step(
            tti_index=1,
            observation=observation,
            traffic_manager=traffic_manager,
            tds_eligibility_config=(
                eligibility_config
            ),
            state_manager=state_manager,
            training_controller=controller,
            state_config=state_config,
            physical_inputs_builder=(
                physical_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=reward_config,
            reward_population="candidates",
            packet_arrivals=torch.tensor(
                [
                    0,
                    0,
                    0,
                ],
                dtype=torch.long,
                device=device,
            ),
            device=device,
        )
    )

    torch.testing.assert_close(
        tti_1
        .tds_eligibility
        .eligible_mask,
        torch.tensor(
            [
                True,
                False,
                True,
            ],
            dtype=torch.bool,
            device=device,
        ),
    )

    #
    # Only two UEs are eligible and K=2, therefore
    # PF-TDS must return exactly UE0 and UE2.
    #
    assert set(
        tti_1
        .prepared
        .candidate_global_ue_indices
        .tolist()
    ) == {
        0,
        2,
    }

    assert not torch.any(
        tti_1
        .prepared
        .candidate_global_ue_indices
        == 1
    )


def test_data_available_policy_allows_padded_candidate_slots():
    device = preferred_device()

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    ) = build_system(
        device
    )

    observation = build_observation(
        device
    )

    physical_builder = (
        build_physical_input_factory(
            device=device,
            h_freq=(
                build_three_ue_channel(
                    device
                )
            ),
        )
    )

    result = (
        run_traffic_aware_ppo_cell_tti_step(
            tti_index=0,
            observation=observation,
            traffic_manager=traffic_manager,
            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode="data_available_only",
                )
            ),
            state_manager=state_manager,
            training_controller=controller,
            state_config=state_config,
            physical_inputs_builder=(
                physical_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    100.0e6
                ),
            ),
            reward_population="candidates",

            #
            # Both FTP UEs empty.
            # Only UE0 Full Buffer is active.
            #
            packet_arrivals=torch.tensor(
                [
                    0,
                    0,
                    0,
                ],
                dtype=torch.long,
                device=device,
            ),
            device=device,
        )
    )

    assert (
        result
        .tds_eligibility
        .num_eligible_ues
        == 1
    )

    assert (
        result
        .prepared
        .candidate_valid_mask
        .sum()
        .item()
        == 1
    )

    torch.testing.assert_close(
        result
        .prepared
        .candidate_global_ue_indices[
            0
        ],
        torch.tensor(
            0,
            dtype=torch.long,
            device=device,
        ),
    )

    assert (
        result
        .prepared
        .candidate_global_ue_indices[
            1
        ]
        .item()
        == -1
    )


def test_traffic_tti_adds_one_expert_demo_per_layer():
    device = preferred_device()

    (
        state_config,
        controller,
        state_manager,
        traffic_manager,
    ) = build_system(
        device
    )

    observation = build_observation(
        device
    )

    physical_builder = (
        build_physical_input_factory(
            device=device,
            h_freq=(
                build_three_ue_channel(
                    device
                )
            ),
        )
    )

    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=100,
            )
        )
    )

    result = (
        run_traffic_aware_ppo_cell_tti_step(
            tti_index=0,
            collect_experience=True,
            observation=observation,
            traffic_manager=traffic_manager,
            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode="all_serving_ues",
                )
            ),
            state_manager=state_manager,
            training_controller=controller,
            state_config=state_config,
            physical_inputs_builder=(
                physical_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    100.0e6
                ),
            ),
            reward_population="candidates",
            expert_buffer=expert_buffer,
            pf_expert_config=(
                PPOPFExpertConfig()
            ),
            packet_arrivals=torch.tensor(
                [
                    0,
                    1,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            device=device,
        )
    )

    #
    # build_system() uses four PPO user slots.
    #
    assert (
        result
        .num_expert_demonstrations_added
        == 4
    )

    assert len(
        expert_buffer
    ) == 4

    assert (
        result.expert_labels
        is not None
    )

    assert len(
        result.expert_labels.layers
    ) == 4

    for layer_index, label in enumerate(
        result.expert_labels.layers
    ):
        assert (
            label.user_slot_index
            == layer_index
        )

        torch.testing.assert_close(
            label.action_mask,
            result
            .schedule
            .decisions[
                layer_index
            ]
            .action_mask,
        )


