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
from oran_scheduler.rl.ppo_single_cell_step import (
    run_full_buffer_ppo_cell_tti_step,
)
from oran_scheduler.rl.ppo_training_controller import (
    OneLDSPPOTrainingController,
    OneLDSPPOTrainingControllerConfig,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizerConfig,
    create_ppo_optimizers,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
    OneLDSCellTTIStateManager,
    PreparedOneLDSCellTTI,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
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
    """
    Three orthogonal single-RX UEs and one BS with
    three TX dimensions.

    Shape:
        [batch, UE, RX, BS, TX, symbol, subcarrier]
    """

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
        dl_buffer=torch.ones(
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
        buffer_normalization=1.0,
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
    # Test-only initialization:
    #
    # slot 0 strongly selects candidate 0.
    # Once candidate 0 becomes masked, slot 1
    # strongly selects candidate 1.
    # Once both are used, NO ALLOCATION is the only
    # legal option.
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

    agent_buffer = (
        PPOAgentUpdateCoordinator(
            config=PPOAgentBufferConfig(
                update_size=128,
            )
        )
    )

    training_controller = (
        OneLDSPPOTrainingController(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            agent_buffer=agent_buffer,
            gae_config=PPOGAEConfig(
                #
                # PAPER-UNSPECIFIED.
                # Integration-test choice only.
                #
                gae_lambda=0.9,
            ),
            loss_config=PPOLossConfig(
                #
                # PAPER-UNSPECIFIED.
                # Integration-test choice only.
                #
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
            tds_config=(
                PFTimeDomainConfig(
                    num_candidates=2,
                )
            ),
            #
            # OPEN-REPRODUCTION test choice.
            #
            throughput_forgetting_factor=0.5,
        )
    )

    return (
        state_config,
        training_controller,
        state_manager,
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



def test_two_tti_real_physical_ppo_environment_loop():
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
        training_controller,
        state_manager,
    ) = build_system(
        device
    )

    observation = build_observation(
        device
    )

    h_freq = build_three_ue_channel(
        device
    )

    physical_inputs_builder = (
        build_physical_input_factory(
            device=device,
            h_freq=h_freq,
        )
    )

    reward_config = PPORewardConfig(
        #
        # PAPER-UNSPECIFIED G_max.
        #
        # Explicit integration-test value only.
        #
        geometric_mean_normalizer_bps=(
            100.0e6
        ),
    )

    # ==========================================================
    # TTI 0
    # ==========================================================

    tti_0 = (
        run_full_buffer_ppo_cell_tti_step(
            tti_index=0,
            observation=observation,
            state_manager=state_manager,
            training_controller=(
                training_controller
            ),
            state_config=state_config,
            physical_inputs_builder=(
                physical_inputs_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=reward_config,
            reward_reduction="mean",
            device=device,
        )
    )

    torch.testing.assert_close(
        tti_0
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

    assert tuple(
        tti_0.schedule.actions.shape
    ) == (
        4,
        2,
    )

    assert tuple(
        tti_0
        .physical_outcome
        .candidate_total_target_compliant_rate_bps
        .shape
    ) == (
        2,
    )

    assert torch.all(
        tti_0
        .physical_outcome
        .candidate_total_target_compliant_rate_bps
        > 0.0
    )

    assert tuple(
        tti_0
        .reward
        .reward_data
        .reward_by_layer_rbg
        .shape
    ) == (
        4,
        2,
    )

    assert tuple(
        tti_0
        .reward
        .reduced_reward
        .shape
    ) == (
        4,
    )

    assert (
        tti_0
        .reward
        .reward_population_name
        ==
        "pf_tds_candidates_full_buffer_capacity"
    )

    #
    # After finishing TTI 0:
    #
    # slots 0,1,2 have known next states;
    # slot 3 still waits for TTI 1 slot-0.
    #
    assert (
        training_controller
        .agent_buffer_size
        == 3
    )

    assert (
        training_controller
        .has_unresolved_tti_boundary
    )

    first_state_tti_0 = (
        tti_0
        .schedule
        .decisions[0]
        .state_data
        .state
        .detach()
        .clone()
    )

    history_after_tti_0 = (
        state_manager
        .current_average_throughput_bps
    )

    assert not torch.equal(
        history_after_tti_0,
        torch.full(
            (
                3,
            ),
            1.0e6,
            dtype=torch.float32,
            device=device,
        ),
    )

    # ==========================================================
    # TTI 1
    # ==========================================================

    tti_1 = (
        run_full_buffer_ppo_cell_tti_step(
            tti_index=1,
            observation=observation,
            state_manager=state_manager,
            training_controller=(
                training_controller
            ),
            state_config=state_config,
            physical_inputs_builder=(
                physical_inputs_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=reward_config,
            reward_reduction="mean",
            device=device,
        )
    )

    #
    # UE 2 received no service in TTI 0, therefore
    # its PF history decayed while the scheduled UEs
    # received real physical throughput.
    #
    # It should now enter the TDS candidate set.
    #
    assert torch.any(
        tti_1
        .prepared
        .candidate_global_ue_indices
        == 2
    )

    first_state_tti_1 = (
        tti_1
        .schedule
        .decisions[0]
        .state_data
        .state
    )

    assert not torch.equal(
        first_state_tti_0,
        first_state_tti_1,
    )

    #
    # Starting TTI 1 resolved TTI 0's final
    # transition:
    #
    # 3 previous resolved
    # + 1 cross-TTI boundary
    # + 3 newly resolved from TTI 1
    # =
    # 7 transitions.
    #
    assert (
        training_controller
        .agent_buffer_size
        == 7
    )

    assert (
        training_controller
        .has_unresolved_tti_boundary
    )

    assert (
        training_controller.num_updates
        == 0
    )

    assert torch.isfinite(
        tti_1
        .reward
        .reduced_reward
    ).all()

    assert torch.isfinite(
        state_manager
        .current_average_throughput_bps
    ).all()

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    assert (
        first_state_tti_1.device
        == device
    )


def test_physical_candidate_order_must_match_tds():
    device = preferred_device()

    (
        state_config,
        training_controller,
        state_manager,
    ) = build_system(
        device
    )

    observation = build_observation(
        device
    )

    h_freq = build_three_ue_channel(
        device
    )

    def broken_builder(
        prepared: PreparedOneLDSCellTTI,
    ) -> PPOPhysicalScoreInputs:
        wrong_candidates = (
            prepared
            .candidate_global_ue_indices
            .flip(
                dims=[
                    0,
                ]
            )
        )

        return PPOPhysicalScoreInputs(
            candidate_global_ue_indices=(
                wrong_candidates
            ),
            h_freq=h_freq,
            serving_cell_index=0,
            recommended_rank=torch.ones(
                (
                    1,
                    3,
                ),
                dtype=torch.long,
                device=device,
            ),
            rx_combiners=torch.ones(
                (
                    1,
                    3,
                    2,
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

    try:
        run_full_buffer_ppo_cell_tti_step(
            tti_index=0,
            observation=observation,
            state_manager=state_manager,
            training_controller=(
                training_controller
            ),
            state_config=state_config,
            physical_inputs_builder=(
                broken_builder
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
            reward_config=PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    100.0e6
                ),
            ),
            device=device,
        )

    except ValueError:
        return

    raise AssertionError(
        "Mismatched PHY candidate ordering must "
        "raise ValueError."
    )


