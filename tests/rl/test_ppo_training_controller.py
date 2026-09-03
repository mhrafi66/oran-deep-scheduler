import pytest
import torch

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
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
)
from oran_scheduler.rl.ppo_training_controller import (
    OneLDSPPOTrainingController,
    OneLDSPPOTrainingControllerConfig,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizerConfig,
    create_ppo_optimizers,
)
from oran_scheduler.schedulers.one_lds_loop import (
    run_1lds_user_slot_loop,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)
from oran_scheduler.state.one_lds_decision import (
    OneLDSDecisionInputs,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def build_test_inputs(
    device: torch.device,
) -> OneLDSDecisionInputs:
    directions = torch.zeros(
        (
            2,
            2,
            1,
            2,
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

    return OneLDSDecisionInputs(
        past_average_throughput=torch.tensor(
            [
                1.0,
                1.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        rank=torch.tensor(
            [
                1,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
        dl_buffer=torch.tensor(
            [
                1.0,
                1.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        wideband_cqi=torch.tensor(
            [
                10.0,
                10.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        subband_cqi=torch.tensor(
            [
                [
                    10.0,
                    10.0,
                ],
                [
                    10.0,
                    10.0,
                ],
            ],
            dtype=torch.float32,
            device=device,
        ),
        candidate_precoder_directions=(
            directions
        ),
        candidate_valid_mask=torch.tensor(
            [
                True,
                True,
            ],
            dtype=torch.bool,
            device=device,
        ),
    )


def build_controller(
    device: torch.device,
    *,
    update_size: int = 4,
):
    state_config = OneLDSStateConfig(
        throughput_normalization_bps=1.0,
        buffer_normalization=1.0,
        subband_cqi_normalization=15.0,
        num_candidates=2,
        num_rbgs=2,
        max_rank=2,
    )

    state_size = (
        state_config.state_size
    )

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=state_size,
            hidden_size=8,
            num_rbgs=2,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=state_size,
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
                update_size=update_size,
            )
        )
    )

    controller = (
        OneLDSPPOTrainingController(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            agent_buffer=agent_buffer,
            gae_config=PPOGAEConfig(
                #
                # PAPER-UNSPECIFIED.
                # Unit-test choice only.
                #
                gae_lambda=0.9,
            ),
            loss_config=PPOLossConfig(
                #
                # PAPER-UNSPECIFIED.
                # Zero isolates PPO mechanics.
                #
                entropy_coefficient=0.0,
            ),
            config=(
                OneLDSPPOTrainingControllerConfig(
                    num_user_slots=2,
                )
            ),
        )
    )

    return (
        controller,
        actor,
        critic,
        state_config,
    )


def test_controller_rejects_update_size_not_divisible_by_slots():
    device = preferred_device()

    with pytest.raises(
        ValueError
    ):
        build_controller(
            device,
            update_size=3,
        )


def test_controller_collects_one_complete_tti():
    device = preferred_device()

    (
        controller,
        _,
        _,
        state_config,
    ) = build_controller(
        device
    )

    result = run_1lds_user_slot_loop(
        num_user_slots=2,
        inputs=build_test_inputs(
            device
        ),
        state_config=state_config,
        action_policy=(
            controller.make_action_policy(
                tti_index=0,
            )
        ),
        device=device,
    )

    assert tuple(
        result.actions.shape
    ) == (
        2,
        2,
    )

    controller.finish_tti(
        reward_by_rbg=torch.tensor(
            [
                [
                    0.2,
                    -0.2,
                ],
                [
                    0.2,
                    0.2,
                ],
            ],
            dtype=torch.float32,
            device=device,
        ),
        reduced_reward=torch.tensor(
            [
                0.0,
                0.2,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # Slot 0 -> slot 1 is resolved.
    #
    assert (
        controller.agent_buffer_size
        == 1
    )

    #
    # Slot 1 still needs next TTI slot 0.
    #
    assert (
        controller
        .has_unresolved_tti_boundary
    )

    assert controller.num_updates == 0


def test_next_tti_resolves_previous_last_transition():
    device = preferred_device()

    (
        controller,
        _,
        _,
        state_config,
    ) = build_controller(
        device
    )

    inputs = build_test_inputs(
        device
    )

    schedule_0 = (
        run_1lds_user_slot_loop(
            num_user_slots=2,
            inputs=inputs,
            state_config=state_config,
            action_policy=(
                controller.make_action_policy(
                    tti_index=0,
                )
            ),
            device=device,
        )
    )

    del schedule_0

    controller.finish_tti(
        reward_by_rbg=torch.zeros(
            (
                2,
                2,
            ),
            dtype=torch.float32,
            device=device,
        ),
        reduced_reward=torch.zeros(
            2,
            dtype=torch.float32,
            device=device,
        ),
    )

    assert (
        controller.agent_buffer_size
        == 1
    )

    #
    # Calling the next scheduling loop causes
    # slot-0 policy evaluation. Before sampling the
    # action, the controller sees s_(1,0), resolves
    # TTI 0's final transition, and places it in the
    # buffer.
    #
    run_1lds_user_slot_loop(
        num_user_slots=2,
        inputs=inputs,
        state_config=state_config,
        action_policy=(
            controller.make_action_policy(
                tti_index=1,
            )
        ),
        device=device,
    )

    assert (
        controller.agent_buffer_size
        == 2
    )


def snapshot_parameters(
    module: torch.nn.Module,
) -> list[
    torch.Tensor
]:
    return [
        parameter
        .detach()
        .clone()
        for parameter
        in module.parameters()
    ]


def any_parameter_changed(
    before: list[
        torch.Tensor
    ],
    module: torch.nn.Module,
) -> bool:
    return any(
        not torch.equal(
            old,
            new.detach(),
        )
        for old, new in zip(
            before,
            module.parameters(),
        )
    )


def test_controller_updates_at_safe_next_tti_boundary():
    device = preferred_device()

    torch.manual_seed(
        1234
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            1234
        )

    (
        controller,
        actor,
        critic,
        state_config,
    ) = build_controller(
        device,
        update_size=4,
    )

    inputs = build_test_inputs(
        device
    )

    #
    # TTI 0
    #
    run_1lds_user_slot_loop(
        num_user_slots=2,
        inputs=inputs,
        state_config=state_config,
        action_policy=(
            controller.make_action_policy(
                tti_index=0,
            )
        ),
        device=device,
    )

    controller.finish_tti(
        reward_by_rbg=torch.full(
            (
                2,
                2,
            ),
            0.2,
            dtype=torch.float32,
            device=device,
        ),
        reduced_reward=torch.full(
            (
                2,
            ),
            0.2,
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # TTI 1
    #
    run_1lds_user_slot_loop(
        num_user_slots=2,
        inputs=inputs,
        state_config=state_config,
        action_policy=(
            controller.make_action_policy(
                tti_index=1,
            )
        ),
        device=device,
    )

    controller.finish_tti(
        reward_by_rbg=torch.full(
            (
                2,
                2,
            ),
            -0.2,
            dtype=torch.float32,
            device=device,
        ),
        reduced_reward=torch.full(
            (
                2,
            ),
            -0.2,
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # We currently have 3 resolved transitions.
    #
    assert (
        controller.agent_buffer_size
        == 3
    )

    assert controller.num_updates == 0

    actor_before = snapshot_parameters(
        actor
    )

    critic_before = snapshot_parameters(
        critic
    )

    #
    # Starting TTI 2 exposes s_(2,0).
    #
    # That resolves transition #4, fills the
    # buffer, runs GAE + Adam, clears the buffer,
    # and only THEN samples TTI 2's first action.
    #
    run_1lds_user_slot_loop(
        num_user_slots=2,
        inputs=inputs,
        state_config=state_config,
        action_policy=(
            controller.make_action_policy(
                tti_index=2,
            )
        ),
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    assert controller.num_updates == 1

    assert (
        controller.agent_buffer_size
        == 0
    )

    assert any_parameter_changed(
        actor_before,
        actor,
    )

    assert any_parameter_changed(
        critic_before,
        critic,
    )

    updates = (
        controller.pop_update_results()
    )

    assert len(updates) == 1

    assert (
        updates[0].num_transitions
        == 4
    )

    assert torch.isfinite(
        updates[0].optimizer.actor_loss
    )

    assert torch.isfinite(
        updates[0].optimizer.critic_loss
    )


def test_controller_collection_does_not_retain_autograd_graph():
    device = preferred_device()

    (
        controller,
        _,
        _,
        state_config,
    ) = build_controller(
        device
    )

    run_1lds_user_slot_loop(
        num_user_slots=2,
        inputs=build_test_inputs(
            device
        ),
        state_config=state_config,
        action_policy=(
            controller.make_action_policy(
                tti_index=0,
            )
        ),
        device=device,
    )

    controller.finish_tti(
        reward_by_rbg=torch.zeros(
            (
                2,
                2,
            ),
            dtype=torch.float32,
            device=device,
        ),
        reduced_reward=torch.zeros(
            2,
            dtype=torch.float32,
            device=device,
        ),
    )

    assert (
        controller.agent_buffer_size
        == 1
    )


