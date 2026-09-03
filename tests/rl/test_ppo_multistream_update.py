import pytest
import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
    compute_joint_action_log_prob,
)
from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.rl.ppo_critic import (
    OneLDSPPOCritic,
    OneLDSPPOCriticConfig,
)
from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_multistream_buffer import (
    PPOMultiStreamBufferConfig,
    PPOMultiStreamTransitionBuffer,
)
from oran_scheduler.rl.ppo_multistream_update import (
    PPOMultiStreamUpdateConfig,
    prepare_multistream_ppo_update_if_ready,
)
from oran_scheduler.rl.ppo_rollout import (
    build_pending_ppo_transition,
    finalize_ppo_transition,
)
from oran_scheduler.rl.ppo_tti_trajectory import (
    prepare_ppo_trajectory_with_gae,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def build_transition(
    *,
    actor: OneLDSPPOActor,
    critic: OneLDSPPOCritic,
    state: torch.Tensor,
    next_state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    reward: float,
    tti_index: int,
    user_slot_index: int,
):
    with torch.no_grad():
        (
            old_log_prob_by_rbg,
            _,
        ) = actor.evaluate_actions(
            state=state.unsqueeze(
                0
            ),
            action_mask=(
                action_mask.unsqueeze(
                    0
                )
            ),
            actions=actions.unsqueeze(
                0
            ),
        )

        old_joint_log_prob = (
            compute_joint_action_log_prob(
                old_log_prob_by_rbg
            )[
                0
            ]
        )

        old_value = critic(
            state.unsqueeze(
                0
            )
        )[
            0
        ]

    pending = build_pending_ppo_transition(
        tti_index=tti_index,
        user_slot_index=user_slot_index,
        state=state,
        actions=actions,
        action_mask=action_mask,
        old_log_prob_by_rbg=(
            old_log_prob_by_rbg[
                0
            ]
        ),
        old_joint_log_prob=(
            old_joint_log_prob
        ),
        old_value=old_value,
    )

    return finalize_ppo_transition(
        pending,
        reward_by_rbg=torch.full(
            (
                actions.shape[0],
            ),
            fill_value=reward,
            dtype=torch.float32,
            device=state.device,
        ),
        reduced_reward=torch.tensor(
            reward,
            dtype=torch.float32,
            device=state.device,
        ),
        next_state=next_state,
        terminated=torch.tensor(
            False,
            dtype=torch.bool,
            device=state.device,
        ),
    )


def build_two_stream_system(
    device: torch.device,
):
    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=6,
            hidden_size=8,
            num_rbgs=2,
            num_actions_per_rbg=4,
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=6,
            hidden_size=8,
        )
    ).to(
        device
    )

    action_mask = torch.ones(
        (
            2,
            4,
        ),
        dtype=torch.bool,
        device=device,
    )

    buffer = PPOMultiStreamTransitionBuffer(
        config=PPOMultiStreamBufferConfig(
            num_streams=2,
            update_size=4,
        )
    )

    #
    # Cell 0 states:
    #
    # 0 -> 1 -> 2
    #
    cell_0_state_0 = torch.zeros(
        6,
        device=device,
    )

    cell_0_state_1 = torch.ones(
        6,
        device=device,
    )

    cell_0_state_2 = torch.full(
        (
            6,
        ),
        2.0,
        device=device,
    )

    #
    # Cell 1 starts somewhere completely different:
    #
    # 100 -> 101 -> 102
    #
    cell_1_state_0 = torch.full(
        (
            6,
        ),
        100.0,
        device=device,
    )

    cell_1_state_1 = torch.full(
        (
            6,
        ),
        101.0,
        device=device,
    )

    cell_1_state_2 = torch.full(
        (
            6,
        ),
        102.0,
        device=device,
    )

    cell_0 = (
        build_transition(
            actor=actor,
            critic=critic,
            state=cell_0_state_0,
            next_state=cell_0_state_1,
            actions=torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            action_mask=action_mask,
            reward=0.1,
            tti_index=0,
            user_slot_index=0,
        ),
        build_transition(
            actor=actor,
            critic=critic,
            state=cell_0_state_1,
            next_state=cell_0_state_2,
            actions=torch.tensor(
                [
                    1,
                    2,
                ],
                dtype=torch.long,
                device=device,
            ),
            action_mask=action_mask,
            reward=0.2,
            tti_index=0,
            user_slot_index=1,
        ),
    )

    cell_1 = (
        build_transition(
            actor=actor,
            critic=critic,
            state=cell_1_state_0,
            next_state=cell_1_state_1,
            actions=torch.tensor(
                [
                    2,
                    0,
                ],
                dtype=torch.long,
                device=device,
            ),
            action_mask=action_mask,
            reward=1.0,
            tti_index=0,
            user_slot_index=0,
        ),
        build_transition(
            actor=actor,
            critic=critic,
            state=cell_1_state_1,
            next_state=cell_1_state_2,
            actions=torch.tensor(
                [
                    3,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            action_mask=action_mask,
            reward=2.0,
            tti_index=0,
            user_slot_index=1,
        ),
    )

    buffer.add_stream_transitions(
        stream_id=0,
        transitions=cell_0,
    )

    buffer.add_stream_transitions(
        stream_id=1,
        transitions=cell_1,
    )

    return (
        actor,
        critic,
        buffer,
        cell_0,
        cell_1,
    )


def test_multistream_gae_is_computed_independently():
    device = preferred_device()

    (
        actor,
        critic,
        buffer,
        cell_0,
        cell_1,
    ) = build_two_stream_system(
        device
    )

    gae_config = PPOGAEConfig(
        gae_lambda=0.9,
    )

    #
    # Ground truth:
    # independently prepare each trajectory.
    #
    prepared_0 = (
        prepare_ppo_trajectory_with_gae(
            transitions=cell_0,
            critic=critic,
            config=gae_config,
        )
    )

    prepared_1 = (
        prepare_ppo_trajectory_with_gae(
            transitions=cell_1,
            critic=critic,
            config=gae_config,
        )
    )

    centralized = (
        prepare_multistream_ppo_update_if_ready(
            buffer=buffer,
            actor=actor,
            critic=critic,
            gae_config=gae_config,
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode="require_exact",
                )
            ),
        )
    )

    assert centralized is not None

    expected_advantage = torch.cat(
        (
            prepared_0.gae.advantage,
            prepared_1.gae.advantage,
        ),
        dim=0,
    )

    expected_target_return = torch.cat(
        (
            prepared_0.gae.target_return,
            prepared_1.gae.target_return,
        ),
        dim=0,
    )

    torch.testing.assert_close(
        centralized.real_advantage,
        expected_advantage,
    )

    torch.testing.assert_close(
        centralized.real_target_return,
        expected_target_return,
    )

    assert (
        centralized.num_transitions_by_stream
        == (
            2,
            2,
        )
    )


def test_multistream_batch_concatenates_streams_after_gae():
    device = preferred_device()

    (
        actor,
        critic,
        buffer,
        _,
        _,
    ) = build_two_stream_system(
        device
    )

    result = (
        prepare_multistream_ppo_update_if_ready(
            buffer=buffer,
            actor=actor,
            critic=critic,
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode="require_exact",
                )
            ),
        )
    )

    assert result is not None

    assert (
        result.num_real_transitions
        == 4
    )

    assert tuple(
        result.real_batch.states.shape
    ) == (
        4,
        6,
    )

    #
    # Stream-major deterministic concatenation:
    #
    torch.testing.assert_close(
        result.real_batch.states[
            0
        ],
        torch.zeros(
            6,
            device=device,
        ),
    )

    torch.testing.assert_close(
        result.real_batch.states[
            2
        ],
        torch.full(
            (
                6,
            ),
            100.0,
            device=device,
        ),
    )


def test_multistream_update_waits_below_threshold():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=6,
            hidden_size=8,
            num_rbgs=2,
            num_actions_per_rbg=4,
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=6,
            hidden_size=8,
        )
    ).to(
        device
    )

    buffer = PPOMultiStreamTransitionBuffer(
        config=PPOMultiStreamBufferConfig(
            num_streams=2,
            update_size=4,
        )
    )

    result = (
        prepare_multistream_ppo_update_if_ready(
            buffer=buffer,
            actor=actor,
            critic=critic,
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode="require_exact",
                )
            ),
        )
    )

    assert result is None


def test_require_exact_rejects_centralized_overflow():
    device = preferred_device()

    (
        actor,
        critic,
        buffer,
        _,
        _,
    ) = build_two_stream_system(
        device
    )

    #
    # Buffer currently has 4.
    #
    # Add one new independent stream-0 transition
    # continuing state 2 -> 3.
    #
    action_mask = torch.ones(
        (
            2,
            4,
        ),
        dtype=torch.bool,
        device=device,
    )

    extra = build_transition(
        actor=actor,
        critic=critic,
        state=torch.full(
            (
                6,
            ),
            2.0,
            device=device,
        ),
        next_state=torch.full(
            (
                6,
            ),
            3.0,
            device=device,
        ),
        actions=torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
        action_mask=action_mask,
        reward=0.3,
        tti_index=1,
        user_slot_index=0,
    )

    buffer.add_transition(
        stream_id=0,
        transition=extra,
    )

    assert len(
        buffer
    ) == 5

    with pytest.raises(
        RuntimeError,
        match="crossed",
    ):
        prepare_multistream_ppo_update_if_ready(
            buffer=buffer,
            actor=actor,
            critic=critic,
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode="require_exact",
                )
            ),
        )


def test_use_all_when_reached_consumes_all_current_rows():
    device = preferred_device()

    (
        actor,
        critic,
        buffer,
        _,
        _,
    ) = build_two_stream_system(
        device
    )

    action_mask = torch.ones(
        (
            2,
            4,
        ),
        dtype=torch.bool,
        device=device,
    )

    buffer.add_transition(
        stream_id=0,
        transition=build_transition(
            actor=actor,
            critic=critic,
            state=torch.full(
                (
                    6,
                ),
                2.0,
                device=device,
            ),
            next_state=torch.full(
                (
                    6,
                ),
                3.0,
                device=device,
            ),
            actions=torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            action_mask=action_mask,
            reward=0.3,
            tti_index=1,
            user_slot_index=0,
        ),
    )

    #
    # M=4, available=5.
    #
    result = (
        prepare_multistream_ppo_update_if_ready(
            buffer=buffer,
            actor=actor,
            critic=critic,
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode=(
                        "use_all_when_reached"
                    ),
                )
            ),
        )
    )

    assert result is not None

    assert (
        result.num_real_transitions
        == 5
    )

    assert (
        result.num_optimizer_samples
        == 5
    )

    assert (
        result.num_transitions_by_stream
        == (
            3,
            2,
        )
    )


def test_multistream_candidate_augmentation_occurs_after_gae():
    device = preferred_device()

    (
        actor,
        critic,
        buffer,
        _,
        _,
    ) = build_two_stream_system(
        device
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        1234
    )

    result = (
        prepare_multistream_ppo_update_if_ready(
            buffer=buffer,
            actor=actor,
            critic=critic,
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode="require_exact",
                )
            ),
            candidate_augmentation_config=(
                PPOCandidateAugmentationConfig(
                    num_permutations=1,
                    include_original=True,
                )
            ),
            candidate_augmentation_generator=(
                generator
            ),
        )
    )

    assert result is not None

    #
    # 4 real temporal samples.
    #
    assert (
        result.num_real_transitions
        == 4
    )

    #
    # original + one permutation per sample.
    #
    assert (
        result.num_optimizer_samples
        == 8
    )

    #
    # GAE labels remain the four REAL trajectory
    # labels.
    #
    assert tuple(
        result.real_advantage.shape
    ) == (
        4,
    )

    #
    # Optimizer sees augmented copies.
    #
    assert tuple(
        result
        .optimization
        .advantage
        .shape
    ) == (
        8,
    )


def test_multistream_augmented_old_probabilities_match_behavior_actor():
    device = preferred_device()

    (
        actor,
        critic,
        buffer,
        _,
        _,
    ) = build_two_stream_system(
        device
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        1234
    )

    result = (
        prepare_multistream_ppo_update_if_ready(
            buffer=buffer,
            actor=actor,
            critic=critic,
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode="require_exact",
                )
            ),
            candidate_augmentation_config=(
                PPOCandidateAugmentationConfig(
                    num_permutations=1,
                    include_original=True,
                )
            ),
            candidate_augmentation_generator=(
                generator
            ),
        )
    )

    assert result is not None

    batch = (
        result
        .optimization
        .batch
    )

    with torch.no_grad():
        (
            log_prob_by_rbg,
            _,
        ) = actor.evaluate_actions(
            state=batch.states,
            action_mask=(
                batch.action_masks
            ),
            actions=batch.actions,
        )

        joint_log_prob = (
            compute_joint_action_log_prob(
                log_prob_by_rbg
            )
        )

    torch.testing.assert_close(
        batch.old_log_prob_by_rbg,
        log_prob_by_rbg,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    torch.testing.assert_close(
        batch.old_joint_log_prob,
        joint_log_prob,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_paper_shaped_joint_multicell_boundary_requires_interpretation():
    #
    # Current joint-layer interpretation:
    #
    num_cells = 21
    num_user_slots = 4

    transitions_per_tti = (
        num_cells
        * num_user_slots
    )

    update_size = 128

    assert transitions_per_tti == 84

    assert (
        transitions_per_tti
        < update_size
    )

    assert (
        2 * transitions_per_tti
        == 168
    )

    assert (
        2 * transitions_per_tti
        > update_size
    )


