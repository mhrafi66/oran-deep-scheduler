import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
    compute_joint_action_log_prob,
)
from oran_scheduler.rl.ppo_agent_augmentation import (
    build_candidate_augmented_ppo_update,
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
    tti_index: int,
    user_slot_index: int,
    state: torch.Tensor,
    next_state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    reward: float,
):
    with torch.no_grad():
        (
            log_prob_by_rbg,
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

        joint_log_prob = (
            compute_joint_action_log_prob(
                log_prob_by_rbg
            )[
                0
            ]
        )

        value = critic(
            state.unsqueeze(
                0
            )
        )[
            0
        ]

    pending = build_pending_ppo_transition(
        tti_index=tti_index,
        user_slot_index=(
            user_slot_index
        ),
        state=state,
        actions=actions,
        action_mask=action_mask,
        old_log_prob_by_rbg=(
            log_prob_by_rbg[
                0
            ]
        ),
        old_joint_log_prob=(
            joint_log_prob
        ),
        old_value=value,
    )

    reward_by_rbg = torch.full(
        (
            actions.shape[0],
        ),
        fill_value=reward,
        dtype=torch.float32,
        device=state.device,
    )

    return finalize_ppo_transition(
        pending,
        reward_by_rbg=(
            reward_by_rbg
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


def build_two_transition_trajectory(
    device: torch.device,
):
    #
    # K = 3 candidates.
    #
    # State size 6 means:
    #     2 features / candidate.
    #
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

    state_0 = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ],
        device=device,
    )

    state_1 = torch.tensor(
        [
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        device=device,
    )

    state_2 = torch.tensor(
        [
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
        ],
        device=device,
    )

    mask = torch.ones(
        (
            2,
            4,
        ),
        dtype=torch.bool,
        device=device,
    )

    transition_0 = build_transition(
        actor=actor,
        critic=critic,
        tti_index=0,
        user_slot_index=0,
        state=state_0,
        next_state=state_1,
        actions=torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
        action_mask=mask,
        reward=0.1,
    )

    transition_1 = build_transition(
        actor=actor,
        critic=critic,
        tti_index=0,
        user_slot_index=1,
        state=state_1,
        next_state=state_2,
        actions=torch.tensor(
            [
                2,
                3,
            ],
            dtype=torch.long,
            device=device,
        ),
        action_mask=mask,
        reward=0.2,
    )

    return (
        actor,
        critic,
        (
            transition_0,
            transition_1,
        ),
    )



def test_augmented_ppo_recomputes_behavior_log_prob():
    device = preferred_device()

    (
        actor,
        critic,
        transitions,
    ) = build_two_transition_trajectory(
        device
    )

    prepared = (
        prepare_ppo_trajectory_with_gae(
            transitions=transitions,
            critic=critic,
            config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
        )
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        1234
    )

    result = (
        build_candidate_augmented_ppo_update(
            actor=actor,
            batch=prepared.batch,
            advantage=(
                prepared.gae.advantage
            ),
            target_return=(
                prepared
                .gae
                .target_return
            ),
            config=(
                PPOCandidateAugmentationConfig(
                    num_permutations=2,
                    include_original=True,
                )
            ),
            generator=generator,
        )
    )

    #
    # 2 real transitions
    #
    # each:
    #     original + 2 shuffled
    #
    # gives 6 optimizer rows.
    #
    assert (
        result.num_original_transitions
        == 2
    )

    assert (
        result.num_optimizer_samples
        == 6
    )

    assert tuple(
        result.batch.states.shape
    ) == (
        6,
        6,
    )

    assert tuple(
        result.batch.actions.shape
    ) == (
        6,
        2,
    )

    #
    # Re-evaluate ALL optimizer rows under the
    # unchanged behavior actor.
    #
    with torch.no_grad():
        (
            recomputed_branch_log_prob,
            _,
        ) = actor.evaluate_actions(
            state=(
                result.batch.states
            ),
            action_mask=(
                result.batch.action_masks
            ),
            actions=(
                result.batch.actions
            ),
        )

        recomputed_joint_log_prob = (
            compute_joint_action_log_prob(
                recomputed_branch_log_prob
            )
        )

    torch.testing.assert_close(
        result
        .batch
        .old_log_prob_by_rbg,
        recomputed_branch_log_prob,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    torch.testing.assert_close(
        result
        .batch
        .old_joint_log_prob,
        recomputed_joint_log_prob,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_augmented_ppo_preserves_advantage_and_return():
    device = preferred_device()

    (
        actor,
        critic,
        transitions,
    ) = build_two_transition_trajectory(
        device
    )

    prepared = (
        prepare_ppo_trajectory_with_gae(
            transitions=transitions,
            critic=critic,
            config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
        )
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        1234
    )

    result = (
        build_candidate_augmented_ppo_update(
            actor=actor,
            batch=prepared.batch,
            advantage=(
                prepared.gae.advantage
            ),
            target_return=(
                prepared
                .gae
                .target_return
            ),
            config=(
                PPOCandidateAugmentationConfig(
                    num_permutations=2,
                    include_original=True,
                )
            ),
            generator=generator,
        )
    )

    expected_advantage = (
        prepared
        .gae
        .advantage
        .repeat_interleave(
            3
        )
    )

    expected_target_return = (
        prepared
        .gae
        .target_return
        .repeat_interleave(
            3
        )
    )

    torch.testing.assert_close(
        result.advantage,
        expected_advantage,
    )

    torch.testing.assert_close(
        result.target_return,
        expected_target_return,
    )


def test_no_augmentation_preserves_original_update_data():
    device = preferred_device()

    (
        actor,
        critic,
        transitions,
    ) = build_two_transition_trajectory(
        device
    )

    prepared = (
        prepare_ppo_trajectory_with_gae(
            transitions=transitions,
            critic=critic,
            config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
        )
    )

    result = (
        build_candidate_augmented_ppo_update(
            actor=actor,
            batch=prepared.batch,
            advantage=(
                prepared.gae.advantage
            ),
            target_return=(
                prepared
                .gae
                .target_return
            ),
            config=None,
        )
    )

    assert (
        result.batch
        is prepared.batch
    )

    assert (
        result.num_original_transitions
        == 2
    )

    assert (
        result.num_optimizer_samples
        == 2
    )

    torch.testing.assert_close(
        result.advantage,
        prepared.gae.advantage,
    )


