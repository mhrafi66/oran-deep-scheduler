import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
    compute_joint_action_log_prob,
)


def test_actor_architecture_and_output_shape():
    config = OneLDSPPOActorConfig()

    actor = OneLDSPPOActor(
        config
    )

    assert config.state_size == 410
    assert config.hidden_size == 32
    assert config.num_rbgs == 18
    assert config.num_actions_per_rbg == 11
    assert config.output_size == 198

    linear_layers = [
        module
        for module in actor.network
        if isinstance(
            module,
            nn.Linear,
        )
    ]

    assert len(
        linear_layers
    ) == 3

    assert (
        linear_layers[0].in_features
        == 410
    )

    assert (
        linear_layers[0].out_features
        == 32
    )

    assert (
        linear_layers[1].in_features
        == 32
    )

    assert (
        linear_layers[1].out_features
        == 32
    )

    assert (
        linear_layers[2].in_features
        == 32
    )

    assert (
        linear_layers[2].out_features
        == 198
    )

    state = torch.zeros(
        (
            2,
            410,
        ),
        dtype=torch.float32,
    )

    logits = actor(
        state
    )

    assert tuple(
        logits.shape
    ) == (
        2,
        18,
        11,
    )


def test_sample_actions_returns_per_rbg_values():
    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig()
    )

    state = torch.zeros(
        (
            3,
            410,
        ),
        dtype=torch.float32,
    )

    action_mask = torch.ones(
        (
            3,
            18,
            11,
        ),
        dtype=torch.bool,
    )

    result = actor.sample_actions(
        state=state,
        action_mask=action_mask,
    )

    assert tuple(
        result.actions.shape
    ) == (
        3,
        18,
    )

    assert tuple(
        result.log_prob_by_rbg.shape
    ) == (
        3,
        18,
    )

    assert tuple(
        result.entropy_by_rbg.shape
    ) == (
        3,
        18,
    )

    assert tuple(
        result.masked_logits.shape
    ) == (
        3,
        18,
        11,
    )

    assert torch.all(
        result.actions >= 0
    )

    assert torch.all(
        result.actions < 11
    )


def test_deterministic_action_respects_mask():
    config = OneLDSPPOActorConfig(
        state_size=4,
        hidden_size=3,
        num_rbgs=2,
        num_actions_per_rbg=3,
    )

    actor = OneLDSPPOActor(
        config
    )

    # Make the network produce deterministic logits
    # controlled entirely by the final-layer bias.
    with torch.no_grad():
        for parameter in (
            actor.parameters()
        ):
            parameter.zero_()

        final_linear = (
            actor.network[-1]
        )

        final_linear.bias.copy_(
            torch.tensor(
                [
                    # RBG 0
                    1.0,
                    100.0,
                    2.0,

                    # RBG 1
                    9.0,
                    3.0,
                    1.0,
                ],
                dtype=torch.float32,
            )
        )

    state = torch.zeros(
        (
            1,
            4,
        ),
        dtype=torch.float32,
    )

    action_mask = torch.tensor(
        [
            [
                # RBG 0:
                # action 1 has the largest logit,
                # but is invalid.
                [
                    True,
                    False,
                    True,
                ],

                # RBG 1:
                [
                    True,
                    True,
                    True,
                ],
            ]
        ],
        dtype=torch.bool,
    )

    actions = actor.deterministic_actions(
        state=state,
        action_mask=action_mask,
    )

    expected = torch.tensor(
        [
            [
                2,
                0,
            ]
        ],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        actions,
        expected,
    )


def test_masked_action_has_zero_probability():
    config = OneLDSPPOActorConfig(
        state_size=4,
        hidden_size=3,
        num_rbgs=1,
        num_actions_per_rbg=3,
    )

    actor = OneLDSPPOActor(
        config
    )

    state = torch.zeros(
        (
            1,
            4,
        ),
        dtype=torch.float32,
    )

    action_mask = torch.tensor(
        [
            [
                [
                    True,
                    False,
                    True,
                ]
            ]
        ],
        dtype=torch.bool,
    )

    (
        distribution,
        _,
    ) = actor.build_distribution(
        state=state,
        action_mask=action_mask,
    )

    probabilities = (
        distribution.probs
    )

    assert float(
        probabilities[
            0,
            0,
            1,
        ].item()
    ) == 0.0

    torch.testing.assert_close(
        probabilities.sum(
            dim=-1
        ),
        torch.ones(
            (
                1,
                1,
            )
        ),
    )


def test_evaluate_actions_returns_branchwise_values():
    config = OneLDSPPOActorConfig(
        state_size=4,
        hidden_size=3,
        num_rbgs=2,
        num_actions_per_rbg=3,
    )

    actor = OneLDSPPOActor(
        config
    )

    state = torch.zeros(
        (
            1,
            4,
        ),
        dtype=torch.float32,
    )

    action_mask = torch.ones(
        (
            1,
            2,
            3,
        ),
        dtype=torch.bool,
    )

    actions = torch.tensor(
        [
            [
                0,
                2,
            ]
        ],
        dtype=torch.long,
    )

    (
        log_prob_by_rbg,
        entropy_by_rbg,
    ) = actor.evaluate_actions(
        state=state,
        action_mask=action_mask,
        actions=actions,
    )

    assert tuple(
        log_prob_by_rbg.shape
    ) == (
        1,
        2,
    )

    assert tuple(
        entropy_by_rbg.shape
    ) == (
        1,
        2,
    )

    assert torch.isfinite(
        log_prob_by_rbg
    ).all()

    assert torch.isfinite(
        entropy_by_rbg
    ).all()


def test_actor_rejects_rbg_with_no_valid_action():
    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=4,
            hidden_size=3,
            num_rbgs=1,
            num_actions_per_rbg=3,
        )
    )

    state = torch.zeros(
        (
            1,
            4,
        ),
        dtype=torch.float32,
    )

    action_mask = torch.zeros(
        (
            1,
            1,
            3,
        ),
        dtype=torch.bool,
    )

    try:
        actor.sample_actions(
            state=state,
            action_mask=action_mask,
        )

    except ValueError:
        return

    raise AssertionError(
        "An RBG with no valid action must "
        "raise ValueError."
    )



def test_joint_action_log_prob_sums_rbg_log_probs():
    log_prob_by_rbg = torch.tensor(
        [
            [
                -0.5,
                -1.0,
                -0.25,
            ],
            [
                -0.2,
                -0.3,
                -0.4,
            ],
        ],
        dtype=torch.float32,
    )

    joint_log_prob = (
        compute_joint_action_log_prob(
            log_prob_by_rbg
        )
    )

    expected = torch.tensor(
        [
            -1.75,
            -0.9,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        joint_log_prob,
        expected,
    )




