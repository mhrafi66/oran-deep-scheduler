import torch
from torch import nn

from oran_scheduler.rl.ppo_critic import (
    OneLDSPPOCritic,
    OneLDSPPOCriticConfig,
)


def test_critic_architecture():
    config = OneLDSPPOCriticConfig()

    critic = OneLDSPPOCritic(
        config
    )

    assert config.state_size == 410
    assert config.hidden_size == 32

    linear_layers = [
        module
        for module in critic.network
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
        == 1
    )


def test_critic_returns_one_value_per_state():
    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig()
    )

    state = torch.zeros(
        (
            4,
            410,
        ),
        dtype=torch.float32,
    )

    value = critic(
        state
    )

    assert tuple(
        value.shape
    ) == (
        4,
    )

    assert torch.isfinite(
        value
    ).all()


def test_critic_can_produce_known_scalar_value():
    config = OneLDSPPOCriticConfig(
        state_size=4,
        hidden_size=3,
    )

    critic = OneLDSPPOCritic(
        config
    )

    with torch.no_grad():
        for parameter in (
            critic.parameters()
        ):
            parameter.zero_()

        final_linear = (
            critic.network[-1]
        )

        final_linear.bias.fill_(
            0.75
        )

    state = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            [
                5.0,
                6.0,
                7.0,
                8.0,
            ],
        ],
        dtype=torch.float32,
    )

    value = critic(
        state
    )

    expected = torch.tensor(
        [
            0.75,
            0.75,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        value,
        expected,
    )


def test_critic_supports_backpropagation():
    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=4,
            hidden_size=3,
        )
    )

    state = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            [
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        ],
        dtype=torch.float32,
    )

    predicted_value = critic(
        state
    )

    target_return = torch.tensor(
        [
            1.0,
            -0.5,
        ],
        dtype=torch.float32,
    )

    value_loss = (
        0.5
        * torch.mean(
            (
                predicted_value
                - target_return
            )
            ** 2
        )
    )

    value_loss.backward()

    gradients = [
        parameter.grad
        for parameter
        in critic.parameters()
        if parameter.requires_grad
    ]

    assert all(
        gradient is not None
        for gradient in gradients
    )

    assert any(
        torch.any(
            gradient != 0.0
        )
        for gradient in gradients
    )

def test_critic_rejects_wrong_state_size():
    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig()
    )

    invalid_state = torch.zeros(
        (
            1,
            409,
        ),
        dtype=torch.float32,
    )

    try:
        critic(
            invalid_state
        )

    except ValueError:
        return

    raise AssertionError(
        "Wrong state size must raise ValueError."
    )


