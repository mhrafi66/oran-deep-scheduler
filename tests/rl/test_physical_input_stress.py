import math

import pytest
import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.rl.physical_input_stress import (
    PhysicalExecutionStressConfig,
    stress_physical_inputs,
)
from oran_scheduler.rl.ppo_physical_score import (
    PPOPhysicalScoreInputs,
)


def _inputs() -> PPOPhysicalScoreInputs:

    h_freq = torch.ones(
        (
            1,
            2,
            1,
            3,
            2,
            1,
            12,
        ),
        dtype=torch.complex64,
    )

    return PPOPhysicalScoreInputs(
        candidate_global_ue_indices=(
            torch.tensor(
                [
                    10,
                    20,
                ],
                dtype=torch.long,
            )
        ),

        candidate_physical_ue_indices=(
            torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.long,
            )
        ),

        h_freq=h_freq,

        serving_cell_index=1,

        recommended_rank=torch.ones(
            (
                1,
                2,
            ),
            dtype=torch.long,
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
        ),

        csi_subcarrier_index=0,

        subcarriers_per_rbg=12,

        tx_power_per_subcarrier_w=1.0,

        noise_power_per_subcarrier_w=1.0e-3,

        link_adaptation_config=(
            LinkAdaptationConfig(
                num_rbgs=1,
                device="cpu",
            )
        ),

        rate_config=(
            RateConfig(
                device="cpu",
            )
        ),
    )


def test_interference_power_scale_changes_only_non_serving_bs() -> None:

    inputs = _inputs()

    stressed = stress_physical_inputs(
        inputs=inputs,

        config=(
            PhysicalExecutionStressConfig(
                non_serving_interference_power_scale=4.0,
            )
        ),
    )

    #
    # Serving BS = 1 unchanged.
    #
    torch.testing.assert_close(
        stressed.h_freq[
            :,
            :,
            :,
            1,
            :,
            :,
            :,
        ],
        inputs.h_freq[
            :,
            :,
            :,
            1,
            :,
            :,
            :,
        ],
    )

    #
    # +6 dB power -> x2 channel amplitude.
    #
    torch.testing.assert_close(
        stressed.h_freq[
            :,
            :,
            :,
            0,
            :,
            :,
            :,
        ],
        2.0
        * inputs.h_freq[
            :,
            :,
            :,
            0,
            :,
            :,
            :,
        ],
    )

    torch.testing.assert_close(
        stressed.h_freq[
            :,
            :,
            :,
            2,
            :,
            :,
            :,
        ],
        2.0
        * inputs.h_freq[
            :,
            :,
            :,
            2,
            :,
            :,
            :,
        ],
    )


def test_serving_power_scale_only_changes_serving_link() -> None:

    inputs = _inputs()

    stressed = stress_physical_inputs(
        inputs=inputs,

        config=(
            PhysicalExecutionStressConfig(
                serving_signal_power_scale=0.25,
            )
        ),
    )

    #
    # quarter power -> half amplitude.
    #
    torch.testing.assert_close(
        stressed.h_freq[
            :,
            :,
            :,
            1,
            :,
            :,
            :,
        ],
        0.5
        * inputs.h_freq[
            :,
            :,
            :,
            1,
            :,
            :,
            :,
        ],
    )

    torch.testing.assert_close(
        stressed.h_freq[
            :,
            :,
            :,
            0,
            :,
            :,
            :,
        ],
        inputs.h_freq[
            :,
            :,
            :,
            0,
            :,
            :,
            :,
        ],
    )


def test_failed_bs_is_zeroed() -> None:

    inputs = _inputs()

    stressed = stress_physical_inputs(
        inputs=inputs,

        config=(
            PhysicalExecutionStressConfig(
                failed_bs_index=2,
            )
        ),
    )

    assert torch.all(
        stressed.h_freq[
            :,
            :,
            :,
            2,
            :,
            :,
            :,
        ] == 0
    )


def test_invalid_negative_scale_rejected() -> None:

    with pytest.raises(
        ValueError
    ):
        PhysicalExecutionStressConfig(
            non_serving_interference_power_scale=-1.0,
        )


def test_failed_serving_bs_uses_near_zero_outage_floor() -> None:

    inputs = _inputs()

    #
    # _inputs() uses serving_cell_index = 1.
    #
    stressed = stress_physical_inputs(
        inputs=inputs,

        config=(
            PhysicalExecutionStressConfig(
                failed_bs_index=1,
            )
        ),
    )

    serving = stressed.h_freq[
        :,
        :,
        :,
        1,
        :,
        :,
        :,
    ]

    original = inputs.h_freq[
        :,
        :,
        :,
        1,
        :,
        :,
        :,
    ]

    #
    # Default outage power = 1e-12,
    # hence channel-amplitude scale = 1e-6.
    #
    torch.testing.assert_close(
        serving,
        original * 1.0e-6,
        rtol=1.0e-5,
        atol=1.0e-12,
    )

    #
    # Critically, the desired channel must not be
    # mathematically zero.
    #
    assert torch.linalg.vector_norm(
        serving
    ).item() > 0.0
