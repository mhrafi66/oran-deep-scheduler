from types import SimpleNamespace

import pytest
import torch

import oran_scheduler.simulator.sionna_handover_measurement as measurement_module


from oran_scheduler.simulator.sionna_handover_measurement import (
    SionnaHandoverMeasurementConfig,
    SionnaPathlossHandoverMeasurementProvider,
)

from oran_scheduler.simulator.topology import (
    TopologyData,
)


def _topology() -> TopologyData:

    num_ues = 3
    num_bs = 2

    ut_loc = torch.tensor(
        [
            [
                [0.0, 0.0, 1.5],
                [1.0, 0.0, 1.5],
                [2.0, 0.0, 1.5],
            ]
        ],
        dtype=torch.float32,
    )

    bs_loc = torch.tensor(
        [
            [
                [0.0, 0.0, 25.0],
                [10.0, 0.0, 25.0],
            ]
        ],
        dtype=torch.float32,
    )

    bs_virtual_loc = (
        bs_loc[
            :,
            :,
            None,
            :,
        ]
        .expand(
            1,
            num_bs,
            num_ues,
            3,
        )
        .clone()
    )

    return TopologyData(
        ut_loc=ut_loc,

        bs_loc=bs_loc,

        ut_orientations=torch.zeros(
            (
                1,
                num_ues,
                3,
            ),
            dtype=torch.float32,
        ),

        bs_orientations=torch.zeros(
            (
                1,
                num_bs,
                3,
            ),
            dtype=torch.float32,
        ),

        ut_velocities=torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),

        in_state=torch.zeros(
            (
                1,
                num_ues,
            ),
            dtype=torch.bool,
        ),

        los=None,

        bs_virtual_loc=(
            bs_virtual_loc
        ),

        grid=object(),
    )


def _provider():

    return (
        SionnaPathlossHandoverMeasurementProvider(
            initial_topology=(
                _topology()
            ),

            #
            # The mocked channel builder does not
            # consume topology_config.
            #
            topology_config=object(),

            global_ue_indices=torch.tensor(
                [
                    2,
                    0,
                ],
                dtype=torch.long,
            ),

            config=(
                SionnaHandoverMeasurementConfig(
                    tti_duration_s=0.5,

                    max_horizontal_displacement_m=(
                        100.0
                    ),

                    device="cpu",
                )
            ),
        )
    )


def test_pathloss_is_converted_to_stronger_is_larger(
    monkeypatch,
):

    provider = _provider()

    generation_calls = []

    def fake_generate_frequency_channel(
        *,
        topology,
        topology_config,
        channel_config,
    ):

        del topology_config
        del channel_config

        generation_calls.append(
            topology
            .ut_loc
            .detach()
            .clone()
        )

        return SimpleNamespace(
            h_freq=torch.zeros(
                (
                    1,
                    2,
                    1,
                    2,
                    1,
                    1,
                    12,
                ),
                dtype=torch.complex64,
            )
        )

    def fake_pathloss(
        *,
        h_freq,
        precision,
    ):

        del h_freq
        del precision

        return torch.tensor(
            [
                [
                    [
                        2.0,
                        8.0,
                    ],
                    [
                        10.0,
                        5.0,
                    ],
                ]
            ],
            dtype=torch.float32,
        )

    monkeypatch.setattr(
        measurement_module,
        "generate_frequency_channel",
        fake_generate_frequency_channel,
    )

    monkeypatch.setattr(
        measurement_module,
        "compute_channel_averaged_pathloss",
        fake_pathloss,
    )

    result = provider(
        0
    )

    expected = torch.tensor(
        [
            [
                0.5,
                0.125,
            ],
            [
                0.1,
                0.2,
            ],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        result,
        expected,
    )

    assert len(
        generation_calls
    ) == 1

    #
    # Persistent identity order was [2, 0].
    #
    torch.testing.assert_close(
        generation_calls[
            0
        ][
            0,
            :,
            0,
        ],

        torch.tensor(
            [
                2.0,
                0.0,
            ],
            dtype=torch.float32,
        ),
    )


def test_same_tti_is_cached(
    monkeypatch,
):

    provider = _provider()

    calls = 0

    def fake_generate(
        **kwargs,
    ):

        nonlocal calls

        del kwargs

        calls += 1

        return SimpleNamespace(
            h_freq=torch.zeros(
                (
                    1,
                    2,
                    1,
                    2,
                    1,
                    1,
                    12,
                ),
                dtype=torch.complex64,
            )
        )

    def fake_pathloss(
        **kwargs,
    ):

        del kwargs

        return torch.ones(
            (
                1,
                2,
                2,
            ),
            dtype=torch.float32,
        )

    monkeypatch.setattr(
        measurement_module,
        "generate_frequency_channel",
        fake_generate,
    )

    monkeypatch.setattr(
        measurement_module,
        "compute_channel_averaged_pathloss",
        fake_pathloss,
    )

    first = provider(
        0
    )

    second = provider(
        0
    )

    assert calls == 1

    torch.testing.assert_close(
        first,
        second,
    )


def test_time_reversal_is_rejected(
    monkeypatch,
):

    provider = _provider()

    monkeypatch.setattr(
        measurement_module,
        "generate_frequency_channel",

        lambda **kwargs: SimpleNamespace(
            h_freq=torch.zeros(
                (
                    1,
                    2,
                    1,
                    2,
                    1,
                    1,
                    12,
                ),
                dtype=torch.complex64,
            )
        ),
    )

    monkeypatch.setattr(
        measurement_module,
        "compute_channel_averaged_pathloss",

        lambda **kwargs: torch.ones(
            (
                1,
                2,
                2,
            ),
            dtype=torch.float32,
        ),
    )

    provider(
        1
    )

    with pytest.raises(
        ValueError,
        match="backwards",
    ):
        provider(
            0
        )


def test_duplicate_global_ids_are_rejected():

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        SionnaPathlossHandoverMeasurementProvider(
            initial_topology=(
                _topology()
            ),

            topology_config=object(),

            global_ue_indices=torch.tensor(
                [
                    1,
                    1,
                ],
                dtype=torch.long,
            ),

            config=(
                SionnaHandoverMeasurementConfig(
                    device="cpu",
                )
            ),
        )
