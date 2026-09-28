import pytest
import torch

from oran_scheduler.simulator.mobility import (
    MobilityConfig,
    MobilityEngine,
    build_mobility_snapshot,
)
from oran_scheduler.simulator.topology import (
    TopologyData,
)


def _topology() -> TopologyData:

    return TopologyData(
        ut_loc=torch.tensor(
            [
                [
                    [0.0, 0.0, 1.5],
                    [10.0, 20.0, 1.5],
                ]
            ],
            dtype=torch.float32,
        ),

        bs_loc=torch.tensor(
            [
                [
                    [0.0, 0.0, 25.0],
                ]
            ],
            dtype=torch.float32,
        ),

        ut_orientations=torch.zeros(
            (1, 2, 3),
            dtype=torch.float32,
        ),

        bs_orientations=torch.zeros(
            (1, 1, 3),
            dtype=torch.float32,
        ),

        ut_velocities=torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),

        in_state=torch.zeros(
            (1, 2),
            dtype=torch.int32,
        ),

        los=None,

        bs_virtual_loc=torch.zeros(
            (1, 1, 2, 3),
            dtype=torch.float32,
        ),

        grid=object(),
    )


def test_constant_velocity_snapshot() -> None:

    topology = _topology()

    config = MobilityConfig(
        tti_duration_s=0.5,
        trajectory_mode=(
            "constant_velocity"
        ),
    )

    snapshot = (
        build_mobility_snapshot(
            initial_topology=topology,
            config=config,
            tti_index=2,
        )
    )

    #
    # Two TTIs x 0.5 s = 1.0 s.
    #
    expected = torch.tensor(
        [
            [
                [1.0, 0.0, 1.5],
                [10.0, 22.0, 1.5],
            ]
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        snapshot.topology.ut_loc,
        expected,
    )

    torch.testing.assert_close(
        snapshot.displacement_m,
        torch.tensor(
            [[1.0, 2.0]],
            dtype=torch.float32,
        ),
    )

    assert snapshot.elapsed_time_s == pytest.approx(
        1.0
    )


def test_mobility_does_not_mutate_initial_topology() -> None:

    topology = _topology()

    original = (
        topology
        .ut_loc
        .clone()
    )

    _ = build_mobility_snapshot(
        initial_topology=topology,
        config=MobilityConfig(
            tti_duration_s=1.0,
        ),
        tti_index=3,
    )

    torch.testing.assert_close(
        topology.ut_loc,
        original,
    )


def test_static_trajectory_keeps_position() -> None:

    topology = _topology()

    snapshot = (
        build_mobility_snapshot(
            initial_topology=topology,

            config=MobilityConfig(
                tti_duration_s=0.1,
                trajectory_mode="static",
            ),

            tti_index=100,
        )
    )

    torch.testing.assert_close(
        snapshot.topology.ut_loc,
        topology.ut_loc,
    )

    assert torch.count_nonzero(
        snapshot.displacement_m
    ).item() == 0


def test_height_is_preserved() -> None:

    topology = _topology()

    #
    # Deliberately inject an invalid vertical
    # component. First mobility implementation
    # explicitly ignores it.
    #
    topology.ut_velocities[
        0,
        0,
        2,
    ] = 5.0

    snapshot = (
        build_mobility_snapshot(
            initial_topology=topology,

            config=MobilityConfig(
                tti_duration_s=1.0,
            ),

            tti_index=10,
        )
    )

    torch.testing.assert_close(
        snapshot.topology.ut_loc[..., 2],
        topology.ut_loc[..., 2],
    )

    assert (
        snapshot
        .topology
        .ut_velocities[
            0,
            0,
            2,
        ]
        .item()
        == 0.0
    )


def test_engine_is_monotonic_and_idempotent() -> None:

    engine = MobilityEngine(
        initial_topology=_topology(),

        config=MobilityConfig(
            tti_duration_s=0.25,
        ),
    )

    first = engine.advance_to(
        4
    )

    repeated = engine.advance_to(
        4
    )

    assert repeated is first

    with pytest.raises(
        ValueError,
        match="backward",
    ):
        engine.advance_to(
            3
        )


def test_displacement_guard() -> None:

    with pytest.raises(
        RuntimeError,
        match="displacement",
    ):
        build_mobility_snapshot(
            initial_topology=_topology(),

            config=MobilityConfig(
                tti_duration_s=1.0,

                max_horizontal_displacement_m=(
                    1.5
                ),
            ),

            tti_index=1,
        )
