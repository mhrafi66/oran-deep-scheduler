import pytest
import torch

from oran_scheduler.rl.ppo_sionna_snapshot import (
    FixedSionnaPPOSnapshotConfig,
    _gather_global_ue_values_to_serving,
)
from oran_scheduler.simulator.serving_layout import (
    ServingCellData,
)

import oran_scheduler.rl.ppo_sionna_snapshot as snapshot_module

from oran_scheduler.rl.ppo_sionna_snapshot import (
    FixedSionnaPPOSnapshotConfig,
    IndependentSionnaPPOTTIInputProvider,
    _gather_global_ue_values_to_serving,
)


def build_serving_data(
    device: torch.device,
) -> ServingCellData:
    return ServingCellData(
        power=torch.zeros(
            (
                1,
                1,
                3,
                2,
            ),
            dtype=torch.float32,
            device=device,
        ),
        global_ue_indices=torch.tensor(
            [
                [
                    [
                        2,
                        0,
                        -1,
                    ]
                ]
            ],
            dtype=torch.long,
            device=device,
        ),
        valid_ue_mask=torch.tensor(
            [
                [
                    [
                        True,
                        True,
                        False,
                    ]
                ]
            ],
            dtype=torch.bool,
            device=device,
        ),
        num_ues_per_cell=torch.tensor(
            [
                [
                    2,
                ]
            ],
            dtype=torch.long,
            device=device,
        ),
    )


def test_gather_scalar_global_values_to_serving():
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    serving_data = build_serving_data(
        device
    )

    global_values = torch.tensor(
        [
            [
                10.0,
                20.0,
                30.0,
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    result = (
        _gather_global_ue_values_to_serving(
            global_ue_values=global_values,
            serving_data=serving_data,
        )
    )

    assert tuple(
        result.shape
    ) == (
        1,
        1,
        3,
    )

    torch.testing.assert_close(
        result[
            0,
            0,
            :,
        ],
        torch.tensor(
            [
                30.0,
                10.0,
                0.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )


def test_gather_multidimensional_global_values_to_serving():
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    serving_data = build_serving_data(
        device
    )

    global_values = torch.arange(
        1 * 3 * 2 * 4,
        dtype=torch.float32,
        device=device,
    ).reshape(
        1,
        3,
        2,
        4,
    )

    result = (
        _gather_global_ue_values_to_serving(
            global_ue_values=global_values,
            serving_data=serving_data,
        )
    )

    assert tuple(
        result.shape
    ) == (
        1,
        1,
        3,
        2,
        4,
    )

    torch.testing.assert_close(
        result[
            0,
            0,
            0,
            :,
            :,
        ],
        global_values[
            0,
            2,
            :,
            :,
        ],
    )

    torch.testing.assert_close(
        result[
            0,
            0,
            1,
            :,
            :,
        ],
        global_values[
            0,
            0,
            :,
            :,
        ],
    )

    assert torch.all(
        result[
            0,
            0,
            2,
            :,
            :,
        ]
        == 0
    )


def test_snapshot_config_requires_one_rb_per_rbg():
    with pytest.raises(
        ValueError
    ):
        FixedSionnaPPOSnapshotConfig(
            num_rbs=36,
            num_rbgs=18,
        )


def test_snapshot_config_rejects_too_many_training_cells():
    with pytest.raises(
        ValueError
    ):
        FixedSionnaPPOSnapshotConfig(
            num_training_cells=22,
        )


def test_temporal_provider_refreshes_only_once_per_tti(
    monkeypatch,
):
    """
    Cell 0 and Cell 1 within one TTI must use one
    shared radio snapshot.
    """

    class FakeSnapshot:
        def __init__(
            self,
            marker: int,
        ) -> None:
            self.marker = marker

        def training_inputs(
            self,
            *,
            tti_index: int,
            stream_index: int,
        ):
            return (
                self.marker,
                tti_index,
                stream_index,
            )

    initial = FakeSnapshot(
        marker=0
    )

    refresh_calls = []

    def fake_refresh(
        *,
        reference_snapshot,
        tti_index: int,
    ):
        refresh_calls.append(
            (
                reference_snapshot.marker,
                tti_index,
            )
        )

        return FakeSnapshot(
            marker=tti_index
        )

    monkeypatch.setattr(
        snapshot_module,
        "build_refreshed_sionna_ppo_snapshot",
        fake_refresh,
    )

    provider = (
        IndependentSionnaPPOTTIInputProvider(
            initial_snapshot=initial,
        )
    )

    cell_0_tti_0 = provider(
        0,
        0,
    )

    cell_1_tti_0 = provider(
        0,
        1,
    )

    assert (
        cell_0_tti_0[0]
        == 0
    )

    assert (
        cell_1_tti_0[0]
        == 0
    )

    assert refresh_calls == []

    cell_0_tti_1 = provider(
        1,
        0,
    )

    cell_1_tti_1 = provider(
        1,
        1,
    )

    assert (
        cell_0_tti_1[0]
        == 1
    )

    assert (
        cell_1_tti_1[0]
        == 1
    )

    #
    # Exactly ONE refresh for all cells in TTI 1.
    #
    assert refresh_calls == [
        (
            0,
            1,
        )
    ]

    assert (
        provider
        .num_channel_refreshes
        == 1
    )


def test_temporal_provider_rejects_time_reversal(
    monkeypatch,
):
    class FakeSnapshot:
        def training_inputs(
            self,
            *,
            tti_index: int,
            stream_index: int,
        ):
            return (
                tti_index,
                stream_index,
            )

    def fake_refresh(
        *,
        reference_snapshot,
        tti_index: int,
    ):
        del reference_snapshot
        del tti_index

        return FakeSnapshot()

    monkeypatch.setattr(
        snapshot_module,
        "build_refreshed_sionna_ppo_snapshot",
        fake_refresh,
    )

    provider = (
        IndependentSionnaPPOTTIInputProvider(
            initial_snapshot=(
                FakeSnapshot()
            ),
        )
    )

    provider(
        1,
        0,
    )

    with pytest.raises(
        ValueError
    ):
        provider(
            0,
            0,
        )


