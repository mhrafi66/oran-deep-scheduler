from types import SimpleNamespace

import pytest

from oran_scheduler.rl.ppo_sionna_chunked import (
    ChunkedSionnaPPOConfig,
    _temporal_microbatch_channel_seed,
    _temporal_window_coordinates,
)


def test_default_mode_preserves_independent_radio() -> None:

    config = ChunkedSionnaPPOConfig()

    assert (
        config.temporal_radio_mode
        == "independent"
    )


def test_temporal_window_coordinates() -> None:

    assert (
        _temporal_window_coordinates(
            tti_index=0,
            window_ttis=8,
        )
        == (
            0,
            0,
            0,
        )
    )

    assert (
        _temporal_window_coordinates(
            tti_index=7,
            window_ttis=8,
        )
        == (
            0,
            0,
            7,
        )
    )

    assert (
        _temporal_window_coordinates(
            tti_index=8,
            window_ttis=8,
        )
        == (
            1,
            8,
            0,
        )
    )

    assert (
        _temporal_window_coordinates(
            tti_index=11,
            window_ttis=8,
        )
        == (
            1,
            8,
            3,
        )
    )


def test_temporal_window_coordinates_reject_bad_values() -> None:

    with pytest.raises(
        ValueError
    ):
        _temporal_window_coordinates(
            tti_index=-1,
            window_ttis=8,
        )

    with pytest.raises(
        ValueError
    ):
        _temporal_window_coordinates(
            tti_index=0,
            window_ttis=0,
        )


def test_temporal_seed_is_shared_inside_window() -> None:

    context = SimpleNamespace(
        num_cells=21,

        num_global_ues=420,

        config=SimpleNamespace(
            mimo_channel_seed=2000,
        ),
    )

    seed = (
        _temporal_microbatch_channel_seed(
            context=context,

            window_index=3,

            real_cell_index=7,

            microbatch_index=2,
        )
    )

    same = (
        _temporal_microbatch_channel_seed(
            context=context,

            window_index=3,

            real_cell_index=7,

            microbatch_index=2,
        )
    )

    assert seed == same


def test_temporal_seed_changes_between_windows() -> None:

    context = SimpleNamespace(
        num_cells=21,

        num_global_ues=420,

        config=SimpleNamespace(
            mimo_channel_seed=2000,
        ),
    )

    first = (
        _temporal_microbatch_channel_seed(
            context=context,

            window_index=0,

            real_cell_index=4,

            microbatch_index=1,
        )
    )

    second = (
        _temporal_microbatch_channel_seed(
            context=context,

            window_index=1,

            real_cell_index=4,

            microbatch_index=1,
        )
    )

    assert first != second


def test_temporal_config_validation() -> None:

    with pytest.raises(
        ValueError,
        match="temporal_radio_mode",
    ):
        ChunkedSionnaPPOConfig(
            temporal_radio_mode="fake",
        )

    with pytest.raises(
        ValueError,
        match="temporal_window_ttis",
    ):
        ChunkedSionnaPPOConfig(
            temporal_window_ttis=0,
        )

    with pytest.raises(
        ValueError,
        match="tti_duration_s",
    ):
        ChunkedSionnaPPOConfig(
            tti_duration_s=0.0,
        )
