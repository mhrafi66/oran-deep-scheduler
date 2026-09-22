import pytest
import torch

from oran_scheduler.simulator.temporal_channel import (
    TemporalChannelWindow,
    normalized_temporal_correlation,
)


def _synthetic_window() -> torch.Tensor:

    #
    # Frequency vector at time 0:
    #
    #     [1, 2]
    #
    # Time 1 is identical up to common phase.
    #
    # Time 2:
    #
    #     [2, -1]
    #
    # which is orthogonal to [1, 2].
    #
    base = torch.tensor(
        [
            1.0 + 0.0j,
            2.0 + 0.0j,
        ],
        dtype=torch.complex64,
    )

    phase = torch.exp(
        torch.tensor(
            0.4j,
            dtype=torch.complex64,
        )
    )

    orthogonal = torch.tensor(
        [
            2.0 + 0.0j,
            -1.0 + 0.0j,
        ],
        dtype=torch.complex64,
    )

    time = torch.stack(
        (
            base,
            base * phase,
            orthogonal,
        ),
        dim=0,
    )

    #
    # [batch, UE, RX, BS, TX, time, SC]
    #
    return time.reshape(
        1,
        1,
        1,
        1,
        1,
        3,
        2,
    )


def test_temporal_window_tti_slice() -> None:

    h = _synthetic_window()

    window = TemporalChannelWindow(
        h_freq=h,

        seed=123,

        tti_duration_s=0.001,
    )

    assert window.num_ttis == 3

    assert window.num_subcarriers == 2

    assert (
        window.sampling_frequency_hz
        == pytest.approx(
            1000.0
        )
    )

    second = window.tti_slice(
        1
    )

    assert second.shape == (
        1,
        1,
        1,
        1,
        1,
        1,
        2,
    )

    torch.testing.assert_close(
        second[..., 0, :],
        h[..., 1, :],
    )


def test_temporal_correlation() -> None:

    correlation = (
        normalized_temporal_correlation(
            _synthetic_window()
        )
    )

    assert correlation.shape == (
        3,
    )

    assert correlation[0].item() == pytest.approx(
        1.0,
        abs=1.0e-6,
    )

    #
    # Common phase rotation does not change
    # normalized channel direction.
    #
    assert correlation[1].item() == pytest.approx(
        1.0,
        abs=1.0e-6,
    )

    assert correlation[2].item() == pytest.approx(
        0.0,
        abs=1.0e-6,
    )


def test_temporal_window_rejects_bad_offset() -> None:

    window = TemporalChannelWindow(
        h_freq=_synthetic_window(),

        seed=1,

        tti_duration_s=0.001,
    )

    with pytest.raises(
        ValueError,
        match="tti_offset",
    ):
        window.tti_slice(
            3
        )
