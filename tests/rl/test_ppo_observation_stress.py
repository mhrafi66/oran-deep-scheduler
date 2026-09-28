from __future__ import annotations

import torch

from oran_scheduler.rl.ppo_observation_stress import (
    CSIStalenessConfig,
    DelayedCSIInputProvider,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
)


def _observation(
    *,
    tti_index: int,
    stream_index: int,
) -> OneLDSCellTTIObservation:

    value = float(
        100 * stream_index
        + tti_index
    )

    return OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            torch.tensor(
                [
                    10 + stream_index * 2,
                    11 + stream_index * 2,
                ],
                dtype=torch.long,
            )
        ),

        serving_ue_valid_mask=(
            torch.tensor(
                [
                    True,
                    True,
                ]
            )
        ),

        td_instantaneous_rate_bps=(
            torch.full(
                (2,),
                value,
            )
        ),

        rank=(
            torch.full(
                (2,),
                tti_index + 1,
                dtype=torch.long,
            )
        ),

        dl_buffer=(
            torch.full(
                (2,),
                1000.0 + value,
            )
        ),

        wideband_cqi=(
            torch.full(
                (2,),
                value,
            )
        ),

        subband_cqi=(
            torch.full(
                (
                    2,
                    3,
                ),
                value,
            )
        ),

        precoder_directions=(
            torch.full(
                (
                    2,
                    3,
                    2,
                    4,
                ),
                value,
            )
        ),
    )


def _base_provider(
    tti_index: int,
    stream_index: int,
) -> PPOTrainingTTIInputs:

    def physical_builder(
        prepared,
    ):
        return (
            "current_phy",
            tti_index,
            stream_index,
            prepared,
        )

    return PPOTrainingTTIInputs(
        observation=_observation(
            tti_index=tti_index,
            stream_index=stream_index,
        ),

        physical_inputs_builder=(
            physical_builder
        ),

        packet_arrivals=None,
    )


def test_zero_delay_matches_current_observation():
    provider = DelayedCSIInputProvider(
        base_provider=_base_provider,
        config=CSIStalenessConfig(
            delay_ttis=0,
        ),
    )

    result = provider(
        3,
        0,
    )

    torch.testing.assert_close(
        result.observation.wideband_cqi,
        torch.full(
            (2,),
            3.0,
        ),
    )

    torch.testing.assert_close(
        result.observation.dl_buffer,
        torch.full(
            (2,),
            1003.0,
        ),
    )


def test_two_tti_delay_uses_old_reports_but_current_phy():
    provider = DelayedCSIInputProvider(
        base_provider=_base_provider,
        config=CSIStalenessConfig(
            delay_ttis=2,
        ),
    )

    provider(
        0,
        0,
    )

    provider(
        1,
        0,
    )

    at_tti_2 = provider(
        2,
        0,
    )

    #
    # Scheduler CSI comes from TTI 0.
    #
    torch.testing.assert_close(
        at_tti_2
        .observation
        .wideband_cqi,
        torch.full(
            (2,),
            0.0,
        ),
    )

    torch.testing.assert_close(
        at_tti_2
        .observation
        .td_instantaneous_rate_bps,
        torch.full(
            (2,),
            0.0,
        ),
    )

    assert torch.equal(
        at_tti_2
        .observation
        .rank,
        torch.full(
            (2,),
            1,
            dtype=torch.long,
        ),
    )

    #
    # Current traffic remains TTI 2.
    #
    torch.testing.assert_close(
        at_tti_2
        .observation
        .dl_buffer,
        torch.full(
            (2,),
            1002.0,
        ),
    )

    #
    # Physical truth remains TTI 2.
    #
    physical_result = (
        at_tti_2
        .physical_inputs_builder(
            "prepared"
        )
    )

    assert (
        physical_result[
            0:3
        ]
        == (
            "current_phy",
            2,
            0,
        )
    )


def test_delay_advances_after_history_is_full():
    provider = DelayedCSIInputProvider(
        base_provider=_base_provider,
        config=CSIStalenessConfig(
            delay_ttis=2,
        ),
    )

    for tti_index in range(
        4
    ):
        result = provider(
            tti_index,
            0,
        )

    #
    # At TTI 3 with delay 2,
    # scheduler sees TTI 1 reports.
    #
    torch.testing.assert_close(
        result
        .observation
        .wideband_cqi,
        torch.full(
            (2,),
            1.0,
        ),
    )


def test_stream_histories_are_independent():
    provider = DelayedCSIInputProvider(
        base_provider=_base_provider,
        config=CSIStalenessConfig(
            delay_ttis=1,
        ),
    )

    provider(
        0,
        0,
    )

    provider(
        0,
        1,
    )

    stream_0 = provider(
        1,
        0,
    )

    stream_1 = provider(
        1,
        1,
    )

    torch.testing.assert_close(
        stream_0
        .observation
        .wideband_cqi,
        torch.full(
            (2,),
            0.0,
        ),
    )

    torch.testing.assert_close(
        stream_1
        .observation
        .wideband_cqi,
        torch.full(
            (2,),
            100.0,
        ),
    )


def test_same_tti_request_is_idempotent():
    calls = []

    def counting_provider(
        tti_index: int,
        stream_index: int,
    ):
        calls.append(
            (
                tti_index,
                stream_index,
            )
        )

        return _base_provider(
            tti_index,
            stream_index,
        )

    provider = DelayedCSIInputProvider(
        base_provider=(
            counting_provider
        ),
        config=CSIStalenessConfig(
            delay_ttis=2,
        ),
    )

    first = provider(
        4,
        0,
    )

    second = provider(
        4,
        0,
    )

    assert first is second

    assert calls == [
        (
            4,
            0,
        )
    ]
