import pytest
import torch

from oran_scheduler.rl.ppo_sionna_chunked import (
    ChunkedSionnaPPOConfig,
    _map_candidate_global_to_local_phy,
)

from types import SimpleNamespace

import oran_scheduler.rl.ppo_sionna_chunked as chunked_module

from oran_scheduler.rl.ppo_sionna_chunked import (
    CellChunkedSionnaPPOInputProvider,
)

def preferred_device():
    return torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )


def test_chunked_training_config_uses_420_ues():
    config = ChunkedSionnaPPOConfig(
        num_ut_per_sector=20,
    )

    assert (
        21
        * config.num_ut_per_sector
        == 420
    )


def test_candidate_global_to_local_mapping():
    device = preferred_device()

    cell_global = torch.tensor(
        [
            17,
            52,
            84,
            103,
        ],
        dtype=torch.long,
        device=device,
    )

    candidate_global = torch.tensor(
        [
            84,
            17,
            -1,
            103,
        ],
        dtype=torch.long,
        device=device,
    )

    candidate_valid = torch.tensor(
        [
            True,
            True,
            False,
            True,
        ],
        dtype=torch.bool,
        device=device,
    )

    result = (
        _map_candidate_global_to_local_phy(
            candidate_global_ue_indices=(
                candidate_global
            ),
            candidate_valid_mask=(
                candidate_valid
            ),
            cell_global_ue_indices=(
                cell_global
            ),
        )
    )

    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                2,
                0,
                -1,
                3,
            ],
            dtype=torch.long,
            device=device,
        ),
    )


def test_candidate_mapping_rejects_unknown_valid_ue():
    device = preferred_device()

    with pytest.raises(
        ValueError
    ):
        _map_candidate_global_to_local_phy(
            candidate_global_ue_indices=(
                torch.tensor(
                    [
                        999,
                    ],
                    dtype=torch.long,
                    device=device,
                )
            ),

            candidate_valid_mask=(
                torch.tensor(
                    [
                        True,
                    ],
                    dtype=torch.bool,
                    device=device,
                )
            ),

            cell_global_ue_indices=(
                torch.tensor(
                    [
                        17,
                        52,
                    ],
                    dtype=torch.long,
                    device=device,
                )
            ),
        )

def test_chunked_provider_builds_cells_lazily(
    monkeypatch,
):
    build_calls = []

    context = SimpleNamespace(
        num_streams=3,
    )

    def fake_build_cell_training_inputs(
        *,
        context,
        tti_index,
        stream_index,
    ):
        del context

        build_calls.append(
            (
                tti_index,
                stream_index,
            )
        )

        return SimpleNamespace(
            observation=(
                tti_index,
                stream_index,
            ),

            physical_inputs_builder=(
                lambda prepared: prepared
            ),

            packet_arrivals=None,
        )

    monkeypatch.setattr(
        chunked_module,
        "_build_cell_training_inputs",
        fake_build_cell_training_inputs,
    )

    provider = (
        CellChunkedSionnaPPOInputProvider(
            context=context
        )
    )


    # Merely entering TTI 0 must NOT build any cell.
    provider.prepare_tti(
        0
    )

    assert (
        provider.num_tti_builds
        == 1
    )

    assert (
        provider.num_stream_builds
        == 0
    )

    assert build_calls == []


    # Request only stream 1.
    first = provider(
        0,
        1,
    )

    assert first.observation == (
        0,
        1,
    )

    assert build_calls == [
        (
            0,
            1,
        )
    ]

    assert (
        provider.num_stream_builds
        == 1
    )


    # Requesting the same stream again must reuse it.
    second = provider(
        0,
        1,
    )

    assert second is first

    assert build_calls == [
        (
            0,
            1,
        )
    ]


    # Another stream is generated independently.
    provider(
        0,
        2,
    )

    assert build_calls == [
        (
            0,
            1,
        ),
        (
            0,
            2,
        ),
    ]

    assert (
        provider.num_stream_builds
        == 2
    )


    # Entering a new TTI drops the provider-owned
    # previous-TTI input cache.
    provider(
        1,
        0,
    )

    assert (
        provider.num_tti_builds
        == 2
    )

    assert (
        provider.num_stream_builds
        == 3
    )

    assert build_calls == [
        (
            0,
            1,
        ),
        (
            0,
            2,
        ),
        (
            1,
            0,
        ),
    ]


