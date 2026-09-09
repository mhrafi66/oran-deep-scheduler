# import pytest
# import torch

# from oran_scheduler.rl.ppo_sionna_chunked import (
#     ChunkedSionnaPPOConfig,
#     _map_candidate_global_to_local_phy,
# )

# from types import SimpleNamespace

# import oran_scheduler.rl.ppo_sionna_chunked as chunked_module

# from oran_scheduler.rl.ppo_sionna_chunked import (
#     CellChunkedSionnaPPOInputProvider,
# )

from types import SimpleNamespace

import pytest
import torch

import oran_scheduler.rl.ppo_sionna_chunked as chunked_module


from oran_scheduler.rl.ppo_physical_score import (
    CachedPPOPhysicalRBGScorer,
)
from oran_scheduler.rl.ppo_sionna_chunked import (
    CellChunkedSionnaPPOInputProvider,
    ChunkedSionnaPPOConfig,
    _CellRadioMicrobatch,
    _combine_cell_radio_microbatches,
    _compact_candidate_phy_from_microbatches,
    _map_candidate_global_to_local_phy,
    _microbatch_channel_seed,
    _partition_cell_global_ue_indices,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIStateManager,
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


def _make_fake_radio_microbatch(
    *,
    global_ue_indices: torch.Tensor,
    serving_start_index: int,
    td_rates: torch.Tensor,
    num_rbgs: int = 3,
    num_bs: int = 21,
    num_rx: int = 2,
    num_tx: int = 4,
    num_subcarriers: int = 6,
) -> _CellRadioMicrobatch:
    """
    Pure-logic fake microbatch.

    No Sionna call occurs in these tests.
    """

    num_ues = int(
        global_ue_indices.numel()
    )

    #
    # Encode:
    #
    #     real(H) = global UE ID
    #     imag(H) = BS index
    #
    # so tests can verify both identity and
    # preservation of all BS links.
    #
    gid_values = (
        global_ue_indices
        .to(
            dtype=torch.float32
        )
        .reshape(
            1,
            num_ues,
            1,
            1,
            1,
            1,
            1,
        )
    )

    bs_values = (
        torch.arange(
            num_bs,
            dtype=torch.float32,
            device=(
                global_ue_indices.device
            ),
        )
        .reshape(
            1,
            1,
            1,
            num_bs,
            1,
            1,
            1,
        )
    )

    h_freq = (
        gid_values
        + 1j * bs_values
    ).to(
        dtype=torch.complex64
    )

    h_freq = (
        h_freq
        .expand(
            1,
            num_ues,
            num_rx,
            num_bs,
            num_tx,
            1,
            num_subcarriers,
        )
        .clone()
    )

    recommended_rank = (
        (
            global_ue_indices
            % 2
        )
        + 1
    ).reshape(
        1,
        num_ues,
    )

    rx_combiners = (
        global_ue_indices
        .to(
            dtype=torch.float32
        )
        .reshape(
            1,
            num_ues,
            1,
            1,
            1,
        )
        .expand(
            1,
            num_ues,
            num_rbgs,
            2,
            num_rx,
        )
        .to(
            dtype=torch.complex64
        )
        .clone()
    )

    precoder_directions = (
        global_ue_indices
        .to(
            dtype=torch.float32
        )
        .reshape(
            num_ues,
            1,
            1,
            1,
        )
        .expand(
            num_ues,
            num_rbgs,
            2,
            num_tx,
        )
        .to(
            dtype=torch.complex64
        )
        .clone()
    )

    wideband_cqi = (
        global_ue_indices
        .to(
            dtype=torch.float32
        )
    )

    subband_cqi = (
        wideband_cqi[
            :,
            None,
        ]
        .expand(
            num_ues,
            num_rbgs,
        )
        .clone()
    )

    return _CellRadioMicrobatch(
        serving_start_index=(
            serving_start_index
        ),

        serving_stop_index=(
            serving_start_index
            + num_ues
        ),

        global_ue_indices=(
            global_ue_indices
        ),

        td_instantaneous_rate_bps=(
            td_rates
        ),

        rank=(
            recommended_rank[
                0
            ]
        ),

        wideband_cqi=(
            wideband_cqi
        ),

        subband_cqi=(
            subband_cqi
        ),

        precoder_directions=(
            precoder_directions
        ),

        h_freq=h_freq,

        recommended_rank=(
            recommended_rank
        ),

        rx_combiners=(
            rx_combiners
        ),

        csi_subcarrier_index=0,
    )


def test_ue_microbatch_partition_covers_serving_population_once():
    global_ue_indices = torch.tensor(
        [
            103,
            151,
            317,
            400,
            401,
        ],
        dtype=torch.long,
    )

    partitions = (
        _partition_cell_global_ue_indices(
            cell_global_ue_indices=(
                global_ue_indices
            ),

            ue_microbatch_size=2,
        )
    )

    assert [
        (
            start,
            stop,
        )
        for start, stop, _ in partitions
    ] == [
        (
            0,
            2,
        ),
        (
            2,
            4,
        ),
        (
            4,
            5,
        ),
    ]

    recombined = torch.cat(
        tuple(
            values

            for _, _, values
            in partitions
        ),
        dim=0,
    )

    torch.testing.assert_close(
        recombined,
        global_ue_indices,
    )

    assert (
        torch.unique(
            recombined
        ).numel()
        == global_ue_indices.numel()
    )


def test_microbatch_recombination_preserves_full_cell_pf_tds_population():
    global_ue_indices = torch.tensor(
        [
            103,
            151,
            317,
            400,
            401,
        ],
        dtype=torch.long,
    )

    td_rates = torch.tensor(
        [
            10.0,
            100.0,
            20.0,
            80.0,
            30.0,
        ],
        dtype=torch.float32,
    )

    microbatches = (
        _make_fake_radio_microbatch(
            global_ue_indices=(
                global_ue_indices[
                    0:2
                ]
            ),

            serving_start_index=0,

            td_rates=(
                td_rates[
                    0:2
                ]
            ),
        ),

        _make_fake_radio_microbatch(
            global_ue_indices=(
                global_ue_indices[
                    2:4
                ]
            ),

            serving_start_index=2,

            td_rates=(
                td_rates[
                    2:4
                ]
            ),
        ),

        _make_fake_radio_microbatch(
            global_ue_indices=(
                global_ue_indices[
                    4:5
                ]
            ),

            serving_start_index=4,

            td_rates=(
                td_rates[
                    4:5
                ]
            ),
        ),
    )

    observation = (
        _combine_cell_radio_microbatches(
            cell_global_ue_indices=(
                global_ue_indices
            ),

            microbatches=(
                microbatches
            ),
        )
    )

    assert tuple(
        observation
        .td_instantaneous_rate_bps
        .shape
    ) == (
        5,
    )

    assert tuple(
        observation
        .subband_cqi
        .shape
    ) == (
        5,
        3,
    )

    assert tuple(
        observation
        .precoder_directions
        .shape
    ) == (
        5,
        3,
        2,
        4,
    )

    torch.testing.assert_close(
        observation
        .serving_global_ue_indices,

        global_ue_indices,
    )

    manager = (
        OneLDSCellTTIStateManager(
            initial_average_throughput_bps=(
                torch.ones(
                    5,
                    dtype=torch.float32,
                )
            ),

            serving_global_ue_indices=(
                global_ue_indices
            ),

            serving_ue_valid_mask=(
                torch.ones(
                    5,
                    dtype=torch.bool,
                )
            ),

            tds_config=(
                PFTimeDomainConfig(
                    num_candidates=2,
                )
            ),

            throughput_forgetting_factor=0.9,
        )
    )

    prepared = manager.prepare_tti(
        tti_index=0,

        observation=observation,
    )

    selected_global_ids = set(
        prepared
        .candidate_global_ue_indices[
            prepared
            .candidate_valid_mask
        ]
        .tolist()
    )

    #
    # 151 belongs to microbatch 0.
    # 400 belongs to microbatch 1.
    #
    # If TDS had been performed separately inside
    # each microbatch, this test would not be testing
    # the intended full serving population.
    #
    assert selected_global_ids == {
        151,
        400,
    }


def test_candidate_phy_compaction_preserves_all_21_bs_links_and_mapping():
    global_ue_indices = torch.tensor(
        [
            103,
            151,
            317,
            400,
            401,
        ],
        dtype=torch.long,
    )

    td_rates = torch.tensor(
        [
            10.0,
            100.0,
            20.0,
            80.0,
            30.0,
        ],
        dtype=torch.float32,
    )

    microbatches = (
        _make_fake_radio_microbatch(
            global_ue_indices=(
                global_ue_indices[
                    0:2
                ]
            ),
            serving_start_index=0,
            td_rates=td_rates[0:2],
        ),

        _make_fake_radio_microbatch(
            global_ue_indices=(
                global_ue_indices[
                    2:4
                ]
            ),
            serving_start_index=2,
            td_rates=td_rates[2:4],
        ),

        _make_fake_radio_microbatch(
            global_ue_indices=(
                global_ue_indices[
                    4:5
                ]
            ),
            serving_start_index=4,
            td_rates=td_rates[4:5],
        ),
    )

    prepared = SimpleNamespace(
        candidate_global_ue_indices=(
            torch.tensor(
                [
                    400,
                    103,
                    -1,
                    317,
                ],
                dtype=torch.long,
            )
        ),

        candidate_valid_mask=(
            torch.tensor(
                [
                    True,
                    True,
                    False,
                    True,
                ],
                dtype=torch.bool,
            )
        ),
    )

    context = SimpleNamespace(
        num_cells=21,

        config=SimpleNamespace(
            subcarriers_per_rb=2,
        ),

        tx_power_per_subcarrier_w=(
            torch.tensor(
                1.0
            )
        ),

        noise_power_per_subcarrier_w=(
            torch.tensor(
                0.1
            )
        ),

        link_adaptation_config=None,

        rate_config=None,
    )

    compact = (
        _compact_candidate_phy_from_microbatches(
            prepared=prepared,

            cell_global_ue_indices=(
                global_ue_indices
            ),

            microbatches=(
                microbatches
            ),

            real_cell_index=6,

            context=context,
        )
    )

    assert tuple(
        compact.h_freq.shape
    ) == (
        1,
        4,
        2,
        21,
        4,
        1,
        6,
    )

    torch.testing.assert_close(
        compact
        .candidate_physical_ue_indices,

        torch.tensor(
            [
                0,
                1,
                -1,
                3,
            ],
            dtype=torch.long,
        ),
    )

    #
    # Fake-H real part = global UE ID.
    #
    torch.testing.assert_close(
        compact.h_freq[
            0,
            0,
            0,
            :,
            0,
            0,
            0,
        ].real,

        torch.full(
            (
                21,
            ),
            400.0,
        ),
    )

    #
    # Fake-H imaginary part = BS index.
    #
    # Therefore this proves all 21 BS links survived
    # candidate compaction.
    #
    torch.testing.assert_close(
        compact.h_freq[
            0,
            0,
            0,
            :,
            0,
            0,
            0,
        ].imag,

        torch.arange(
            21,
            dtype=torch.float32,
        ),
    )

    #
    # Invalid candidate slot is physically zeroed.
    #
    assert (
        torch.count_nonzero(
            compact.h_freq[
                :,
                2,
            ]
        )
        == 0
    )

    #
    # Also exercise the existing PPO physical mapping
    # validation on the new compact representation.
    #
    CachedPPOPhysicalRBGScorer(
        compact
    )


def test_microbatch_channel_seed_is_deterministic():
    context = SimpleNamespace(
        num_cells=21,

        num_global_ues=420,

        config=SimpleNamespace(
            mimo_channel_seed=2000,
        ),
    )

    first = _microbatch_channel_seed(
        context=context,
        tti_index=1,
        real_cell_index=2,
        microbatch_index=3,
    )

    second = _microbatch_channel_seed(
        context=context,
        tti_index=1,
        real_cell_index=2,
        microbatch_index=3,
    )

    assert first == second

    nearby_seeds = {
        _microbatch_channel_seed(
            context=context,

            tti_index=tti_index,

            real_cell_index=cell_index,

            microbatch_index=(
                microbatch_index
            ),
        )

        for tti_index in range(
            2
        )

        for cell_index in range(
            3
        )

        for microbatch_index in range(
            4
        )
    }

    assert len(
        nearby_seeds
    ) == (
        2
        * 3
        * 4
    )


