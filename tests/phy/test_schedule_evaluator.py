import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.phy.schedule_evaluator import (
    evaluate_cell_allocation,
)
from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)


def test_evaluate_complete_cell_allocation():
    device = "cuda:0"

    h_freq = torch.zeros(
        (
            1,   # batch
            2,   # global UEs
            1,   # RX antenna
            1,   # BS
            2,   # TX antennas
            1,   # OFDM symbol
            24,  # 2 RBG x 12 SC
        ),
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0,
        0,
        0,
        0,
        0,
        0,
        :,
    ] = 1.0

    h_freq[
        0,
        1,
        0,
        0,
        1,
        0,
        :,
    ] = 1.0

    candidate_global_ues = torch.tensor(
        [0, 1],
        dtype=torch.long,
        device=device,
    )

    recommended_rank = torch.tensor(
        [
            [1, 1]
        ],
        dtype=torch.long,
        device=device,
    )


    rx_combiners = torch.ones(
        (
            1,
            2,
            2,
            2,
            1,
        ),
        dtype=torch.complex64,
        device=device,
    )


    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0, 0],
                [-1, 1],
            ],
            dtype=torch.long,
            device=device,
        )
    )


    link_config = LinkAdaptationConfig(
        num_rbgs=2,
        subcarriers_per_rbg=12,
        device=device,
    )

    rate_config = RateConfig(
        subcarriers_per_rbg=12,
        device=device,
    )


    result = evaluate_cell_allocation(
        allocation=allocation,
        candidate_global_ue_indices=(
            candidate_global_ues
        ),
        h_freq=h_freq,
        serving_cell_index=0,
        recommended_rank=(
            recommended_rank
        ),
        rx_combiners=(
            rx_combiners
        ),
        csi_subcarrier_index=6,
        subcarriers_per_rbg=12,
        tx_power_per_subcarrier_w=2.0,
        noise_power_per_subcarrier_w=1.0e-3,
        link_adaptation_config=(
            link_config
        ),
        rate_config=(
            rate_config
        ),
    )

    assert (
        result.candidate_target_compliant_rate_bps.shape
        == (
            2,
            2,
        )
    )

    assert (
        result.total_rbg_target_compliant_rate_bps.shape
        == (
            2,
        )
    )

    expected_scheduled_ues = torch.tensor(
        [1, 2],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result.num_scheduled_ues_per_rbg,
        expected_scheduled_ues,
    )


    expected_layers = torch.tensor(
        [1, 2],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result.num_physical_layers_per_rbg,
        expected_layers,
    )

    torch.testing.assert_close(
        result.candidate_target_compliant_rate_bps[
            1,
            0,
        ],
        torch.tensor(
            0.0,
            device=device,
        ),
    )


    assert (
        result.candidate_target_compliant_rate_bps[
            1,
            1,
        ]
        > 0
    )

    expected_rbg_totals = (
        result
        .candidate_target_compliant_rate_bps
        .sum(
            dim=0
        )
    )

    torch.testing.assert_close(
        result.total_rbg_target_compliant_rate_bps,
        expected_rbg_totals,
    )


    torch.testing.assert_close(
        result.total_cell_target_compliant_rate_bps,
        result
        .total_rbg_target_compliant_rate_bps
        .sum(),
    )


def test_empty_rbg_has_zero_rate():
    device = "cuda:0"

    h_freq = torch.zeros(
        (
            1,
            1,
            1,
            1,
            1,
            1,
            12,
        ),
        dtype=torch.complex64,
        device=device,
    )

    h_freq[:] = 1.0

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [-1]
            ],
            dtype=torch.long,
            device=device,
        )
    )

    candidate_global_ues = torch.tensor(
        [0],
        dtype=torch.long,
        device=device,
    )

    recommended_rank = torch.tensor(
        [
            [1]
        ],
        dtype=torch.long,
        device=device,
    )

    rx_combiners = torch.ones(
        (
            1,
            1,
            1,
            2,
            1,
        ),
        dtype=torch.complex64,
        device=device,
    )


    result = evaluate_cell_allocation(
        allocation=allocation,
        candidate_global_ue_indices=(
            candidate_global_ues
        ),
        h_freq=h_freq,
        serving_cell_index=0,
        recommended_rank=(
            recommended_rank
        ),
        rx_combiners=rx_combiners,
        csi_subcarrier_index=6,
        subcarriers_per_rbg=12,
        tx_power_per_subcarrier_w=1.0,
        noise_power_per_subcarrier_w=1.0e-3,
        link_adaptation_config=(
            LinkAdaptationConfig(
                num_rbgs=1,
                device=device,
            )
        ),
        rate_config=RateConfig(
            device=device,
        ),
    )

    assert float(
        result
        .total_cell_target_compliant_rate_bps
        .item()
    ) == 0.0

    assert int(
        result
        .num_scheduled_ues_per_rbg[
            0
        ].item()
    ) == 0

    assert int(
        result
        .num_physical_layers_per_rbg[
            0
        ].item()
    ) == 0

    