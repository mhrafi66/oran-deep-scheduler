import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
    select_rbg_mcs,
)
from oran_scheduler.phy.rate import (
    RateConfig,
    compute_rbg_rates,
)
from oran_scheduler.phy.sinr import (
    SINRConfig,
    compute_siso_multicell_sinr,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    run_pf_tds,
)
from oran_scheduler.simulator.cell_association import (
    build_cell_association,
)
from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
)
from oran_scheduler.simulator.rbg import (
    RBGConfig,
    compute_rbg_channel_power,
)
from oran_scheduler.simulator.serving_layout import (
    build_serving_cell_data,
    gather_candidate_global_ue_indices,
    gather_candidate_rbg_values,
    gather_candidate_ue_values,
    gather_serving_ue_rbg_values,
)
from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
)

def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Milestone 4D: Physical Rate + Per-Cell PF TDS")
    print("=" * 72)
    print(f"Device: {device}")

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,
        num_ut_per_sector=20,
        device=device,
    )

    topology = generate_topology(
        topology_config
    )

    num_cells = topology_config.num_cells
    num_ues = topology_config.num_ues

    print()
    print(
        "Generated topology:       "
        f"{num_cells} cells, {num_ues} UEs"
    )

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,
        subcarrier_spacing_hz=30e3,
        num_rbs=18,
        subcarriers_per_rb=12,
        num_ofdm_symbols=1,
        device=device,
    )

    channel = generate_frequency_channel(
        topology=topology,
        topology_config=topology_config,
        channel_config=channel_config,
    )

    channel_shape = tuple(
        channel.h_freq.shape
    )

    print(
        "Channel shape:            "
        f"{channel_shape}"
    )

    print(
        "Channel device:           "
        f"{channel.h_freq.device}"
    )

    association = build_cell_association(
        h_freq=channel.h_freq,
        precision=channel_config.precision,
    )

    min_cell_ues = int(
        association.num_ues_per_cell.min().item()
    )

    max_cell_ues = int(
        association.num_ues_per_cell.max().item()
    )

    print(
        "Minimum associated UEs:   "
        f"{min_cell_ues}"
    )

    print(
        "Maximum associated UEs:   "
        f"{max_cell_ues}"
    )

    rbg_config = RBGConfig(
        num_rbgs=18,
        subcarriers_per_rb=12,
        rbs_per_rbg=1,
    )

    rbg_data = compute_rbg_channel_power(
        channel=channel,
        config=rbg_config,
    )

    rbg_shape = tuple(
        rbg_data.power.shape
    )

    print(
        "All-pair RBG power:       "
        f"{rbg_shape}"
    )

    serving_data = build_serving_cell_data(
        rbg_data=rbg_data,
        association=association,
    )

    serving_power_shape = tuple(
        serving_data.power.shape
    )

    serving_index_shape = tuple(
        serving_data.global_ue_indices.shape
    )

    valid_mask_shape = tuple(
        serving_data.valid_ue_mask.shape
    )

    print()
    print("=" * 72)
    print("Serving-Cell Layout")
    print("=" * 72)

    print(
        "Serving power:            "
        f"{serving_power_shape}"
    )

    print(
        "Global UE index map:      "
        f"{serving_index_shape}"
    )

    print(
        "Valid UE mask:            "
        f"{valid_mask_shape}"
    )

    sinr_config = SINRConfig(
        tx_power_dbm=44.0,
        subcarrier_spacing_hz=30e3,
        temperature_k=294.0,
        receiver_noise_figure_db=None,
        precision=channel_config.precision,
        device=device,
    )

    sinr_data = compute_siso_multicell_sinr(
        h_freq=channel.h_freq,
        serving_bs=association.serving_bs,
        config=sinr_config,
        bs_activity_mask=None,
    )

    link_config = LinkAdaptationConfig(
        bler_target=0.10,
        mcs_category=1,
        mcs_table_index=2,
        num_rbgs=18,
        subcarriers_per_rbg=12,
        num_slot_ofdm_symbols=14,
        precision=channel_config.precision,
        device=device,
    )

    link_data = select_rbg_mcs(
        sinr_linear=sinr_data.sinr_linear,
        config=link_config,
    )

    rate_config = RateConfig(
        mcs_category=1,
        mcs_table_index=2,
        subcarriers_per_rbg=12,
        num_slot_ofdm_symbols=14,
        num_streams_per_ue=1,
        slot_duration_s=0.5e-3,
        bler_target=0.10,
        device=device,
    )

    rate_data = compute_rbg_rates(
        mcs_index=link_data.mcs_index,
        tbler=link_data.tbler,
        config=rate_config,
    )

    global_rbg_rate = (
        rate_data.target_compliant_rate_bps
    )

    global_rate_shape = tuple(
        global_rbg_rate.shape
    )

    print()
    print("=" * 72)
    print("Physical Rate")
    print("=" * 72)

    print(
        "Global physical rate:     "
        f"{global_rate_shape}"
    )

    serving_rbg_rate = (
        gather_serving_ue_rbg_values(
            global_ue_rbg_values=(
                global_rbg_rate
            ),
            serving_data=serving_data,
        )
    )

    serving_rate_shape = tuple(
        serving_rbg_rate.shape
    )

    print(
        "Serving-cell rate:        "
        f"{serving_rate_shape}"
    )

    padded_rate_values = (
        serving_rbg_rate[
            ~serving_data.valid_ue_mask
        ]
    )

    if padded_rate_values.numel() > 0:
        assert torch.all(
            padded_rate_values == 0.0
        )

    wideband_rate = (
        serving_rbg_rate.sum(
            dim=-1
        )
    )

    wideband_shape = tuple(
        wideband_rate.shape
    )

    print(
        "Wideband rate:            "
        f"{wideband_shape}"
    )
    initial_history_bps = 1.0e6
    past_average_throughput = torch.where(
        serving_data.valid_ue_mask,
        torch.full_like(
            wideband_rate,
            fill_value=initial_history_bps,
        ),
        torch.zeros_like(
            wideband_rate
        ),
    )

    pf_config = PFTimeDomainConfig(
        num_candidates=10,
    )

    pf_result = run_pf_tds(
        instantaneous_rate=wideband_rate,
        past_average_throughput=(
            past_average_throughput
        ),
        config=pf_config,
        valid_ue_mask=(
            serving_data.valid_ue_mask
        ),
    )

    candidate_shape = tuple(
        pf_result.candidate_indices.shape
    )

    candidate_mask_shape = tuple(
        pf_result.candidate_valid_mask.shape
    )

    print()
    print("=" * 72)
    print("PF TDS")
    print("=" * 72)

    print(
        "Candidate slots:          "
        f"{candidate_shape}"
    )

    print(
        "Candidate valid mask:     "
        f"{candidate_mask_shape}"
    )

    candidate_global_ue_indices = (
        gather_candidate_global_ue_indices(
            serving_data=serving_data,
            candidate_indices=(
                pf_result.candidate_indices
            ),
            candidate_valid_mask=(
                pf_result.candidate_valid_mask
            ),
        )
    )

    candidate_rbg_rate = (
        gather_candidate_rbg_values(
            ue_rbg_values=(
                serving_rbg_rate
            ),
            candidate_indices=(
                pf_result.candidate_indices
            ),
            candidate_valid_mask=(
                pf_result.candidate_valid_mask
            ),
        )
    )

    candidate_rbg_shape = tuple(
        candidate_rbg_rate.shape
    )

    print(
        "Candidate RBG rates:      "
        f"{candidate_rbg_shape}"
    )

    candidate_wideband_rate = (
        gather_candidate_ue_values(
            ue_values=wideband_rate,
            candidate_indices=(
                pf_result.candidate_indices
            ),
            candidate_valid_mask=(
                pf_result.candidate_valid_mask
            ),
        )
    )

    candidate_history = (
        gather_candidate_ue_values(
            ue_values=(
                past_average_throughput
            ),
            candidate_indices=(
                pf_result.candidate_indices
            ),
            candidate_valid_mask=(
                pf_result.candidate_valid_mask
            ),
        )
    )

    valid_candidates_per_cell = (
        pf_result.candidate_valid_mask.sum(
            dim=-1
        )
    )

    min_candidates = int(
        valid_candidates_per_cell.min().item()
    )

    max_candidates = int(
        valid_candidates_per_cell.max().item()
    )

    mean_candidates = float(
        valid_candidates_per_cell.float().mean().item()
    )

    print()
    print(
        "Minimum valid candidates: "
        f"{min_candidates}"
    )

    print(
        "Maximum valid candidates: "
        f"{max_candidates}"
    )

    print(
        "Mean valid candidates:    "
        f"{mean_candidates:.2f}"
    )

    example_cell = 0

    associated_ues = int(
        serving_data.num_ues_per_cell[
            0,
            example_cell,
        ].item()
    )

    valid_candidate_count = int(
        pf_result.candidate_valid_mask[
            0,
            example_cell,
            :,
        ].sum().item()
    )

    example_candidate_ids = (
        candidate_global_ue_indices[
            0,
            example_cell,
            :,
        ]
        .detach()
        .cpu()
        .tolist()
    )

    example_candidate_wideband_mbps = (
        candidate_wideband_rate[
            0,
            example_cell,
            :,
        ]
        .detach()
        .cpu()
        / 1.0e6
    )

    example_candidate_wideband_mbps = (
        example_candidate_wideband_mbps.tolist()
    )

    print()
    print("=" * 72)
    print(f"Example Cell {example_cell}")
    print("=" * 72)

    print(
        "Associated UEs:           "
        f"{associated_ues}"
    )

    print(
        "Valid PF candidates:      "
        f"{valid_candidate_count}"
    )

    print(
        "Candidate global UE IDs:"
    )

    print(
        example_candidate_ids
    )

    print(
        "Candidate wideband rates "
        "(Mbps):"
    )

    print(
        example_candidate_wideband_mbps
    )

    for cell_index in range(num_cells):

        valid_mask = (
            pf_result.candidate_valid_mask[
                0,
                cell_index,
                :,
            ]
        )

        valid_candidate_rates = (
            candidate_wideband_rate[
                0,
                cell_index,
                :,
            ][valid_mask]
        )

        if valid_candidate_rates.numel() <= 1:
            continue

        assert torch.all(
            valid_candidate_rates[:-1]
            >= valid_candidate_rates[1:]
        )

    for cell_index in range(num_cells):

        valid_mask = (
            pf_result.candidate_valid_mask[
                0,
                cell_index,
                :,
            ]
        )

        global_ids = (
            candidate_global_ue_indices[
                0,
                cell_index,
                :,
            ][valid_mask]
        )

        if global_ids.numel() == 0:
            continue

        candidate_serving_cells = (
            association.serving_bs[
                0,
                global_ids,
            ]
        )

        assert torch.all(
            candidate_serving_cells
            == cell_index
        )

    max_ues_per_cell = (
        serving_data.valid_ue_mask.shape[2]
    )

    assert tuple(
        serving_rbg_rate.shape
    ) == (
        1,
        num_cells,
        max_ues_per_cell,
        18,
    )

    assert tuple(
        wideband_rate.shape
    ) == (
        1,
        num_cells,
        max_ues_per_cell,
    )

    assert tuple(
        pf_result.candidate_indices.shape
    ) == (
        1,
        num_cells,
        10,
    )

    assert tuple(
        candidate_global_ue_indices.shape
    ) == (
        1,
        num_cells,
        10,
    )

    assert tuple(
        candidate_rbg_rate.shape
    ) == (
        1,
        num_cells,
        10,
        18,
    )

    print()
    print("=" * 72)
    print(
        "Physical-rate PF TDS sanity check PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()