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
from oran_scheduler.simulator.cell_association import (
    build_cell_association,
)
from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
)
from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
)

def print_statistics(
    name: str,
    values: torch.Tensor,
    unit: str,
) -> None:
    values = (
        values
        .detach()
        .flatten()
    )

    quantile_levels = torch.tensor(
        [
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ],
        dtype=values.dtype,
        device=values.device,
    )

    quantiles = torch.quantile(
        values,
        quantile_levels,
    )

    minimum = values.min().item()
    q05 = quantiles[0].item()
    q25 = quantiles[1].item()
    median = quantiles[2].item()
    mean = values.mean().item()
    q75 = quantiles[3].item()
    q95 = quantiles[4].item()
    maximum = values.max().item()

    print(name)
    print(f"  min:       {minimum:10.3f} {unit}")
    print(f"  5th pct:   {q05:10.3f} {unit}")
    print(f"  25th pct:  {q25:10.3f} {unit}")
    print(f"  median:    {median:10.3f} {unit}")
    print(f"  mean:      {mean:10.3f} {unit}")
    print(f"  75th pct:  {q75:10.3f} {unit}")
    print(f"  95th pct:  {q95:10.3f} {unit}")
    print(f"  max:       {maximum:10.3f} {unit}")

def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Milestone 4C: Physical Per-RBG Rate")
    print("=" * 72)

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,
        num_ut_per_sector=20,
        device=device,
    )

    topology = generate_topology(
        topology_config
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

    association = build_cell_association(
        h_freq=channel.h_freq,
        precision=channel_config.precision,
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

    sinr_shape = tuple(
        sinr_data.sinr_linear.shape
    )

    mcs_shape = tuple(
        link_data.mcs_index.shape
    )

    tbler_shape = tuple(
        link_data.tbler.shape
    )

    rate_shape = tuple(
        rate_data.expected_goodput_bps.shape
    )

    print()
    print("=" * 72)
    print("Tensor Shapes")
    print("=" * 72)

    print(
        "Physical SINR:            "
        f"{sinr_shape}"
    )

    print(
        "MCS:                      "
        f"{mcs_shape}"
    )

    print(
        "TBLER:                    "
        f"{tbler_shape}"
    )

    print(
        "Expected goodput:         "
        f"{rate_shape}"
    )

    modulation_order = (
        rate_data.modulation_order
    )

    unique_qm, qm_counts = torch.unique(
        modulation_order,
        return_counts=True,
    )

    total_values = (
        modulation_order.numel()
    )

    print()
    print("=" * 72)
    print("Modulation Order Distribution")
    print("=" * 72)

    for qm, count in zip(
        unique_qm,
        qm_counts,
    ):
        qm_value = int(
            qm.item()
        )

        count_value = int(
            count.item()
        )

        percentage = (
            100.0
            * count_value
            / total_values
        )

        constellation_size = (
            2 ** qm_value
        )

        print(
            f"Qm={qm_value}: "
            f"{constellation_size}-QAM, "
            f"{count_value} "
            f"({percentage:.2f}%)"
        )

    tb_size_float = (
        rate_data.tb_size_bits.float()
    )

    print()
    print("=" * 72)
    print("Transport Block Statistics")
    print("=" * 72)
    print()

    print_statistics(
        name="Information bits per RBG/slot",
        values=tb_size_float,
        unit="bits",
    )

    nominal_mbps = (
        rate_data.nominal_rate_bps
        / 1.0e6
    )

    expected_mbps = (
        rate_data.expected_goodput_bps
        / 1.0e6
    )

    target_mbps = (
        rate_data.target_compliant_rate_bps
        / 1.0e6
    )

    print()
    print("=" * 72)
    print("Per-RBG Rate Statistics")
    print("=" * 72)
    print()

    print_statistics(
        name="Nominal PHY rate",
        values=nominal_mbps,
        unit="Mbps",
    )

    print()

    print_statistics(
        name="Expected goodput",
        values=expected_mbps,
        unit="Mbps",
    )

    print()

    print_statistics(
        name="10%-BLER target-compliant rate",
        values=target_mbps,
        unit="Mbps",
    )

    outage_mask = (
        ~rate_data.meets_bler_target
    )

    num_outage = int(
        outage_mask.sum().item()
    )

    total_links = (
        outage_mask.numel()
    )

    outage_percentage = (
        100.0
        * num_outage
        / total_links
    )

    expected_goodput_in_outage = (
        rate_data.expected_goodput_bps[
            outage_mask
        ]
    )

    print()
    print("=" * 72)
    print("Outage Resources")
    print("=" * 72)

    print(
        "Outage UE-RBG pairs:      "
        f"{num_outage} "
        f"({outage_percentage:.2f}%)"
    )

    if num_outage > 0:
        outage_goodput_mean_mbps = (
            expected_goodput_in_outage
            .mean()
            .item()
            / 1.0e6
        )

        print(
            "Mean expected goodput "
            "inside outage set:     "
            f"{outage_goodput_mean_mbps:.6f} Mbps"
        )

    example_ue = 0

    example_serving_bs = int(
        association.serving_bs[
            0,
            example_ue,
        ].item()
    )

    example_mcs = (
        link_data.mcs_index[
            0,
            example_ue,
            :,
        ]
        .detach()
        .cpu()
        .tolist()
    )

    example_tbler = (
        link_data.tbler[
            0,
            example_ue,
            :,
        ]
        .detach()
        .cpu()
        .tolist()
    )

    example_goodput = (
        expected_mbps[
            0,
            example_ue,
            :,
        ]
        .detach()
        .cpu()
        .tolist()
    )

    print()
    print("=" * 72)
    print(f"Example UE {example_ue}")
    print("=" * 72)

    print(
        "Serving BS:               "
        f"{example_serving_bs}"
    )

    print(
        "MCS over 18 RBGs:"
    )
    print(example_mcs)

    print(
        "TBLER over 18 RBGs:"
    )
    print(example_tbler)

    print(
        "Expected goodput Mbps/RBG:"
    )
    print(example_goodput)

    expected_shape = (
        topology_config.batch_size,
        topology_config.num_ues,
        18,
    )

    assert tuple(
        rate_data.nominal_rate_bps.shape
    ) == expected_shape

    assert tuple(
        rate_data.expected_goodput_bps.shape
    ) == expected_shape

    assert torch.all(
        rate_data.nominal_rate_bps >= 0.0
    )

    assert torch.all(
        rate_data.expected_goodput_bps >= 0.0
    )

    assert torch.all(
        rate_data.expected_goodput_bps
        <= rate_data.nominal_rate_bps
    )

    assert torch.all(
        rate_data.target_compliant_rate_bps[
            ~rate_data.meets_bler_target
        ]
        == 0.0
    )

    print()
    print("=" * 72)
    print("Physical per-RBG rate sanity check PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()

