import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
    select_rbg_mcs,
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

def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Milestone 4B: Per-RBG Downlink MCS Selection")
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

    print(
        "Physical SINR shape:      "
        f"{tuple(sinr_data.sinr_linear.shape)}"
    )

    la_config = LinkAdaptationConfig(
        bler_target=0.10,

        # Downlink / PDSCH
        mcs_category=1,

        # PDSCH Table 2 includes 256-QAM.
        mcs_table_index=2,

        num_rbgs=18,
        subcarriers_per_rbg=12,

        # Open-reproduction assumption:
        # one channel snapshot is constant over a 14-symbol slot.
        num_slot_ofdm_symbols=14,

        precision=channel_config.precision,
        device=device,
    )

    la_data = select_rbg_mcs(
        sinr_linear=sinr_data.sinr_linear,
        config=la_config,
    )

    print(
        "MCS tensor shape:         "
        f"{tuple(la_data.mcs_index.shape)}"
    )

    mcs = la_data.mcs_index

    tbler = la_data.tbler

    meets_target = (
        la_data.meets_bler_target
    )

    effective_sinr_db = (
        10.0
        * torch.log10(
            torch.clamp(
                la_data.effective_sinr_linear,
                min=torch.finfo(
                    la_data.effective_sinr_linear.dtype
                ).tiny,
            )
        )
    )

    total_links = meets_target.numel()

    num_meeting_target = int(
        meets_target.sum().item()
    )

    num_outage = (
        total_links
        - num_meeting_target
    )

    target_percentage = (
        100.0
        * num_meeting_target
        / total_links
    )

    outage_percentage = (
        100.0
        * num_outage
        / total_links
    )

    print()
    print("=" * 72)
    print("Link-Adaptation Reliability")
    print("=" * 72)

    print(
        "BLER target:              "
        f"{la_config.bler_target:.3f}"
    )

    print(
        "UE-RBG pairs meeting target: "
        f"{num_meeting_target} "
        f"({target_percentage:.2f}%)"
    )

    print(
        "UE-RBG pairs in outage:      "
        f"{num_outage} "
        f"({outage_percentage:.2f}%)"
    )

    tbler_flat = tbler.flatten()

    tbler_quantiles = torch.quantile(
        tbler_flat,
        torch.tensor(
            [
                0.05,
                0.50,
                0.95,
            ],
            dtype=tbler_flat.dtype,
            device=tbler_flat.device,
        ),
    )

    tbler_5 = tbler_quantiles[0].item()
    tbler_median = tbler_quantiles[1].item()
    tbler_95 = tbler_quantiles[2].item()

    print()
    print(
        "TBLER 5th percentile:     "
        f"{tbler_5:.4f}"
    )

    print(
        "TBLER median:             "
        f"{tbler_median:.4f}"
    )

    print(
        "TBLER 95th percentile:    "
        f"{tbler_95:.4f}"
    )

    lowest_mask = (
        mcs
        == la_data.lowest_available_mcs_index
    )

    lowest_count = int(
        lowest_mask.sum().item()
    )

    lowest_outage_mask = (
        lowest_mask
        & (~meets_target)
    )

    lowest_outage_count = int(
        lowest_outage_mask.sum().item()
    )

    if lowest_count > 0:
        lowest_outage_percentage = (
            100.0
            * lowest_outage_count
            / lowest_count
        )
    else:
        lowest_outage_percentage = 0.0

    print()
    print(
        "Selections at lowest available MCS: "
        f"{lowest_count}"
    )

    print(
        "Lowest-MCS selections still above "
        "BLER target: "
        f"{lowest_outage_count} "
        f"({lowest_outage_percentage:.2f}%)"
    )

    effective_flat = (
        effective_sinr_db.flatten()
    )

    effective_quantiles = torch.quantile(
        effective_flat,
        torch.tensor(
            [
                0.05,
                0.50,
                0.95,
            ],
            dtype=effective_flat.dtype,
            device=effective_flat.device,
        ),
    )

    effective_5 = (
        effective_quantiles[0].item()
    )

    effective_median = (
        effective_quantiles[1].item()
    )

    effective_95 = (
        effective_quantiles[2].item()
    )

    print()
    print(
        "Effective SINR 5th pct:   "
        f"{effective_5:.3f} dB"
    )

    print(
        "Effective SINR median:    "
        f"{effective_median:.3f} dB"
    )

    print(
        "Effective SINR 95th pct:  "
        f"{effective_95:.3f} dB"
    )

    print()
    print("=" * 72)
    print("MCS Distribution")
    print("=" * 72)

    print(
        "Minimum MCS:              "
        f"{int(mcs.min().item())}"
    )

    print(
        "Maximum MCS:              "
        f"{int(mcs.max().item())}"
    )

    print(
        "Mean MCS index:           "
        f"{mcs.float().mean().item():.3f}"
    )

    unique_mcs, counts = torch.unique(
        mcs,
        return_counts=True,
    )

    total = mcs.numel()

    print()
    print("MCS index usage:")

    for index, count in zip(
        unique_mcs,
        counts,
    ):
        percentage = (
            100.0
            * count.item()
            / total
        )

        print(
            f"  MCS {int(index.item()):2d}: "
            f"{int(count.item()):6d} "
            f"({percentage:6.2f}%)"
        )

    example_ue = 0

    print()
    print("=" * 72)
    print(f"Example UE {example_ue}")
    print("=" * 72)

    example_serving_bs = int(
        association.serving_bs[
            0,
            example_ue,
        ].item()
    )

    print(
        "Serving BS:               "
        f"{example_serving_bs}"
    )
    example_serving_bs = int(
        association.serving_bs[
            0,
            example_ue,
        ].item()
    )

    print(
        "Serving BS:               "
        f"{example_serving_bs}"
    )

    example_mcs = (
        mcs[
            0,
            example_ue,
            :,
        ]
        .detach()
        .cpu()
        .tolist()
    )

    print(
        "MCS over 18 RBGs:"
    )

    print(
        example_mcs
    )

    assert tuple(
        mcs.shape
    ) == (
        1,
        topology_config.num_ues,
        18,
    )

    assert torch.all(
        mcs >= 0
    )

    print()
    print("=" * 72)
    print("Per-RBG MCS sanity check PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()