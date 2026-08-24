import torch

from sionna.phy.constants import (
    BOLTZMANN_CONSTANT,
)
from sionna.phy.utils import (
    dbm_to_watt,
)

from oran_scheduler.phy.rank_diagnostic import (
    compute_ideal_svd_rank_diagnostic,
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

def print_distribution(
    name: str,
    values: torch.Tensor,
) -> None:
    values = (
        values.detach()
        .flatten()
        .float()
        .cpu()
    )

    quantile_levels = torch.tensor(
        [
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )

    quantiles = torch.quantile(
        values,
        quantile_levels,
    )

    minimum = float(values.min().item())
    maximum = float(values.max().item())
    mean = float(values.mean().item())

    p05 = float(quantiles[0].item())
    p25 = float(quantiles[1].item())
    p50 = float(quantiles[2].item())
    p75 = float(quantiles[3].item())
    p95 = float(quantiles[4].item())

    print(name)
    print(f"  min:     {minimum:.3f}")
    print(f"  p05:     {p05:.3f}")
    print(f"  p25:     {p25:.3f}")
    print(f"  median:  {p50:.3f}")
    print(f"  mean:    {mean:.3f}")
    print(f"  p75:     {p75:.3f}")
    print(f"  p95:     {p95:.3f}")
    print(f"  max:     {maximum:.3f}")


def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Real-Channel Rank-1 vs Rank-2 Diagnostic")
    print("=" * 72)

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,

        # MIMO memory-safe diagnostic only.
        num_ut_per_sector=1,

        device=device,
    )

    topology = generate_topology(
        topology_config
    )

    num_cells = topology_config.num_cells
    num_ues = topology_config.num_ues

    print()
    print(
        "Cells:                    "
        f"{num_cells}"
    )

    print(
        "UEs:                      "
        f"{num_ues}"
    )

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,
        subcarrier_spacing_hz=30e3,
        num_rbs=18,
        subcarriers_per_rb=12,
        num_ofdm_symbols=1,
        antenna_mode="paper",
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
        "MIMO H shape:             "
        f"{channel_shape}"
    )

    association = build_cell_association(
        h_freq=channel.h_freq,
        precision=channel_config.precision,
    )

    min_ues = int(
        association.num_ues_per_cell.min().item()
    )

    max_ues = int(
        association.num_ues_per_cell.max().item()
    )

    print(
        "Minimum UEs/cell:         "
        f"{min_ues}"
    )

    print(
        "Maximum UEs/cell:         "
        f"{max_ues}"
    )

    tx_power_total_w = dbm_to_watt(
        44.0,
        precision=channel_config.precision,
        device=device,
    )

    num_subcarriers = (
        channel_config.num_subcarriers
    )

    tx_power_per_subcarrier_w = (
        tx_power_total_w
        / num_subcarriers
    )

    temperature_k = 294.0

    noise_power_per_subcarrier_w = (
        BOLTZMANN_CONSTANT
        * temperature_k
        * channel_config.subcarrier_spacing_hz
    )

    tx_total_w = float(
        tx_power_total_w.item()
    )

    tx_subcarrier_w = float(
        tx_power_per_subcarrier_w.item()
    )

    noise_subcarrier_w = float(
        noise_power_per_subcarrier_w
    )

    print()
    print("=" * 72)
    print("Power Model")
    print("=" * 72)

    print(
        "Total BS transmit power:  "
        f"{tx_total_w:.6f} W"
    )

    print(
        "Power per subcarrier:     "
        f"{tx_subcarrier_w:.6e} W"
    )

    print(
        "Thermal noise/subcarrier: "
        f"{noise_subcarrier_w:.6e} W"
    )

    rank_data = (
        compute_ideal_svd_rank_diagnostic(
            h_freq=channel.h_freq,
            serving_bs=association.serving_bs,
            num_rbgs=18,
            tx_power_per_subcarrier_w=(
                tx_power_per_subcarrier_w
            ),
            noise_power_per_subcarrier_w=(
                noise_power_per_subcarrier_w
            ),
        )
    )

    best_rank = (
        rank_data.diagnostic_best_rank
    )

    num_rank1 = int(
        (best_rank == 1)
        .sum()
        .item()
    )

    num_rank2 = int(
        (best_rank == 2)
        .sum()
        .item()
    )

    total_decisions = int(
        best_rank.numel()
    )

    rank1_percent = (
        100.0
        * num_rank1
        / total_decisions
    )

    rank2_percent = (
        100.0
        * num_rank2
        / total_decisions
    )

    print()
    print("=" * 72)
    print("Ideal-SVD Rank Preference")
    print("=" * 72)

    print(
        "Rank-1 preferred:         "
        f"{num_rank1} "
        f"({rank1_percent:.2f}%)"
    )

    print(
        "Rank-2 preferred:         "
        f"{num_rank2} "
        f"({rank2_percent:.2f}%)"
    )

    rank1_se = (
        rank_data
        .rank1_rbg_spectral_efficiency
    )

    rank2_se = (
        rank_data
        .rank2_rbg_spectral_efficiency
    )

    se_gain = (
        rank2_se
        - rank1_se
    )

    print()
    print("=" * 72)
    print("Spectral Efficiency")
    print("=" * 72)

    print_distribution(
        "Rank-1 SE [bit/s/Hz]:",
        rank1_se,
    )

    print()

    print_distribution(
        "Rank-2 sum SE [bit/s/Hz]:",
        rank2_se,
    )

    print()

    print_distribution(
        "Rank-2 minus rank-1:",
        se_gain,
    )

    sinr_floor = 1.0e-12

    rank1_sinr_db = (
        10.0
        * torch.log10(
            torch.clamp(
                rank_data.rank1_sinr,
                min=sinr_floor,
            )
        )
    )

    rank2_layer1_sinr_db = (
        10.0
        * torch.log10(
            torch.clamp(
                rank_data.rank2_sinr[..., 0],
                min=sinr_floor,
            )
        )
    )

    rank2_layer2_sinr_db = (
        10.0
        * torch.log10(
            torch.clamp(
                rank_data.rank2_sinr[..., 1],
                min=sinr_floor,
            )
        )
    )

    print()
    print("=" * 72)
    print("Per-Layer SINR")
    print("=" * 72)

    print_distribution(
        "Rank-1 layer SINR [dB]:",
        rank1_sinr_db,
    )

    print()

    print_distribution(
        "Rank-2 layer 1 SINR [dB]:",
        rank2_layer1_sinr_db,
    )

    print()

    print_distribution(
        "Rank-2 layer 2 SINR [dB]:",
        rank2_layer2_sinr_db,
    )

    example_ue = 0

    example_serving_bs = int(
        association.serving_bs[
            0,
            example_ue,
        ].item()
    )

    print()
    print("=" * 72)
    print(f"Example UE {example_ue}")
    print("=" * 72)

    print(
        "Serving BS:               "
        f"{example_serving_bs}"
    )

    print()
    print(
        "RBG | Rank1 SE | Rank2 SE | Gain | Best"
    )

    for rbg_index in range(18):

        r1 = float(
            rank1_se[
                0,
                example_ue,
                rbg_index,
            ].item()
        )

        r2 = float(
            rank2_se[
                0,
                example_ue,
                rbg_index,
            ].item()
        )

        gain = float(
            se_gain[
                0,
                example_ue,
                rbg_index,
            ].item()
        )

        best = int(
            best_rank[
                0,
                example_ue,
                rbg_index,
            ].item()
        )

        print(
            f"{rbg_index:3d} | "
            f"{r1:8.3f} | "
            f"{r2:8.3f} | "
            f"{gain:7.3f} | "
            f"{best}"
        )

    rank2_mask = (
        best_rank == 2
    )

    if torch.any(rank2_mask):
        assert torch.all(
            rank2_se[rank2_mask]
            > rank1_se[rank2_mask]
        )

    rank1_mask = (
        best_rank == 1
    )

    if torch.any(rank1_mask):
        assert torch.all(
            rank1_se[rank1_mask]
            >= rank2_se[rank1_mask]
        )

    print()
    print("=" * 72)
    print(
        "Real-channel rank diagnostic PASSED"
    )
    print("=" * 72)

    print()
    print(
        "IMPORTANT: These are ideal-SVD diagnostic ranks, "
        "not final RI reports."
    )


if __name__ == "__main__":
    main()



