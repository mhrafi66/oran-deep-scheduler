import torch

from oran_scheduler.phy.csi import (
    compute_rbg_spatial_modes,
    extract_serving_mimo_rbg_channel,
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
    """
    Print useful percentiles for a real-valued tensor.
    """

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

    minimum = float(
        values.min().item()
    )

    maximum = float(
        values.max().item()
    )

    mean = float(
        values.mean().item()
    )

    p05 = float(quantiles[0].item())
    p25 = float(quantiles[1].item())
    p50 = float(quantiles[2].item())
    p75 = float(quantiles[3].item())
    p95 = float(quantiles[4].item())

    print(name)
    print(f"  min:     {minimum:.6f}")
    print(f"  p05:     {p05:.6f}")
    print(f"  p25:     {p25:.6f}")
    print(f"  median:  {p50:.6f}")
    print(f"  mean:    {mean:.6f}")
    print(f"  p75:     {p75:.6f}")
    print(f"  p95:     {p95:.6f}")
    print(f"  max:     {maximum:.6f}")

def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print(
        "Milestone 5C.2: Real MIMO Spatial Modes"
    )
    print("=" * 72)
    print(f"Device: {device}")

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,
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
        "Raw MIMO H shape:         "
        f"{channel_shape}"
    )

    association = build_cell_association(
        h_freq=channel.h_freq,
        precision=channel_config.precision,
    )

    serving_bs_shape = tuple(
        association.serving_bs.shape
    )

    min_cell_ues = int(
        association.num_ues_per_cell.min().item()
    )

    max_cell_ues = int(
        association.num_ues_per_cell.max().item()
    )

    print()
    print(
        "Serving-BS map shape:     "
        f"{serving_bs_shape}"
    )

    print(
        "Minimum UEs/cell:         "
        f"{min_cell_ues}"
    )

    print(
        "Maximum UEs/cell:         "
        f"{max_cell_ues}"
    )

    serving_mimo = (
        extract_serving_mimo_rbg_channel(
            h_freq=channel.h_freq,
            serving_bs=association.serving_bs,
            num_rbgs=18,
        )
    )

    serving_shape = tuple(
        serving_mimo.h_serving_rbg.shape
    )

    print()
    print(
        "Serving MIMO RBG shape:   "
        f"{serving_shape}"
    )

    print(
        "Subcarriers per RBG:      "
        f"{serving_mimo.subcarriers_per_rbg}"
    )

    raw_num_bytes = (
        channel.h_freq.numel()
        * channel.h_freq.element_size()
    )

    serving_num_bytes = (
        serving_mimo.h_serving_rbg.numel()
        * serving_mimo.h_serving_rbg.element_size()
    )

    raw_gib = (
        raw_num_bytes
        / (1024 ** 3)
    )

    serving_gib = (
        serving_num_bytes
        / (1024 ** 3)
    )

    reduction_factor = (
        raw_num_bytes
        / serving_num_bytes
    )

    print()
    print(
        "Raw all-BS H size:        "
        f"{raw_gib:.3f} GiB"
    )

    print(
        "Serving-link H size:      "
        f"{serving_gib:.3f} GiB"
    )

    print(
        "Memory reduction factor: "
        f"{reduction_factor:.1f}x"
    )

    spatial_modes = (
        compute_rbg_spatial_modes(
            h_serving_rbg=(
                serving_mimo.h_serving_rbg
            )
        )
    )

    singular_value_shape = tuple(
        spatial_modes.singular_values.shape
    )

    mode_power_shape = tuple(
        spatial_modes.rbg_mode_power.shape
    )

    ratio_shape = tuple(
        spatial_modes
        .second_to_first_power_ratio
        .shape
    )

    print()
    print("=" * 72)
    print("Spatial-Mode Output")
    print("=" * 72)

    print(
        "Singular values:          "
        f"{singular_value_shape}"
    )

    print(
        "RBG mode power:           "
        f"{mode_power_shape}"
    )

    print(
        "P2/P1 ratio:              "
        f"{ratio_shape}"
    )

    singular_values = (
        spatial_modes.singular_values
    )

    assert torch.all(
        singular_values[..., :-1]
        >= singular_values[..., 1:]
    )

    assert torch.all(
        spatial_modes.rbg_mode_power >= 0
    )

    mode_power = (
        spatial_modes.rbg_mode_power
    )

    first_mode_power = (
        mode_power[..., 0]
    )

    epsilon = torch.finfo(
        first_mode_power.dtype
    ).tiny

    safe_first_mode_power = torch.clamp(
        first_mode_power,
        min=epsilon,
    )

    second_ratio = (
        mode_power[..., 1]
        / safe_first_mode_power
    )

    third_ratio = (
        mode_power[..., 2]
        / safe_first_mode_power
    )

    fourth_ratio = (
        mode_power[..., 3]
        / safe_first_mode_power
    )

    print()
    print("=" * 72)
    print("Relative Spatial-Mode Power")
    print("=" * 72)

    print_distribution(
        "P2 / P1:",
        second_ratio,
    )

    print()

    print_distribution(
        "P3 / P1:",
        third_ratio,
    )

    print()

    print_distribution(
        "P4 / P1:",
        fourth_ratio,
    )

    ratio_floor = 1.0e-12

    second_ratio_db = (
        10.0
        * torch.log10(
            torch.clamp(
                second_ratio,
                min=ratio_floor,
            )
        )
    )

    third_ratio_db = (
        10.0
        * torch.log10(
            torch.clamp(
                third_ratio,
                min=ratio_floor,
            )
        )
    )

    fourth_ratio_db = (
        10.0
        * torch.log10(
            torch.clamp(
                fourth_ratio,
                min=ratio_floor,
            )
        )
    )

    print()
    print("=" * 72)
    print("Relative Spatial-Mode Power in dB")
    print("=" * 72)

    print_distribution(
        "10 log10(P2 / P1):",
        second_ratio_db,
    )

    print()

    print_distribution(
        "10 log10(P3 / P1):",
        third_ratio_db,
    )

    print()

    print_distribution(
        "10 log10(P4 / P1):",
        fourth_ratio_db,
    )

    example_ue = 0

    example_serving_bs = int(
        association.serving_bs[
            0,
            example_ue,
        ].item()
    )

    example_mode_power = (
        mode_power[
            0,
            example_ue,
            :,
            :,
        ]
        .detach()
        .cpu()
    )

    example_ratio = (
        second_ratio[
            0,
            example_ue,
            :,
        ]
        .detach()
        .cpu()
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
        "RBG | P1 | P2 | P3 | P4 | P2/P1"
    )

    for rbg_index in range(18):

        p1 = float(
            example_mode_power[
                rbg_index,
                0,
            ].item()
        )

        p2 = float(
            example_mode_power[
                rbg_index,
                1,
            ].item()
        )

        p3 = float(
            example_mode_power[
                rbg_index,
                2,
            ].item()
        )

        p4 = float(
            example_mode_power[
                rbg_index,
                3,
            ].item()
        )

        ratio = float(
            example_ratio[
                rbg_index
            ].item()
        )

        print(
            f"{rbg_index:3d} | "
            f"{p1:.3e} | "
            f"{p2:.3e} | "
            f"{p3:.3e} | "
            f"{p4:.3e} | "
            f"{ratio:.4f}"
        )

    numerical_rank = torch.linalg.matrix_rank(
        serving_mimo.h_serving_rbg
    )

    rank_shape = tuple(
        numerical_rank.shape
    )

    print()
    print("=" * 72)
    print("Numerical Matrix Rank Diagnostic")
    print("=" * 72)

    print(
        "Rank tensor shape:        "
        f"{rank_shape}"
    )

    flattened_rank = (
        numerical_rank.detach()
        .flatten()
        .cpu()
    )

    for rank_value in range(1, 5):

        count = int(
            (
                flattened_rank
                == rank_value
            )
            .sum()
            .item()
        )

        total = int(
            flattened_rank.numel()
        )

        percentage = (
            100.0
            * count
            / total
        )

        print(
            f"Numerical rank {rank_value}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    assert tuple(
        mode_power.shape
    ) == (
        1,
        num_ues,
        18,
        4,
    )

    assert tuple(
        second_ratio.shape
    ) == (
        1,
        num_ues,
        18,
    )

    assert torch.all(
        second_ratio >= 0
    )

    assert torch.all(
        second_ratio <= 1.0 + 1.0e-5
    )

    print()
    print("=" * 72)
    print(
        "Real MIMO spatial-mode sanity check PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()