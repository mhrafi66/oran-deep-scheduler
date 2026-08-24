import torch

from sionna.phy.constants import (
    BOLTZMANN_CONSTANT,
)
from sionna.phy.utils import (
    dbm_to_watt,
)

from oran_scheduler.phy.csi import (
    compute_ideal_svd_csi,
    extract_serving_mimo_rbg_channel,
)
from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.mu_mimo_rate import (
    compute_mu_mimo_rbg_rates,
)
from oran_scheduler.phy.mu_mimo_rbg import (
    compute_isotropic_inter_cell_covariance,
    evaluate_selected_users_on_rbg,
)
from oran_scheduler.phy.rank_diagnostic import (
    compute_ideal_svd_rank_diagnostic,
)
from oran_scheduler.phy.rate import (
    RateConfig,
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


def estimate_rzf_alpha(
    selected_ranks: torch.Tensor,
    selected_rx_combiners: torch.Tensor,
    inter_cell_covariance: torch.Tensor,
    total_tx_power_w: torch.Tensor,
    noise_power_w: torch.Tensor,
) -> float:
    """
    Estimate one RZF regularization value for the RBG.

    Open-reproduction heuristic:

        alpha
            =
        L * mean(interference + noise) / transmit_power

    where L is the number of active spatial layers.

    This is NOT a Bell Labs paper-specified value.
    """

    possible_layers = torch.arange(
        2,
        dtype=torch.long,
        device=selected_ranks.device,
    )

    valid_layer_mask = (
        possible_layers.unsqueeze(0)
        < selected_ranks.unsqueeze(1)
    )

    layer_pairs = torch.nonzero(
        valid_layer_mask,
        as_tuple=False,
    )

    layer_ue_indices = (
        layer_pairs[:, 0]
    )

    layer_index_within_ue = (
        layer_pairs[:, 1]
    )

    layer_combiners = (
        selected_rx_combiners[
            layer_ue_indices,
            layer_index_within_ue,
            :,
        ]
    )

    combiner_norm = torch.linalg.vector_norm(
        layer_combiners,
        dim=-1,
        keepdim=True,
    )

    layer_combiners = (
        layer_combiners
        / combiner_norm
    )

    layer_covariance = (
        inter_cell_covariance[
            layer_ue_indices,
            :,
            :,
            :,
            :,
        ]
    )

    projected_interference = (
        torch.einsum(
            "lr,lscrq,lq->lsc",
            layer_combiners.conj(),
            layer_covariance,
            layer_combiners,
        )
        .real
    )

    projected_interference = torch.clamp(
        projected_interference,
        min=0.0,
    )

    mean_interference = (
        projected_interference.mean()
    )

    tx_power = torch.as_tensor(
        total_tx_power_w,
        dtype=projected_interference.dtype,
        device=projected_interference.device,
    )

    noise_power = torch.as_tensor(
        noise_power_w,
        dtype=projected_interference.dtype,
        device=projected_interference.device,
    )

    num_layers = int(
        selected_ranks.sum().item()
    )

    effective_disturbance = (
        mean_interference
        + noise_power
    )

    alpha = (
        float(num_layers)
        * effective_disturbance
        / tx_power
    )

    return float(
        alpha.item()
    )


def evaluate_user_set(
    h_freq: torch.Tensor,
    csi_data,
    selected_global_ue_indices: torch.Tensor,
    serving_cell_index: int,
    rbg_index: int,
    subcarriers_per_rbg: int,
    tx_power_per_subcarrier_w: torch.Tensor,
    noise_power_per_subcarrier_w: torch.Tensor,
    link_config: LinkAdaptationConfig,
    rate_config: RateConfig,
):
    """
    Evaluate one selected UE set on one real RBG.
    """

    selected_ranks = (
        csi_data.recommended_rank[
            0,
            selected_global_ue_indices,
        ]
    )

    selected_rx_combiners = (
        csi_data.rx_combiners[
            0,
            selected_global_ue_indices,
            rbg_index,
            :,
            :,
        ]
    )

    first_subcarrier = (
        rbg_index
        * subcarriers_per_rbg
    )

    last_subcarrier = (
        first_subcarrier
        + subcarriers_per_rbg
    )

    selected_all_bs_channel = (
        h_freq[
            0,
            selected_global_ue_indices,
            :,
            :,
            :,
            :,
            first_subcarrier:last_subcarrier,
        ]
    )

    selected_all_bs_channel = (
        selected_all_bs_channel.permute(
            0,
            4,
            5,
            2,
            1,
            3,
        )
        .contiguous()
    )

    inter_cell_covariance = (
        compute_isotropic_inter_cell_covariance(
            all_bs_channel=(
                selected_all_bs_channel
            ),
            serving_cell_index=(
                serving_cell_index
            ),
            tx_power_per_subcarrier_w=(
                tx_power_per_subcarrier_w
            ),
        )
    )

    selected_serving_channel = (
        selected_all_bs_channel[
            :,
            :,
            :,
            serving_cell_index,
            :,
            :,
        ]
    )

    rzf_alpha = estimate_rzf_alpha(
        selected_ranks=selected_ranks,
        selected_rx_combiners=(
            selected_rx_combiners
        ),
        inter_cell_covariance=(
            inter_cell_covariance
        ),
        total_tx_power_w=(
            tx_power_per_subcarrier_w
        ),
        noise_power_w=(
            noise_power_per_subcarrier_w
        ),
    )

    rbg_data = evaluate_selected_users_on_rbg(
        selected_ue_channel=(
            selected_serving_channel
        ),
        selected_ranks=selected_ranks,
        selected_rx_combiners=(
            selected_rx_combiners
        ),
        total_tx_power_w=(
            tx_power_per_subcarrier_w
        ),
        noise_power_w=(
            noise_power_per_subcarrier_w
        ),
        rzf_alpha=rzf_alpha,
        csi_subcarrier_index=(
            csi_data.csi_subcarrier_index
        ),
        inter_cell_covariance=(
            inter_cell_covariance
        ),
    )

    rate_data = compute_mu_mimo_rbg_rates(
        layer_sinr_linear=(
            rbg_data.sinr_data.sinr_linear
        ),
        layer_ue_indices=(
            rbg_data.layer_ue_indices
        ),
        layer_index_within_ue=(
            rbg_data.layer_index_within_ue
        ),
        selected_ranks=selected_ranks,
        link_adaptation_config=(
            link_config
        ),
        rate_config=rate_config,
    )

    return (
        selected_ranks,
        rzf_alpha,
        rbg_data,
        rate_data,
    )



def print_user_set_result(
    title: str,
    selected_global_ue_indices: torch.Tensor,
    selected_ranks: torch.Tensor,
    rzf_alpha: float,
    rbg_data,
    rate_data,
) -> None:

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    num_ues = int(
        selected_global_ue_indices.numel()
    )

    num_layers = int(
        selected_ranks.sum().item()
    )

    print(
        "Selected UEs:             "
        f"{num_ues}"
    )

    print(
        "Total spatial layers:     "
        f"{num_layers}"
    )

    print(
        "RZF alpha:                "
        f"{rzf_alpha:.6e}"
    )

    for ue_position in range(num_ues):

        global_ue = int(
            selected_global_ue_indices[
                ue_position
            ].item()
        )

        rank = int(
            selected_ranks[
                ue_position
            ].item()
        )

        mcs = int(
            rate_data.link_adaptation.mcs_index[
                ue_position
            ].item()
        )

        tbler = float(
            rate_data.link_adaptation.tbler[
                ue_position
            ].item()
        )

        nominal_mbps = float(
            rate_data.ue_rate_bps[
                ue_position
            ].item()
            / 1.0e6
        )

        expected_mbps = float(
            rate_data.expected_ue_goodput_bps[
                ue_position
            ].item()
            / 1.0e6
        )

        compliant_mbps = float(
            rate_data.target_compliant_ue_rate_bps[
                ue_position
            ].item()
            / 1.0e6
        )

        print()

        print(
            f"UE {global_ue}:"
        )

        print(
            "  rank:                   "
            f"{rank}"
        )

        print(
            "  MCS:                    "
            f"{mcs}"
        )

        print(
            "  TBLER:                  "
            f"{tbler:.6f}"
        )

        print(
            "  nominal rate:           "
            f"{nominal_mbps:.3f} Mbps"
        )

        print(
            "  expected goodput:       "
            f"{expected_mbps:.3f} Mbps"
        )

        print(
            "  target-compliant rate:  "
            f"{compliant_mbps:.3f} Mbps"
        )

    layer_sinr = (
        rbg_data.sinr_data.sinr_linear
    )

    sinr_floor = 1.0e-12

    layer_sinr_db = (
        10.0
        * torch.log10(
            torch.clamp(
                layer_sinr,
                min=sinr_floor,
            )
        )
    )

    print()
    print("Layer SINRs:")

    for layer_index in range(num_layers):

        selected_ue_position = int(
            rbg_data.layer_ue_indices[
                layer_index
            ].item()
        )

        layer_within_ue = int(
            rbg_data.layer_index_within_ue[
                layer_index
            ].item()
        )

        global_ue = int(
            selected_global_ue_indices[
                selected_ue_position
            ].item()
        )

        median_sinr_db = float(
            layer_sinr_db[
                ...,
                layer_index,
            ]
            .median()
            .item()
        )

        human_layer_number = (
            layer_within_ue + 1
        )

        print(
            f"  UE {global_ue}, "
            f"layer {human_layer_number}: "
            f"{median_sinr_db:.3f} dB"
        )

    total_rate_mbps = float(
        rate_data
        .total_target_compliant_rate_bps
        .item()
        / 1.0e6
    )

    print()

    print(
        "TOTAL target-compliant:   "
        f"{total_rate_mbps:.3f} Mbps"
    )

def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Real MU-MIMO RBG Rate Sanity")
    print("=" * 72)

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,
        num_ut_per_sector=1,
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
        "Channel shape:            "
        f"{channel_shape}"
    )

    association = build_cell_association(
        h_freq=channel.h_freq,
        precision=channel_config.precision,
    )

    cell_index = int(
        torch.argmax(
            association.num_ues_per_cell[
                0
            ]
        ).item()
    )

    cell_ue_indices = torch.nonzero(
        association.serving_bs[
            0
        ]
        == cell_index,
        as_tuple=False,
    ).flatten()

    num_cell_ues = int(
        cell_ue_indices.numel()
    )

    print(
        "Chosen cell:              "
        f"{cell_index}"
    )

    print(
        "Associated UEs:           "
        f"{num_cell_ues}"
    )

    if num_cell_ues < 2:
        raise RuntimeError(
            "Need at least two associated UEs "
            "for this MU-MIMO sanity check."
        )

    tx_power_total_w = dbm_to_watt(
        44.0,
        precision=channel_config.precision,
        device=device,
    )

    tx_power_per_subcarrier_w = (
        tx_power_total_w
        / channel_config.num_subcarriers
    )

    noise_power_per_subcarrier_w = (
        BOLTZMANN_CONSTANT
        * 294.0
        * channel_config.subcarrier_spacing_hz
    )

    noise_power_per_subcarrier_w = torch.as_tensor(
        noise_power_per_subcarrier_w,
        dtype=torch.float32,
        device=device,
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

    serving_mimo = (
        extract_serving_mimo_rbg_channel(
            h_freq=channel.h_freq,
            serving_bs=association.serving_bs,
            num_rbgs=18,
        )
    )

    csi_data = compute_ideal_svd_csi(
        h_serving_rbg=(
            serving_mimo.h_serving_rbg
        ),
        rank1_rbg_score=(
            rank_data
            .rank1_rbg_spectral_efficiency
        ),
        rank2_rbg_score=(
            rank_data
            .rank2_rbg_spectral_efficiency
        ),
    )

    ue_a = (
        cell_ue_indices[
            0:1
        ]
    )

    ue_a_b = (
        cell_ue_indices[
            0:2
        ]
    )

    global_ue_a = int(
        ue_a[0].item()
    )

    global_ue_b = int(
        ue_a_b[1].item()
    )

    rank_a = int(
        csi_data.recommended_rank[
            0,
            global_ue_a,
        ].item()
    )

    rank_b = int(
        csi_data.recommended_rank[
            0,
            global_ue_b,
        ].item()
    )

    print(
        "UE A:                     "
        f"{global_ue_a}, rank {rank_a}"
    )

    print(
        "UE B:                     "
        f"{global_ue_b}, rank {rank_b}"
    )

    rbg_index = 6

    print(
        "Test RBG:                 "
        f"{rbg_index}"
    )

    link_config = LinkAdaptationConfig(
        device=device,
    )

    rate_config = RateConfig(
        device=device,
    )

    (
        ranks_a,
        alpha_a,
        rbg_a,
        rate_a,
    ) = evaluate_user_set(
        h_freq=channel.h_freq,
        csi_data=csi_data,
        selected_global_ue_indices=ue_a,
        serving_cell_index=cell_index,
        rbg_index=rbg_index,
        subcarriers_per_rbg=(
            channel_config.subcarriers_per_rb
        ),
        tx_power_per_subcarrier_w=(
            tx_power_per_subcarrier_w
        ),
        noise_power_per_subcarrier_w=(
            noise_power_per_subcarrier_w
        ),
        link_config=link_config,
        rate_config=rate_config,
    )

    (
        ranks_ab,
        alpha_ab,
        rbg_ab,
        rate_ab,
    ) = evaluate_user_set(
        h_freq=channel.h_freq,
        csi_data=csi_data,
        selected_global_ue_indices=ue_a_b,
        serving_cell_index=cell_index,
        rbg_index=rbg_index,
        subcarriers_per_rbg=(
            channel_config.subcarriers_per_rb
        ),
        tx_power_per_subcarrier_w=(
            tx_power_per_subcarrier_w
        ),
        noise_power_per_subcarrier_w=(
            noise_power_per_subcarrier_w
        ),
        link_config=link_config,
        rate_config=rate_config,
    )

    print_user_set_result(
        title="UE A Alone",
        selected_global_ue_indices=ue_a,
        selected_ranks=ranks_a,
        rzf_alpha=alpha_a,
        rbg_data=rbg_a,
        rate_data=rate_a,
    )

    print_user_set_result(
        title="UE A + UE B",
        selected_global_ue_indices=ue_a_b,
        selected_ranks=ranks_ab,
        rzf_alpha=alpha_ab,
        rbg_data=rbg_ab,
        rate_data=rate_ab,
    )

    throughput_a = float(
        rate_a
        .total_target_compliant_rate_bps
        .item()
    )

    throughput_ab = float(
        rate_ab
        .total_target_compliant_rate_bps
        .item()
    )

    throughput_change = (
        throughput_ab
        - throughput_a
    )

    throughput_a_mbps = (
        throughput_a
        / 1.0e6
    )

    throughput_ab_mbps = (
        throughput_ab
        / 1.0e6
    )

    throughput_change_mbps = (
        throughput_change
        / 1.0e6
    )

    print()
    print("=" * 72)
    print("Spatial-Scheduling Comparison")
    print("=" * 72)

    print(
        "A alone:                  "
        f"{throughput_a_mbps:.3f} Mbps"
    )

    print(
        "A + B:                    "
        f"{throughput_ab_mbps:.3f} Mbps"
    )

    print(
        "Change after adding B:    "
        f"{throughput_change_mbps:+.3f} Mbps"
    )

    if throughput_ab > throughput_a:
        print(
            "Decision: adding UE B improves "
            "total RBG throughput."
        )
    else:
        print(
            "Decision: adding UE B does NOT improve "
            "total RBG throughput."
        )

    print()
    print("=" * 72)
    print("Real MU-MIMO RBG rate sanity PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()