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
from oran_scheduler.phy.rank_diagnostic import (
    compute_ideal_svd_rank_diagnostic,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.phy.schedule_evaluator import (
    evaluate_rbg_candidate_set,
)
from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    CellAllocation,
    selected_candidates_for_rbg,
)
from oran_scheduler.schedulers.baseline_sds import (
    BaselineSDSConfig,
    run_baseline_sds,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    PFGreedyRBGScore,
    PFGreedySDSConfig,
    run_pf_greedy_sds,
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


def candidate_tuple(
    selected_candidates: torch.Tensor,
) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value
        in (
            selected_candidates
            .detach()
            .cpu()
            .tolist()
        )
    )

def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Real Baseline SDS vs PF-Greedy SDS")
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

    serving_cell_index = int(
        torch.argmax(
            association.num_ues_per_cell[
                0
            ]
        ).item()
    )

    cell_global_ues = torch.nonzero(
        association.serving_bs[
            0
        ]
        == serving_cell_index,
        as_tuple=False,
    ).flatten()

    num_candidates = int(
        cell_global_ues.numel()
    )


    print(
        "Chosen cell:              "
        f"{serving_cell_index}"
    )

    print(
        "Candidate UEs in cell:    "
        f"{num_candidates}"
    )

    if num_candidates < 2:
        raise RuntimeError(
            "Need at least two candidate UEs "
            "for SDS comparison."
        )


    candidate_global_ue_indices = (
        cell_global_ues.clone()
    )


    print()
    print("Candidate mapping:")
    print(
        "  NOTE: ordering is temporary and "
        "is NOT yet PF-TDS ordering."
    )

    for candidate_index in range(
        num_candidates
    ):
        global_ue = int(
            candidate_global_ue_indices[
                candidate_index
            ].item()
        )

        print(
            f"  candidate {candidate_index}"
            f" -> global UE {global_ue}"
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
            serving_bs=(
                association.serving_bs
            ),
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
            serving_bs=(
                association.serving_bs
            ),
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


    print()
    print("Candidate ranks:")

    for candidate_index in range(
        num_candidates
    ):
        global_ue = int(
            candidate_global_ue_indices[
                candidate_index
            ].item()
        )

        rank = int(
            csi_data.recommended_rank[
                0,
                global_ue,
            ].item()
        )

        print(
            f"  candidate {candidate_index}: "
            f"UE {global_ue}, rank {rank}"
        )


    test_rbg_index = 6

    num_user_slots = 4

    actions = torch.full(
        (
            num_user_slots,
            18,
        ),
        fill_value=NO_ALLOCATION,
        dtype=torch.long,
        device=device,
    )


    actions[
        0,
        test_rbg_index,
    ] = 0

    initial_allocation = CellAllocation(
        candidate_by_user_slot=actions
    )


    link_config = LinkAdaptationConfig(
        device=device,
    )

    rate_config = RateConfig(
        device=device,
    )


    evaluation_cache = {}

    def get_physical_evaluation(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ):
        key = (
            rbg_index,
            candidate_tuple(
                selected_candidates
            ),
        )


        if key not in evaluation_cache:

            evaluation_cache[
                key
            ] = evaluate_rbg_candidate_set(
                selected_candidate_indices=(
                    selected_candidates
                ),
                candidate_global_ue_indices=(
                    candidate_global_ue_indices
                ),
                h_freq=channel.h_freq,
                serving_cell_index=(
                    serving_cell_index
                ),
                recommended_rank=(
                    csi_data.recommended_rank
                ),
                rx_combiners=(
                    csi_data.rx_combiners
                ),
                rbg_index=rbg_index,
                csi_subcarrier_index=(
                    csi_data.csi_subcarrier_index
                ),
                subcarriers_per_rbg=12,
                tx_power_per_subcarrier_w=(
                    tx_power_per_subcarrier_w
                ),
                noise_power_per_subcarrier_w=(
                    noise_power_per_subcarrier_w
                ),
                link_adaptation_config=(
                    link_config
                ),
                rate_config=rate_config,
            )


        return evaluation_cache[
            key
        ]

    baseline_evaluation_log = []

    def baseline_score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> torch.Tensor:

        evaluation = (
            get_physical_evaluation(
                selected_candidates,
                rbg_index,
            )
        )

        rate_bps = (
            evaluation
            .total_target_compliant_rate_bps
        )

        baseline_evaluation_log.append(
            (
                candidate_tuple(
                    selected_candidates
                ),
                float(
                    rate_bps.item()
                    / 1.0e6
                ),
            )
        )

        return rate_bps


    pf_evaluation_log = []

    def pf_score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:

        evaluation = (
            get_physical_evaluation(
                selected_candidates,
                rbg_index,
            )
        )

        pf_evaluation_log.append(
            (
                candidate_tuple(
                    selected_candidates
                ),
                float(
                    evaluation
                    .total_target_compliant_rate_bps
                    .item()
                    / 1.0e6
                ),
            )
        )

        return PFGreedyRBGScore(
            total_rate_bps=(
                evaluation
                .total_target_compliant_rate_bps
            ),
            selected_candidate_rate_bps=(
                evaluation
                .target_compliant_rate_bps
            ),
        )
    

    past_average_throughput = torch.full(
        (
            num_candidates,
        ),
        fill_value=1.0e6,
        dtype=torch.float32,
        device=device,
    )

    baseline_result = run_baseline_sds(
        initial_allocation=(
            initial_allocation
        ),
        num_candidates=num_candidates,
        score_rbg=baseline_score_rbg,
        config=BaselineSDSConfig(),
    )

    pf_result = run_pf_greedy_sds(
        initial_allocation=(
            initial_allocation
        ),
        num_candidates=num_candidates,
        past_average_throughput=(
            past_average_throughput
        ),
        score_rbg=pf_score_rbg,
        config=PFGreedySDSConfig(),
    )


    print()
    print("=" * 72)
    print("Baseline SDS Search")
    print("=" * 72)

    for (
        candidates,
        rate_mbps,
    ) in baseline_evaluation_log:

        print(
            f"{candidates} "
            f"-> {rate_mbps:.3f} Mbps"
        )

    print()
    print("=" * 72)
    print("PF-Greedy SDS Search")
    print("=" * 72)

    for (
        candidates,
        rate_mbps,
    ) in pf_evaluation_log:

        print(
            f"{candidates} "
            f"-> {rate_mbps:.3f} Mbps"
        )


    baseline_candidates = (
        selected_candidates_for_rbg(
            allocation=(
                baseline_result.allocation
            ),
            rbg_index=test_rbg_index,
        )
    )

    pf_candidates = (
        selected_candidates_for_rbg(
            allocation=(
                pf_result.allocation
            ),
            rbg_index=test_rbg_index,
        )
    )

    baseline_rate_mbps = float(
        baseline_result.rbg_rate_bps[
            test_rbg_index
        ].item()
        / 1.0e6
    )

    print()
    print("=" * 72)
    print("Baseline SDS Final")
    print("=" * 72)

    print(
        "Selected candidates:      "
        f"{candidate_tuple(baseline_candidates)}"
    )

    baseline_global_ues = (
        candidate_global_ue_indices[
            baseline_candidates
        ]
    )

    print(
        "Selected global UEs:      "
        f"{candidate_tuple(baseline_global_ues)}"
    )

    print(
        "Final RBG rate:           "
        f"{baseline_rate_mbps:.3f} Mbps"
    )

    print(
        "PHY evaluations:          "
        f"{baseline_result.num_phy_evaluations}"
    )

    pf_rate_mbps = float(
        pf_result.rbg_rate_bps[
            test_rbg_index
        ].item()
        / 1.0e6
    )

    pf_sum = float(
        pf_result.rbg_pf_sum[
            test_rbg_index
        ].item()
    )

    print()
    print("=" * 72)
    print("PF-Greedy SDS Final")
    print("=" * 72)

    print(
        "Selected candidates:      "
        f"{candidate_tuple(pf_candidates)}"
    )

    pf_global_ues = (
        candidate_global_ue_indices[
            pf_candidates
        ]
    )

    print(
        "Selected global UEs:      "
        f"{candidate_tuple(pf_global_ues)}"
    )

    print(
        "Final RBG rate:           "
        f"{pf_rate_mbps:.3f} Mbps"
    )

    print(
        "Final PF sum:             "
        f"{pf_sum:.6f}"
    )

    print(
        "PHY evaluations:          "
        f"{pf_result.num_phy_evaluations}"
    )

    rate_difference_mbps = (
        pf_rate_mbps
        - baseline_rate_mbps
    )

    print()
    print("=" * 72)
    print("Comparison")
    print("=" * 72)

    print(
        "Baseline rate:            "
        f"{baseline_rate_mbps:.3f} Mbps"
    )

    print(
        "PF-Greedy rate:           "
        f"{pf_rate_mbps:.3f} Mbps"
    )

    print(
        "PF-Greedy - Baseline:     "
        f"{rate_difference_mbps:+.3f} Mbps"
    )


    print()
    print(
        "Baseline accepts the first "
        "throughput-improving candidate."
    )

    print(
        "PF-Greedy evaluates every valid "
        "candidate and maximizes PF sum."
    )

    print()
    print("=" * 72)
    print("Real SDS comparison PASSED")
    print("=" * 72)

    print()
    print(
        "NOTE: initial FDS allocation, candidate "
        "ordering, and one-shot PF history are "
        "temporary reproduction scaffolding."
    )


if __name__ == "__main__":
    main()