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


def print_rbg_evaluation(
    title: str,
    evaluation,
) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        "Scheduled UEs:            "
        f"{evaluation.num_scheduled_ues}"
    )

    print(
        "Physical layers:          "
        f"{evaluation.num_physical_layers}"
    )

    print(
        "RZF alpha:                "
        f"{evaluation.rzf_alpha:.6e}"
    )

    num_ues = (
        evaluation.num_scheduled_ues
    )

    for ue_position in range(
        num_ues
    ):
        candidate_index = int(
            evaluation
            .selected_candidate_indices[
                ue_position
            ]
            .item()
        )

        global_ue = int(
            evaluation
            .selected_global_ue_indices[
                ue_position
            ]
            .item()
        )

        rank = int(
            evaluation.selected_ranks[
                ue_position
            ]
            .item()
        )

        rate_mbps = float(
            evaluation
            .target_compliant_rate_bps[
                ue_position
            ]
            .item()
            / 1.0e6
        )

        print()

        print(
            f"Candidate {candidate_index}:"
        )

        print(
            "  global UE:              "
            f"{global_ue}"
        )

        print(
            "  rank:                   "
            f"{rank}"
        )

        print(
            "  target-compliant rate:  "
            f"{rate_mbps:.3f} Mbps"
        )


    total_mbps = float(
        evaluation
        .total_target_compliant_rate_bps
        .item()
        / 1.0e6
    )

    print()

    print(
        "TOTAL RBG rate:           "
        f"{total_mbps:.3f} Mbps"
    )


def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Real Baseline SDS Sanity")
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

    num_cell_ues = int(
        cell_global_ues.numel()
    )

    print(
        "Chosen cell:              "
        f"{serving_cell_index}"
    )

    print(
        "Associated UEs:           "
        f"{num_cell_ues}"
    )

    if num_cell_ues < 2:
        raise RuntimeError(
            "This sanity test needs at least "
            "two UEs in one cell."
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

    num_cell_ues = int(
        cell_global_ues.numel()
    )

    print(
        "Chosen cell:              "
        f"{serving_cell_index}"
    )

    print(
        "Associated UEs:           "
        f"{num_cell_ues}"
    )

    if num_cell_ues < 2:
        raise RuntimeError(
            "This sanity test needs at least "
            "two UEs in one cell."
        )

    candidate_global_ue_indices = (
        cell_global_ues.clone()
    )

    num_candidates = int(
        candidate_global_ue_indices.numel()
    )

    print()
    print("Temporary candidate order:")
    print(
        "  NOTE: this is NOT yet PF-TDS ordering."
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

    noise_power_per_subcarrier_w = (
        torch.as_tensor(
            noise_power_per_subcarrier_w,
            dtype=torch.float32,
            device=device,
        )
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
    evaluation_log = []


    def score_rbg(
        selected_candidate_indices: torch.Tensor,
        rbg_index: int,
    ) -> torch.Tensor:

        candidate_tuple = tuple(
            int(value)
            for value
            in (
                selected_candidate_indices
                .detach()
                .cpu()
                .tolist()
            )
        )

        cache_key = (
            rbg_index,
            candidate_tuple,
        )


        if cache_key not in evaluation_cache:

            evaluation = (
                evaluate_rbg_candidate_set(
                    selected_candidate_indices=(
                        selected_candidate_indices
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
                        csi_data
                        .csi_subcarrier_index
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
                    rate_config=(
                        rate_config
                    ),
                )
            )

            evaluation_cache[
                cache_key
            ] = evaluation

        evaluation = (
            evaluation_cache[
                cache_key
            ]
        )

        rate_bps = (
            evaluation
            .total_target_compliant_rate_bps
        )


        evaluation_log.append(
            (
                rbg_index,
                candidate_tuple,
                float(
                    rate_bps.item()
                    / 1.0e6
                ),
            )
        )

        return rate_bps

    result = run_baseline_sds(
        initial_allocation=(
            initial_allocation
        ),
        num_candidates=num_candidates,
        score_rbg=score_rbg,
        config=BaselineSDSConfig(),
    )

    print()
    print("=" * 72)
    print("Baseline SDS PHY Evaluations")
    print("=" * 72)


    for (
        rbg_index,
        candidate_tuple,
        rate_mbps,
    ) in evaluation_log:

        print(
            f"RBG {rbg_index}: "
            f"candidates {candidate_tuple} "
            f"-> {rate_mbps:.3f} Mbps"
        )

    initial_candidates = (
        selected_candidates_for_rbg(
            allocation=(
                initial_allocation
            ),
            rbg_index=(
                test_rbg_index
            ),
        )
    )

    final_candidates = (
        selected_candidates_for_rbg(
            allocation=result.allocation,
            rbg_index=(
                test_rbg_index
            ),
        )
    )


    initial_key = (
        test_rbg_index,
        tuple(
            int(value)
            for value
            in (
                initial_candidates
                .detach()
                .cpu()
                .tolist()
            )
        ),
    )

    final_key = (
        test_rbg_index,
        tuple(
            int(value)
            for value
            in (
                final_candidates
                .detach()
                .cpu()
                .tolist()
            )
        ),
    )


    if final_key not in evaluation_cache:
        score_rbg(
            final_candidates,
            test_rbg_index,
        )


    print_rbg_evaluation(
        title="Initial FDS Allocation",
        evaluation=(
            evaluation_cache[
                initial_key
            ]
        ),
    )


    print_rbg_evaluation(
        title="Final Baseline SDS Allocation",
        evaluation=(
            evaluation_cache[
                final_key
            ]
        ),
    )


    print()
    print("=" * 72)
    print("Final Scheduler Decision")
    print("=" * 72)

    print(
        "RBG:                      "
        f"{test_rbg_index}"
    )

    print(
        "Selected candidates:      "
        f"{tuple(int(x) for x in final_candidates.tolist())}"
    )

    final_global_ues = (
        candidate_global_ue_indices[
            final_candidates
        ]
    )

    print(
        "Selected global UEs:      "
        f"{tuple(int(x) for x in final_global_ues.tolist())}"
    )

    final_rate_mbps = float(
        result.rbg_rate_bps[
            test_rbg_index
        ].item()
        / 1.0e6
    )

    print(
        "Final RBG rate:           "
        f"{final_rate_mbps:.3f} Mbps"
    )

    print(
        "PHY evaluations:          "
        f"{result.num_phy_evaluations}"
    )

    print()
    print("=" * 72)
    print("Real Baseline SDS sanity PASSED")
    print("=" * 72)

    print()
    print(
        "NOTE: slot-0 initial FDS allocation and "
        "candidate ordering are temporary scaffolding."
    )


if __name__ == "__main__":
    main()



    