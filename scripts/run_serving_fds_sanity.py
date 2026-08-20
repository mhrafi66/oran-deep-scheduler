import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    run_pf_tds,
)
from oran_scheduler.schedulers.su_mimo_fds import (
    SUMIMOFDSConfig,
    run_su_mimo_fds,
)
from oran_scheduler.simulator.cell_association import (
    build_cell_association,
    print_cell_association_summary,
    validate_cell_association,
)
from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
    validate_frequency_channel,
)
from oran_scheduler.simulator.rbg import (
    RBGConfig,
    compute_rbg_channel_power,
    validate_rbg_channel,
)
from oran_scheduler.simulator.serving_layout import (
    build_serving_cell_data,
    gather_candidate_global_ue_indices,
    gather_candidate_rbg_values,
    gather_candidate_ue_values,
)
from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
    validate_evaluation_topology,
)


def build_temporary_rate_proxy(
    serving_power: torch.Tensor,
    valid_ue_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Build a temporary monotonic rate proxy from serving-link RBG power.

    IMPORTANT:
        This is only integration scaffolding.

        It is NOT:
            - SINR,
            - Shannon capacity,
            - CQI,
            - MCS-derived throughput,
            - a quantity suitable for research results.

        Its only purpose is to validate the complete scheduler
        tensor pipeline before physical achievable rate is implemented.
    """

    if serving_power.ndim != 4:
        raise ValueError(
            "Expected serving_power shape "
            "[batch, cell, UE, RBG], "
            f"got {tuple(serving_power.shape)}."
        )

    if tuple(valid_ue_mask.shape) != tuple(
        serving_power.shape[:3]
    ):
        raise ValueError(
            "valid_ue_mask must match the first three "
            "dimensions of serving_power."
        )


    expanded_mask = (
        valid_ue_mask
        .unsqueeze(-1)
        .expand_as(serving_power)
    )

    valid_power = serving_power[
        expanded_mask
    ]

    if valid_power.numel() == 0:
        return torch.zeros_like(
            serving_power
        )

    positive_power = valid_power[
        valid_power > 0
    ]

    if positive_power.numel() == 0:
        return torch.zeros_like(
            serving_power
        )

    power_scale = positive_power.mean()

    rate_proxy = torch.log1p(
        serving_power
        / power_scale
    )

    return torch.where(
        expanded_mask,
        rate_proxy,
        torch.zeros_like(rate_proxy),
    )

def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Milestone 3C: Serving Layout + PF TDS + SU-MIMO FDS")
    print("=" * 72)
    print(f"Device: {device}")
    print()

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,

        # Paper training configuration:
        # 420 total UEs are generated over 21 sectors.
        num_ut_per_sector=20,

        device=device,
    )

    topology = generate_topology(
        topology_config
    )

    validate_evaluation_topology(
        topology=topology,
        config=topology_config,
    )

    print(
        "Generated topology:       "
        f"{topology_config.num_cells} cells, "
        f"{topology_config.num_ues} UEs"
    )

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,
        num_rbs=18,
        device=device,
    )

    channel = generate_frequency_channel(
        topology=topology,
        topology_config=topology_config,
        channel_config=channel_config,
    )

    validate_frequency_channel(
        channel=channel,
        topology=topology,
        channel_config=channel_config,
    )

    print(
        "Channel shape:            "
        f"{tuple(channel.h_freq.shape)}"
    )

    print(
        "Channel device:           "
        f"{channel.h_freq.device}"
    )


    rbg_config = RBGConfig(
        num_rbgs=18,
        rbs_per_rbg=1,
    )

    rbg_data = compute_rbg_channel_power(
        channel=channel,
        config=rbg_config,
    )

    validate_rbg_channel(
        rbg_data=rbg_data,
        channel=channel,
        config=rbg_config,
    )

    print(
        "All-pair RBG shape:       "
        f"{tuple(rbg_data.power.shape)}"
    )


    association = build_cell_association(
        h_freq=channel.h_freq,
        precision=channel_config.precision,
    )

    validate_cell_association(
        association=association,
        expected_batch_size=(
            topology_config.batch_size
        ),
        expected_num_ues=(
            topology_config.num_ues
        ),
        expected_num_bs=(
            topology_config.num_cells
        ),
    )

    print_cell_association_summary(
        association
    )


    serving_data = build_serving_cell_data(
        rbg_data=rbg_data,
        association=association,
    )

    print(
        "Serving power shape:      "
        f"{tuple(serving_data.power.shape)}"
    )

    print(
        "Serving UE-index shape:   "
        f"{tuple(serving_data.global_ue_indices.shape)}"
    )

    print(
        "Serving valid-mask shape: "
        f"{tuple(serving_data.valid_ue_mask.shape)}"
    )

    rbg_rate_proxy = build_temporary_rate_proxy(
        serving_power=serving_data.power,
        valid_ue_mask=(
            serving_data.valid_ue_mask
        ),
    )

    print(
        "Temporary RBG-rate shape: "
        f"{tuple(rbg_rate_proxy.shape)}"
    )

    wideband_rate_proxy = (
        rbg_rate_proxy.mean(
            dim=-1,
        )
    )

    print(
        "Wideband-rate shape:      "
        f"{tuple(wideband_rate_proxy.shape)}"
    )

    past_average_throughput = torch.ones_like(
        wideband_rate_proxy
    )

    past_average_throughput = torch.where(
        serving_data.valid_ue_mask,
        past_average_throughput,
        torch.zeros_like(
            past_average_throughput
        ),
    )

    pf_tds_result = run_pf_tds(
        instantaneous_rate=wideband_rate_proxy,
        past_average_throughput=(
            past_average_throughput
        ),
        config=PFTimeDomainConfig(
            num_candidates=10,
        ),
        valid_ue_mask=(
            serving_data.valid_ue_mask
        ),
    )

    print(
        "PF candidate shape:       "
        f"{tuple(pf_tds_result.candidate_indices.shape)}"
    )

    print(
        "PF candidate-mask shape:  "
        f"{tuple(pf_tds_result.candidate_valid_mask.shape)}"
    )


    candidate_global_ue_indices = (
        gather_candidate_global_ue_indices(
            serving_data=serving_data,
            candidate_indices=(
                pf_tds_result.candidate_indices
            ),
            candidate_valid_mask=(
                pf_tds_result.candidate_valid_mask
            ),
        )
    )

    print(
        "Candidate-global shape:   "
        f"{tuple(candidate_global_ue_indices.shape)}"
    )

    candidate_rbg_rate = (
        gather_candidate_rbg_values(
            ue_rbg_values=rbg_rate_proxy,
            candidate_indices=(
                pf_tds_result.candidate_indices
            ),
            candidate_valid_mask=(
                pf_tds_result.candidate_valid_mask
            ),
        )
    )

    print(
        "Candidate RBG-rate shape: "
        f"{tuple(candidate_rbg_rate.shape)}"
    )

    candidate_history = gather_candidate_ue_values(
        ue_values=past_average_throughput,
        candidate_indices=(
            pf_tds_result.candidate_indices
        ),
        candidate_valid_mask=(
            pf_tds_result.candidate_valid_mask
        ),
    )

    print(
        "Candidate history shape:  "
        f"{tuple(candidate_history.shape)}"
    )

    fds_result = run_su_mimo_fds(
        candidate_rate=candidate_rbg_rate,
        candidate_past_average_throughput=(
            candidate_history
        ),
        candidate_valid_mask=(
            pf_tds_result.candidate_valid_mask
        ),
        config=SUMIMOFDSConfig(),
    )

    print(
        "FDS selection shape:      "
        f"{tuple(fds_result.selected_candidate_indices.shape)}"
    )

    print(
        "FDS valid-mask shape:     "
        f"{tuple(fds_result.selected_valid_mask.shape)}"
    )

    selected_global_ue_indices = torch.gather(
        candidate_global_ue_indices,
        dim=2,
        index=(
            fds_result.selected_candidate_indices
        ),
    )

    selected_global_ue_indices = torch.where(
        fds_result.selected_valid_mask,
        selected_global_ue_indices,
        torch.full_like(
            selected_global_ue_indices,
            fill_value=-1,
        ),
    )

    print(
        "Selected-global shape:    "
        f"{tuple(selected_global_ue_indices.shape)}"
    )


    example_cell = 0

    print()
    print(f"Example cell {example_cell}")
    print("-" * 72)

    print(
        "Associated UEs: "
        f"{int(serving_data.num_ues_per_cell[0, example_cell].item())}"
    )

    valid_candidate_count = int(
        pf_tds_result.candidate_valid_mask[
            0,
            example_cell,
        ].sum().item()
    )

    print(
        "Valid PF candidates: "
        f"{valid_candidate_count}"
    )

    print(
        "Candidate global UE IDs:"
    )

    print(
        candidate_global_ue_indices[
            0,
            example_cell,
        ]
        .detach()
        .cpu()
        .tolist()
    )

    print(
        "Selected global UE per RBG:"
    )

    print(
        selected_global_ue_indices[
            0,
            example_cell,
        ]
        .detach()
        .cpu()
        .tolist()
    )

    assert tuple(
        pf_tds_result.candidate_indices.shape
    ) == (
        topology_config.batch_size,
        topology_config.num_cells,
        10,
    )

    assert tuple(
        candidate_rbg_rate.shape
    ) == (
        topology_config.batch_size,
        topology_config.num_cells,
        10,
        rbg_config.num_rbgs,
    )

    assert tuple(
        fds_result.selected_candidate_indices.shape
    ) == (
        topology_config.batch_size,
        topology_config.num_cells,
        rbg_config.num_rbgs,
    )

    assert tuple(
        selected_global_ue_indices.shape
    ) == (
        topology_config.batch_size,
        topology_config.num_cells,
        rbg_config.num_rbgs,
    )

    print()
    print("=" * 72)
    print("Milestone 3C integration sanity check PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()