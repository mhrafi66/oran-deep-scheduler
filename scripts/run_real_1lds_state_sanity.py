import torch

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
from oran_scheduler.phy.sinr import (
    SINRConfig,
    compute_noise_power_per_subcarrier_w,
    compute_subcarrier_tx_power_w,
)
from oran_scheduler.phy.su_mimo_reports import (
    build_single_user_phy_reports,
)

from oran_scheduler.phy.schedule_evaluator import (
    evaluate_cell_allocation,
)


from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    selected_candidates_for_rbg,
)
from oran_scheduler.schedulers.one_lds_loop import (
    run_1lds_user_slot_loop,
)

from oran_scheduler.schedulers.throughput_history import (
    update_cell_throughput_history,
)
# from oran_scheduler.schedulers.classical_frontend import (
#     build_classical_initial_cell_allocation,
#     run_classical_frontend,
# )
# from oran_scheduler.schedulers.pf_tds import (
#     PFTimeDomainConfig,
# )
# from oran_scheduler.schedulers.su_mimo_fds import (
#     SUMIMOFDSConfig,
# )
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
from oran_scheduler.state.cqi_features import (
    CQISurrogateConfig,
    build_cqi_surrogate,
)
# from oran_scheduler.state.one_lds import (
#     OneLDSRawFeatures,
#     OneLDSStateConfig,
#     build_1lds_state,
# )
# from oran_scheduler.state.spatial_features import (
#     build_spatial_allocation_features,
# )
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)
from oran_scheduler.state.one_lds_decision import (
    OneLDSDecisionInputs,
    build_1lds_decision_data,
)

def print_first_candidate_features(
    feature_segment: torch.Tensor,
    num_rbgs: int,
) -> None:
    """
    Print one candidate UE's normalized 1LDS feature segment.

    Reproduction ordering:

        0       past average throughput
        1       rank
        2       allocated RBG count
        3       DL buffer
        4       wideband CQI
        5:23    sub-band CQIs
        23:41   max precoder cross-correlations
    """

    values = (
        feature_segment
        .detach()
        .cpu()
    )

    subband_start = 5
    subband_end = (
        subband_start
        + num_rbgs
    )

    correlation_start = (
        subband_end
    )

    correlation_end = (
        correlation_start
        + num_rbgs
    )

    print()
    print("Candidate-0 normalized 1LDS features")
    print("-" * 72)

    print(
        "Past average throughput:  "
        f"{float(values[0].item()):.6f}"
    )

    print(
        "Rank:                     "
        f"{float(values[1].item()):.6f}"
    )

    print(
        "Allocated RBG count:      "
        f"{float(values[2].item()):.6f}"
    )

    print(
        "DL buffer:                "
        f"{float(values[3].item()):.6f}"
    )

    print(
        "Wideband CQI:             "
        f"{float(values[4].item()):.6f}"
    )

    subband_values = (
        values[
            subband_start:
            subband_end
        ]
        .tolist()
    )

    correlation_values = (
        values[
            correlation_start:
            correlation_end
        ]
        .tolist()
    )

    print()
    print("18 normalized sub-band CQIs:")
    print(subband_values)

    print()
    print("18 max precoder cross-correlations:")
    print(correlation_values)

def main() -> None:
    device = "cuda:0"

    num_candidates = 10
    num_rbgs = 18

    # PAPER-SPECIFIED training configuration:
    #
    # Maximum MU-MIMO UE layers |L| = 4.
    #
    # Here one "user slot" means one co-scheduled UE
    # position on an RBG, NOT one physical MIMO stream.
    #
    # A rank-2 UE selected in one user slot later
    # expands into two physical MIMO streams.
    num_user_slots = 4

    # TEMPORARY one-shot scheduler-history initialization.
    #
    # The paper specifies the past-throughput feature but
    # does not publish the complete initialization procedure.
    initial_history_bps = 1.0e6

    # OPEN-REPRODUCTION PARAMETER.
    #
    # The paper specifies exponential smoothing of
    # past average throughput but does not publicly
    # provide the exact epsilon used by the simulator.
    #
    # For this integration sanity we use:
    #
    #     R_new
    #       =
    #     (1 - epsilon) * delivered_rate
    #     + epsilon * R_old
    #
    history_forgetting_factor = 0.95

    # TEMPORARY full-buffer-like placeholder.
    #
    # This is not yet a traffic/buffer simulator.
    temporary_buffer_value = 1.0

    print()
    print("=" * 72)
    print("Real 21-Cell -> 410-Dimensional 1LDS State Sanity")
    print("=" * 72)

    print(
        "Device:                    "
        f"{device}"
    )

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,

        # Memory-safe integration configuration.
        #
        # This is deliberately NOT the paper's 420-UE
        # training population.
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

    rbg_config = RBGConfig(
        num_rbgs=num_rbgs,
        subcarriers_per_rb=12,
        rbs_per_rbg=1,
    )

    print(
        "Generating paper-array "
        "multi-cell channel..."
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
        "Raw MIMO H shape:          "
        f"{channel_shape}"
    )

    association = build_cell_association(
        h_freq=channel.h_freq,
        precision=channel_config.precision,
    )

    min_ues_per_cell = int(
        association
        .num_ues_per_cell
        .min()
        .item()
    )

    max_ues_per_cell = int(
        association
        .num_ues_per_cell
        .max()
        .item()
    )

    print(
        "Minimum associated UEs:    "
        f"{min_ues_per_cell}"
    )

    print(
        "Maximum associated UEs:    "
        f"{max_ues_per_cell}"
    )



    # ------------------------------------------------------------------
    # Scheduler serving-cell layout
    # ------------------------------------------------------------------

    # This averaged RBG-power representation is used ONLY to
    # reuse the existing serving-cell padded-layout builder.
    #
    # It is NOT used as the physical rate/CQI input below.
    rbg_power_data = compute_rbg_channel_power(
        channel=channel,
        config=rbg_config,
    )

    serving_data = build_serving_cell_data(
        rbg_data=rbg_power_data,
        association=association,
    )

    print(
        "Serving-layout shape:      "
        f"{tuple(serving_data.global_ue_indices.shape)}"
    )

    # ------------------------------------------------------------------
    # Per-subcarrier power and receiver noise
    # ------------------------------------------------------------------

    sinr_config = SINRConfig(
        tx_power_dbm=44.0,
        subcarrier_spacing_hz=(
            channel_config
            .subcarrier_spacing_hz
        ),
        temperature_k=294.0,
        receiver_noise_figure_db=None,
        precision=channel_config.precision,
        device=device,
    )

    tx_power_per_subcarrier_w = (
        compute_subcarrier_tx_power_w(
            config=sinr_config,
            num_subcarriers=(
                channel_config
                .num_subcarriers
            ),
        )
    )

    noise_power_per_subcarrier_w = (
        compute_noise_power_per_subcarrier_w(
            config=sinr_config,
        )
    )

    tx_subcarrier_value = float(
        tx_power_per_subcarrier_w.item()
    )

    noise_subcarrier_value = float(
        noise_power_per_subcarrier_w.item()
    )

    print(
        "TX power / subcarrier:     "
        f"{tx_subcarrier_value:.6e} W"
    )

    print(
        "Noise / subcarrier:        "
        f"{noise_subcarrier_value:.6e} W"
    )

    # ------------------------------------------------------------------
    # Open-reproduction CSI / RI / PMI-direction surrogate
    # ------------------------------------------------------------------

    rank_data = (
        compute_ideal_svd_rank_diagnostic(
            h_freq=channel.h_freq,
            serving_bs=association.serving_bs,
            num_rbgs=num_rbgs,
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
            num_rbgs=num_rbgs,
        )
    )

    csi_data = compute_ideal_svd_csi(
        h_serving_rbg=(
            serving_mimo
            .h_serving_rbg
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

    print(
        "Recommended-rank shape:    "
        f"{tuple(csi_data.recommended_rank.shape)}"
    )

    print(
        "Precoder-direction shape:  "
        f"{tuple(csi_data.precoder_directions.shape)}"
    )

    # ------------------------------------------------------------------
    # Vectorized physical SU-MIMO reports
    # ------------------------------------------------------------------

    link_adaptation_config = (
        LinkAdaptationConfig(
            num_rbgs=num_rbgs,
            device=device,
        )
    )

    rate_config = RateConfig(
        device=device,
    )

    su_report = build_single_user_phy_reports(
        h_freq=channel.h_freq,
        serving_bs=association.serving_bs,
        recommended_rank=(
            csi_data.recommended_rank
        ),
        precoder_directions=(
            csi_data.precoder_directions
        ),
        num_rbgs=num_rbgs,
        subcarriers_per_rbg=(
            rbg_config
            .subcarriers_per_rbg
        ),
        tx_power_per_subcarrier_w=(
            tx_power_per_subcarrier_w
        ),
        noise_power_per_subcarrier_w=(
            noise_power_per_subcarrier_w
        ),
        link_adaptation_config=(
            link_adaptation_config
        ),
        rate_config=rate_config,
    )

    global_rbg_rate = (
        su_report
        .rate
        .target_compliant_rate_bps
    )

    expected_global_rate_shape = (
        1,
        topology_config.num_ues,
        num_rbgs,
    )

    assert tuple(
        global_rbg_rate.shape
    ) == expected_global_rate_shape

    print(
        "Global SU rate shape:      "
        f"{tuple(global_rbg_rate.shape)}"
    )

    # ------------------------------------------------------------------
    # Global physical rate -> padded serving-cell scheduler layout
    # ------------------------------------------------------------------

    serving_rbg_rate = (
        gather_serving_ue_rbg_values(
            global_ue_rbg_values=(
                global_rbg_rate
            ),
            serving_data=serving_data,
        )
    )

    print(
        "Serving SU rate shape:     "
        f"{tuple(serving_rbg_rate.shape)}"
    )

    # TEMPORARY scheduler history for this one-shot state.
    #
    # Every real UE starts with 1 Mbps history.
    # Padding remains zero.
    past_average_throughput = (
        serving_data
        .valid_ue_mask
        .to(
            dtype=torch.float32
        )
        * initial_history_bps
    )

    # frontend = run_classical_frontend(
    #     ue_rbg_rate=serving_rbg_rate,
    #     global_ue_indices=(
    #         serving_data
    #         .global_ue_indices
    #     ),
    #     valid_ue_mask=(
    #         serving_data
    #         .valid_ue_mask
    #     ),
    #     past_average_throughput=(
    #         past_average_throughput
    #     ),
    #     tds_config=PFTimeDomainConfig(
    #         num_candidates=num_candidates,
    #     ),
    #     fds_config=SUMIMOFDSConfig(),
    # )

    # candidate_shape = tuple(
    #     frontend
    #     .tds_result
    #     .candidate_indices
    #     .shape
    # )

    # print(
    #     "PF candidate shape:        "
    #     f"{candidate_shape}"
    # )

    # assert candidate_shape == (
    #     1,
    #     topology_config.num_cells,
    #     num_candidates,
    # )


    # ------------------------------------------------------------------
    # PF time-domain candidate selection
    # ------------------------------------------------------------------
    #
    # PAPER-SPECIFIED:
    #     PF TDS supplies at most 10 candidate UEs.
    #
    # IMPORTANT:
    #     We deliberately do NOT run the open PF-based FDS
    #     surrogate here. The deep scheduler begins from the
    #     TDS candidate set and an empty allocation.

    td_instantaneous_rate = (
        serving_rbg_rate.sum(
            dim=-1
        )
    )

    tds_result = run_pf_tds(
        instantaneous_rate=(
            td_instantaneous_rate
        ),
        past_average_throughput=(
            past_average_throughput
        ),
        config=PFTimeDomainConfig(
            num_candidates=num_candidates,
        ),
        valid_ue_mask=(
            serving_data
            .valid_ue_mask
        ),
    )

    candidate_global_ue_indices_all = (
        gather_candidate_global_ue_indices(
            serving_data=serving_data,
            candidate_indices=(
                tds_result
                .candidate_indices
            ),
            candidate_valid_mask=(
                tds_result
                .candidate_valid_mask
            ),
        )
    )

    candidate_history_all = (
        gather_candidate_ue_values(
            ue_values=(
                past_average_throughput
            ),
            candidate_indices=(
                tds_result
                .candidate_indices
            ),
            candidate_valid_mask=(
                tds_result
                .candidate_valid_mask
            ),
        )
    )

    candidate_shape = tuple(
        tds_result
        .candidate_indices
        .shape
    )

    print(
        "PF candidate shape:        "
        f"{candidate_shape}"
    )

    assert candidate_shape == (
        1,
        topology_config.num_cells,
        num_candidates,
    )

    # ------------------------------------------------------------------
    # Pick the cell with the most associated UEs for detailed inspection.
    # ------------------------------------------------------------------

    cell_index = int(
        torch.argmax(
            serving_data
            .num_ues_per_cell[
                0
            ]
        )
        .item()
    )

    num_cell_ues = int(
        serving_data
        .num_ues_per_cell[
            0,
            cell_index,
        ]
        .item()
    )

    # candidate_valid_mask = (
    #     frontend
    #     .tds_result
    #     .candidate_valid_mask[
    #         0,
    #         cell_index,
    #         :,
    #     ]
    # )

    # candidate_global_ue_indices = (
    #     frontend
    #     .candidate_global_ue_indices[
    #         0,
    #         cell_index,
    #         :,
    #     ]
    # )

    # candidate_pf_metric = (
    #     frontend
    #     .tds_result
    #     .candidate_metrics[
    #         0,
    #         cell_index,
    #         :,
    #     ]
    # )

    # candidate_history = (
    #     frontend
    #     .candidate_past_average_throughput[
    #         0,
    #         cell_index,
    #         :,
    #     ]
    # )

    candidate_valid_mask = (
        tds_result
        .candidate_valid_mask[
            0,
            cell_index,
            :,
        ]
    )

    candidate_global_ue_indices = (
        candidate_global_ue_indices_all[
            0,
            cell_index,
            :,
        ]
    )

    candidate_pf_metric = (
        tds_result
        .candidate_metrics[
            0,
            cell_index,
            :,
        ]
    )

    candidate_history = (
        candidate_history_all[
            0,
            cell_index,
            :,
        ]
    )




    num_valid_candidates = int(
        candidate_valid_mask
        .sum()
        .item()
    )

    print()
    print("=" * 72)
    print("Chosen Cell")
    print("=" * 72)

    print(
        "Cell index:                "
        f"{cell_index}"
    )

    print(
        "Associated UEs:            "
        f"{num_cell_ues}"
    )

    print(
        "Valid candidate slots:     "
        f"{num_valid_candidates}"
    )

    candidate_id_list = (
        candidate_global_ue_indices
        .detach()
        .cpu()
        .tolist()
    )

    candidate_mask_list = (
        candidate_valid_mask
        .detach()
        .cpu()
        .tolist()
    )

    candidate_pf_list = (
        candidate_pf_metric
        .detach()
        .cpu()
        .tolist()
    )

    print()
    print("Candidate global UE IDs:")
    print(candidate_id_list)

    print()
    print("Candidate validity mask:")
    print(candidate_mask_list)

    print()
    print("Candidate PF metrics:")
    print(candidate_pf_list)

    # Invalid candidate slots contain global UE -1.
    #
    # Never index tensors using those -1 values.
    safe_global_ue_indices = torch.where(
        candidate_valid_mask,
        candidate_global_ue_indices,
        torch.zeros_like(
            candidate_global_ue_indices
        ),
    )

    candidate_rank = (
        csi_data
        .recommended_rank[
            0,
            safe_global_ue_indices,
        ]
    )

    # Give invalid slots a safe rank internally.
    # The complete feature segment will later be blanked.
    candidate_rank = torch.where(
        candidate_valid_mask,
        candidate_rank,
        torch.ones_like(
            candidate_rank
        ),
    )

    candidate_precoder_directions = (
        csi_data
        .precoder_directions[
            0,
            safe_global_ue_indices,
            :,
            :,
            :,
        ]
    )

    direction_valid_mask = (
        candidate_valid_mask[
            :,
            None,
            None,
            None,
        ]
    )

    candidate_precoder_directions = (
        torch.where(
            direction_valid_mask,
            candidate_precoder_directions,
            torch.zeros_like(
                candidate_precoder_directions
            ),
        )
    )

    print(
        "Candidate rank shape:      "
        f"{tuple(candidate_rank.shape)}"
    )

    print(
        "Candidate precoder shape:  "
        f"{tuple(candidate_precoder_directions.shape)}"
    )

    # ------------------------------------------------------------------
    # Physical SU report -> candidate CQI surrogate
    # ------------------------------------------------------------------

    serving_mcs_index = (
        gather_serving_ue_rbg_values(
            global_ue_rbg_values=(
                su_report
                .link_adaptation
                .mcs_index
            ),
            serving_data=serving_data,
        )
    )

    serving_meets_target = (
        gather_serving_ue_rbg_values(
            global_ue_rbg_values=(
                su_report
                .link_adaptation
                .meets_bler_target
            ),
            serving_data=serving_data,
        )
    )

    candidate_mcs_index = (
        gather_candidate_rbg_values(
            ue_rbg_values=(
                serving_mcs_index
            ),
            candidate_indices=(
                tds_result
                .candidate_indices
            ),
            candidate_valid_mask=(
                tds_result
                .candidate_valid_mask
            ),
        )[
            0,
            cell_index,
            :,
            :,
        ]
    )

    candidate_meets_target = (
        gather_candidate_rbg_values(
            ue_rbg_values=(
                serving_meets_target
            ),
            candidate_indices=(
                tds_result
                .candidate_indices
            ),
            candidate_valid_mask=(
                tds_result
                .candidate_valid_mask
            ),
        )[
            0,
            cell_index,
            :,
            :,
        ]
    )

    cqi_features = build_cqi_surrogate(
        mcs_index=candidate_mcs_index,
        meets_bler_target=(
            candidate_meets_target
        ),
        candidate_valid_mask=(
            candidate_valid_mask
        ),
        config=CQISurrogateConfig(),
    )

    print(
        "Sub-band CQI shape:        "
        f"{tuple(cqi_features.subband_cqi.shape)}"
    )

    print(
        "Wideband CQI shape:        "
        f"{tuple(cqi_features.wideband_cqi.shape)}"
    )

    # ------------------------------------------------------------------
    # Temporary initial FDS allocation
    # ------------------------------------------------------------------

    # allocation = (
    #     build_classical_initial_cell_allocation(
    #         frontend_data=frontend,
    #         batch_index=0,
    #         cell_index=cell_index,
    #         num_user_slots=num_user_slots,
    #     )
    # )

    # print(
    #     "Allocation shape:          "
    #     f"{tuple(allocation.candidate_by_user_slot.shape)}"
    # )

    # slot_zero = (
    #     allocation
    #     .candidate_by_user_slot[
    #         0,
    #         :,
    #     ]
    # )

    # slot_zero_list = (
    #     slot_zero
    #     .detach()
    #     .cpu()
    #     .tolist()
    # )

    # print()
    # print(
    #     "Initial FDS candidate slot "
    #     "per RBG:"
    # )

    # print(slot_zero_list)

    # spatial_features = (
    #     build_spatial_allocation_features(
    #         allocation=allocation,
    #         candidate_rank=candidate_rank,
    #         candidate_precoder_directions=(
    #             candidate_precoder_directions
    #         ),
    #         candidate_valid_mask=(
    #             candidate_valid_mask
    #         ),
    #     )
    # )

    # allocated_count_list = (
    #     spatial_features
    #     .allocated_rbg_count
    #     .detach()
    #     .cpu()
    #     .tolist()
    # )

    # print()
    # print("Allocated RBG count by candidate:")
    # print(allocated_count_list)

    # ------------------------------------------------------------------
    # TEMPORARY buffer state
    # ------------------------------------------------------------------

    dl_buffer = (
        candidate_valid_mask
        .to(
            dtype=torch.float32
        )
        * temporary_buffer_value
    )

    # ------------------------------------------------------------------
    # Sanity-only normalization constants
    # ------------------------------------------------------------------
    #
    # Paper:
    #   throughput -> R / R_max
    #   buffer     -> b / b_max
    #   subband    -> g / g_bar
    #
    # The public paper does not provide the numerical R_max,
    # b_max, or g_bar required for exact reproduction.
    #
    # For THIS one-shot integration:
    #
    #   history = 1 Mbps for every valid UE
    #   R_max   = 1 Mbps
    #
    #   buffer  = 1
    #   b_max   = 1
    #
    #   CQI surrogate range = 0..15
    #   g_bar = 15
    #
    # These must NOT later be described as paper constants.

    state_config = OneLDSStateConfig(
        throughput_normalization_bps=(
            initial_history_bps
        ),
        buffer_normalization=(
            temporary_buffer_value
        ),
        subband_cqi_normalization=15.0,
        num_candidates=num_candidates,
        num_rbgs=num_rbgs,
        max_rank=2,
        wideband_cqi_min=0.0,
        wideband_cqi_max=15.0,
    )


    decision_inputs = OneLDSDecisionInputs(
        past_average_throughput=(
            candidate_history
        ),
        rank=candidate_rank,
        dl_buffer=dl_buffer,
        wideband_cqi=(
            cqi_features
            .wideband_cqi
        ),
        subband_cqi=(
            cqi_features
            .subband_cqi
        ),
        candidate_precoder_directions=(
            candidate_precoder_directions
        ),
        candidate_valid_mask=(
            candidate_valid_mask
        ),
    )

    def controlled_policy(
        user_slot_index: int,
        decision,
    ) -> torch.Tensor:
        """
        Deterministic integration-test policy.

        This is NOT an RL policy.

        For every RBG:

            1. Try candidate UEs in a rotating order.
            2. Choose the first candidate allowed by
               the current action mask.
            3. If no real candidate is available,
               choose NO ALLOCATION.

        This intentionally exercises state evolution
        and action masking across all UE slots.
        """

        action_mask = decision.action_mask

        actions = torch.full(
            (
                num_rbgs,
            ),
            fill_value=num_candidates,
            dtype=torch.long,
            device=device,
        )

        for rbg_index in range(
            num_rbgs
        ):
            for offset in range(
                num_candidates
            ):
                candidate_index = (
                    rbg_index
                    + user_slot_index
                    + offset
                ) % num_candidates

                candidate_is_valid = bool(
                    action_mask[
                        rbg_index,
                        candidate_index,
                    ]
                    .item()
                )

                if candidate_is_valid:
                    actions[
                        rbg_index
                    ] = candidate_index

                    break

        return actions

    schedule_result = run_1lds_user_slot_loop(
        num_user_slots=num_user_slots,
        inputs=decision_inputs,
        state_config=state_config,
        action_policy=controlled_policy,
        device=device,
    )


    assert len(
        schedule_result.decisions
    ) == num_user_slots

    assert tuple(
        schedule_result.actions.shape
    ) == (
        num_user_slots,
        num_rbgs,
    )

    assert tuple(
        schedule_result
        .allocation
        .candidate_by_user_slot
        .shape
    ) == (
        num_user_slots,
        num_rbgs,
    )

    print()
    print("=" * 72)
    print("Complete 1LDS UE-Slot Loop")
    print("=" * 72)

    for user_slot_index in range(
        num_user_slots
    ):
        decision = (
            schedule_result
            .decisions[
                user_slot_index
            ]
        )

        actions = (
            schedule_result
            .actions[
                user_slot_index,
                :,
            ]
        )

        allocated_counts = (
            decision
            .spatial_features
            .allocated_rbg_count
            .detach()
            .cpu()
            .tolist()
        )

        maximum_correlation = float(
            decision
            .spatial_features
            .max_precoder_cross_correlation
            .max()
            .item()
        )

        print()
        print(
            f"UE slot {user_slot_index}"
        )

        print(
            "  state shape:           "
            f"{tuple(decision.state_data.state.shape)}"
        )

        print(
            "  action-mask shape:     "
            f"{tuple(decision.action_mask.shape)}"
        )

        print(
            "  allocated counts "
            "before action:"
        )

        print(
            f"    {allocated_counts}"
        )

        print(
            "  maximum correlation "
            "before action: "
            f"{maximum_correlation:.6f}"
        )

        print(
            "  chosen actions:"
        )

        print(
            "    "
            f"{actions.detach().cpu().tolist()}"
        )

    final_allocation = (
        schedule_result
        .allocation
    )

    final_allocation_values = (
        final_allocation
        .candidate_by_user_slot
        .detach()
        .cpu()
        .tolist()
    )

    print()
    print("=" * 72)
    print("Final 1LDS Candidate Allocation")
    print("=" * 72)

    print(
        final_allocation_values
    )

    for rbg_index in range(
        num_rbgs
    ):
        selected_candidates = []

        for user_slot_index in range(
            num_user_slots
        ):
            candidate_index = int(
                final_allocation
                .candidate_by_user_slot[
                    user_slot_index,
                    rbg_index,
                ]
                .item()
            )

            if candidate_index == (
                NO_ALLOCATION
            ):
                continue

            selected_candidates.append(
                candidate_index
            )

        assert len(
            selected_candidates
        ) == len(
            set(
                selected_candidates
            )
        )

    print()
    print("=" * 72)
    print("UE Slots vs Physical MIMO Streams")
    print("=" * 72)

    for rbg_index in range(
        num_rbgs
    ):
        scheduled_ues = 0
        physical_streams = 0

        for user_slot_index in range(
            num_user_slots
        ):
            candidate_index = int(
                final_allocation
                .candidate_by_user_slot[
                    user_slot_index,
                    rbg_index,
                ]
                .item()
            )

            if candidate_index == (
                NO_ALLOCATION
            ):
                continue

            scheduled_ues += 1

            ue_rank = int(
                candidate_rank[
                    candidate_index
                ]
                .item()
            )

            physical_streams += (
                ue_rank
            )

        print(
            f"RBG {rbg_index:2d}: "
            f"{scheduled_ues} scheduled UEs, "
            f"{physical_streams} physical streams"
        )
    for decision in (
        schedule_result.decisions
    ):
        assert tuple(
            decision
            .state_data
            .state
            .shape
        ) == (
            410,
        )

        assert tuple(
            decision
            .action_mask
            .shape
        ) == (
            18,
            11,
        )

        assert torch.isfinite(
            decision
            .state_data
            .state
        ).all()


    # ------------------------------------------------------------------
    # Final 1LDS allocation -> common MU-MIMO PHY
    # ------------------------------------------------------------------
    #
    # The deep scheduler has now finished all of its
    # UE-slot allocation decisions.
    #
    # Only NOW do we evaluate the completed schedule
    # through the physical MU-MIMO chain:
    #
    # allocation
    #     -> rank expansion
    #     -> RZF
    #     -> post-RZF MRC
    #     -> SINR
    #     -> MCS / TBLER
    #     -> rate

    allocation_evaluation = evaluate_cell_allocation(
        allocation=final_allocation,
        candidate_global_ue_indices=(
            candidate_global_ue_indices
        ),
        h_freq=channel.h_freq,
        serving_cell_index=cell_index,
        recommended_rank=(
            csi_data.recommended_rank
        ),
        rx_combiners=(
            csi_data.rx_combiners
        ),
        csi_subcarrier_index=(
            csi_data.csi_subcarrier_index
        ),
        subcarriers_per_rbg=(
            rbg_config.subcarriers_per_rbg
        ),
        tx_power_per_subcarrier_w=(
            tx_power_per_subcarrier_w
        ),
        noise_power_per_subcarrier_w=(
            noise_power_per_subcarrier_w
        ),
        link_adaptation_config=(
            link_adaptation_config
        ),
        rate_config=rate_config,
        batch_index=0,
    )

    assert tuple(
        allocation_evaluation
        .candidate_target_compliant_rate_bps
        .shape
    ) == (
        num_candidates,
        num_rbgs,
    )

    assert tuple(
        allocation_evaluation
        .total_rbg_target_compliant_rate_bps
        .shape
    ) == (
        num_rbgs,
    )

    assert tuple(
        allocation_evaluation
        .num_scheduled_ues_per_rbg
        .shape
    ) == (
        num_rbgs,
    )

    assert tuple(
        allocation_evaluation
        .num_physical_layers_per_rbg
        .shape
    ) == (
        num_rbgs,
    )

    assert tuple(
        allocation_evaluation
        .rzf_alpha_by_rbg
        .shape
    ) == (
        num_rbgs,
    )

    assert torch.isfinite(
        allocation_evaluation
        .candidate_target_compliant_rate_bps
    ).all()

    assert torch.isfinite(
        allocation_evaluation
        .total_rbg_target_compliant_rate_bps
    ).all()

    assert torch.isfinite(
        allocation_evaluation
        .rzf_alpha_by_rbg
    ).all()

    assert torch.all(
        allocation_evaluation
        .candidate_target_compliant_rate_bps
        >= 0.0
    )

    # ------------------------------------------------------------------
    # Cross-check:
    #
    # scheduler UE slots
    #     versus
    # physical MIMO streams
    # ------------------------------------------------------------------

    for rbg_index in range(
        num_rbgs
    ):
        selected_candidates = (
            selected_candidates_for_rbg(
                allocation=final_allocation,
                rbg_index=rbg_index,
            )
        )

        expected_num_ues = int(
            selected_candidates.numel()
        )

        if expected_num_ues == 0:
            expected_num_layers = 0

        else:
            expected_num_layers = int(
                candidate_rank[
                    selected_candidates
                ]
                .sum()
                .item()
            )

        actual_num_ues = int(
            allocation_evaluation
            .num_scheduled_ues_per_rbg[
                rbg_index
            ]
            .item()
        )

        actual_num_layers = int(
            allocation_evaluation
            .num_physical_layers_per_rbg[
                rbg_index
            ]
            .item()
        )

        assert actual_num_ues == (
            expected_num_ues
        )

        assert actual_num_layers == (
            expected_num_layers
        )

    expected_rbg_total_rate = (
        allocation_evaluation
        .candidate_target_compliant_rate_bps
        .sum(
            dim=0
        )
    )

    torch.testing.assert_close(
        allocation_evaluation
        .total_rbg_target_compliant_rate_bps,
        expected_rbg_total_rate,
    )

    expected_cell_total_rate = (
        allocation_evaluation
        .total_rbg_target_compliant_rate_bps
        .sum()
    )

    torch.testing.assert_close(
        allocation_evaluation
        .total_cell_target_compliant_rate_bps,
        expected_cell_total_rate,
    )

    print()
    print("=" * 72)
    print(
        "Final 1LDS Allocation -> "
        "Real MU-MIMO PHY"
    )
    print("=" * 72)

    for rbg_index in range(
        num_rbgs
    ):
        selected_candidates = (
            selected_candidates_for_rbg(
                allocation=final_allocation,
                rbg_index=rbg_index,
            )
        )

        selected_candidate_list = (
            selected_candidates
            .detach()
            .cpu()
            .tolist()
        )

        selected_global_ues = (
            candidate_global_ue_indices[
                selected_candidates
            ]
        )

        selected_global_ue_list = (
            selected_global_ues
            .detach()
            .cpu()
            .tolist()
        )

        selected_ranks = (
            candidate_rank[
                selected_candidates
            ]
        )

        selected_rank_list = (
            selected_ranks
            .detach()
            .cpu()
            .tolist()
        )

        num_scheduled_ues = int(
            allocation_evaluation
            .num_scheduled_ues_per_rbg[
                rbg_index
            ]
            .item()
        )

        num_physical_layers = int(
            allocation_evaluation
            .num_physical_layers_per_rbg[
                rbg_index
            ]
            .item()
        )

        rzf_alpha = float(
            allocation_evaluation
            .rzf_alpha_by_rbg[
                rbg_index
            ]
            .item()
        )

        rbg_total_rate_mbps = float(
            allocation_evaluation
            .total_rbg_target_compliant_rate_bps[
                rbg_index
            ]
            .item()
            / 1.0e6
        )

        print()
        print(
            f"RBG {rbg_index:2d}"
        )

        print(
            "  candidate slots:       "
            f"{selected_candidate_list}"
        )

        print(
            "  global UE IDs:         "
            f"{selected_global_ue_list}"
        )

        print(
            "  UE ranks:              "
            f"{selected_rank_list}"
        )

        print(
            "  scheduled UEs:         "
            f"{num_scheduled_ues}"
        )

        print(
            "  physical streams:      "
            f"{num_physical_layers}"
        )

        print(
            "  RZF alpha:             "
            f"{rzf_alpha:.6e}"
        )

        print(
            "  total PHY rate:        "
            f"{rbg_total_rate_mbps:.6f} Mbps"
        )

        print(
            "  per-UE target-compliant rates:"
        )

        for candidate_index_tensor in (
            selected_candidates
        ):
            candidate_index = int(
                candidate_index_tensor.item()
            )

            global_ue_index = int(
                candidate_global_ue_indices[
                    candidate_index
                ]
                .item()
            )

            ue_rate_mbps = float(
                allocation_evaluation
                .candidate_target_compliant_rate_bps[
                    candidate_index,
                    rbg_index,
                ]
                .item()
                / 1.0e6
            )

            print(
                "    "
                f"candidate {candidate_index}, "
                f"global UE {global_ue_index}: "
                f"{ue_rate_mbps:.6f} Mbps"
            )
    candidate_total_rate_bps = (
        allocation_evaluation
        .candidate_target_compliant_rate_bps
        .sum(
            dim=1
        )
    )

    print()
    print("=" * 72)
    print("Candidate Throughput Across All RBGs")
    print("=" * 72)

    for candidate_index in range(
        num_candidates
    ):
        if not bool(
            candidate_valid_mask[
                candidate_index
            ]
            .item()
        ):
            continue

        global_ue_index = int(
            candidate_global_ue_indices[
                candidate_index
            ]
            .item()
        )

        rank_value = int(
            candidate_rank[
                candidate_index
            ]
            .item()
        )

        candidate_rate_mbps = float(
            candidate_total_rate_bps[
                candidate_index
            ]
            .item()
            / 1.0e6
        )

        print(
            f"Candidate {candidate_index}: "
            f"global UE {global_ue_index}, "
            f"rank {rank_value}, "
            f"{candidate_rate_mbps:.6f} Mbps"
        )

    # ------------------------------------------------------------------
    # Realized PHY throughput -> next-TTI throughput history
    # ------------------------------------------------------------------

    chosen_cell_candidate_indices = (
        tds_result
        .candidate_indices[
            0,
            cell_index,
            :,
        ]
    )

    chosen_cell_previous_history = (
        past_average_throughput[
            0,
            cell_index,
            :,
        ]
    )

    chosen_cell_serving_valid_mask = (
        serving_data
        .valid_ue_mask[
            0,
            cell_index,
            :,
        ]
    )

    history_update = update_cell_throughput_history(
        previous_average_throughput_bps=(
            chosen_cell_previous_history
        ),
        serving_ue_valid_mask=(
            chosen_cell_serving_valid_mask
        ),
        candidate_indices=(
            chosen_cell_candidate_indices
        ),
        candidate_valid_mask=(
            candidate_valid_mask
        ),
        candidate_delivered_rate_bps=(
            candidate_total_rate_bps
        ),
        forgetting_factor=(
            history_forgetting_factor
        ),
    )

    assert tuple(
        history_update
        .delivered_rate_bps
        .shape
    ) == tuple(
        chosen_cell_previous_history.shape
    )

    assert tuple(
        history_update
        .updated_average_throughput_bps
        .shape
    ) == tuple(
        chosen_cell_previous_history.shape
    )

    assert torch.isfinite(
        history_update
        .delivered_rate_bps
    ).all()

    assert torch.isfinite(
        history_update
        .updated_average_throughput_bps
    ).all()

    assert torch.all(
        history_update
        .delivered_rate_bps
        >= 0.0
    )

    assert torch.all(
        history_update
        .updated_average_throughput_bps
        >= 0.0
    )

    expected_updated_history = (
        (
            1.0
            - history_forgetting_factor
        )
        * history_update.delivered_rate_bps
        + history_forgetting_factor
        * chosen_cell_previous_history
    )

    expected_updated_history = torch.where(
        chosen_cell_serving_valid_mask,
        expected_updated_history,
        torch.zeros_like(
            expected_updated_history
        ),
    )

    torch.testing.assert_close(
        history_update
        .updated_average_throughput_bps,
        expected_updated_history,
    )

    print()
    print("=" * 72)
    print("Throughput History: TTI t -> TTI t+1")
    print("=" * 72)

    print(
        "Forgetting factor epsilon: "
        f"{history_forgetting_factor:.4f}"
    )

    print()
    print(
        "History is shown in the persistent "
        "serving-UE-slot order."
    )
    chosen_cell_global_ue_indices = (
        serving_data
        .global_ue_indices[
            0,
            cell_index,
            :,
        ]
    )

    num_serving_slots = (
        chosen_cell_previous_history.shape[0]
    )

    for serving_slot_index in range(
        num_serving_slots
    ):
        if not bool(
            chosen_cell_serving_valid_mask[
                serving_slot_index
            ]
            .item()
        ):
            continue

        global_ue_index = int(
            chosen_cell_global_ue_indices[
                serving_slot_index
            ]
            .item()
        )

        previous_history_mbps = float(
            chosen_cell_previous_history[
                serving_slot_index
            ]
            .item()
            / 1.0e6
        )

        delivered_rate_mbps = float(
            history_update
            .delivered_rate_bps[
                serving_slot_index
            ]
            .item()
            / 1.0e6
        )

        updated_history_mbps = float(
            history_update
            .updated_average_throughput_bps[
                serving_slot_index
            ]
            .item()
            / 1.0e6
        )

        print()
        print(
            f"Serving slot {serving_slot_index}, "
            f"global UE {global_ue_index}"
        )

        print(
            "  previous history:      "
            f"{previous_history_mbps:.6f} Mbps"
        )

        print(
            "  delivered this TTI:    "
            f"{delivered_rate_mbps:.6f} Mbps"
        )

        print(
            "  history for next TTI:  "
            f"{updated_history_mbps:.6f} Mbps"
        )

    next_tti_past_average_throughput = (
        past_average_throughput.clone()
    )

    next_tti_past_average_throughput[
        0,
        cell_index,
        :,
    ] = (
        history_update
        .updated_average_throughput_bps
    )

    assert tuple(
        next_tti_past_average_throughput.shape
    ) == tuple(
        past_average_throughput.shape
    )



    # print(
    #     "- epsilon = 0.95 is currently an "
    #     "open-reproduction parameter."
    # )
















    total_cell_rate_mbps = float(
        allocation_evaluation
        .total_cell_target_compliant_rate_bps
        .item()
        / 1.0e6
    )

    print()
    print(
        "Total chosen-cell target-compliant "
        "PHY rate:"
    )

    print(
        f"{total_cell_rate_mbps:.6f} Mbps"
    )


    print()
    print("=" * 72)
    print(
        "REAL 1LDS -> FINAL ALLOCATION -> "
        "MU-MIMO PHY SANITY PASSED"
    )
    print("=" * 72)

    print()
    print("End-to-end path validated:")

    print(
        "PF TDS"
    )

    print(
        "  -> candidate UE set"
    )

    print(
        "  -> 410-dimensional 1LDS states"
    )

    print(
        "  -> four UE-slot decisions"
    )

    print(
        "  -> final CellAllocation"
    )

    print(
        "  -> rank expansion to physical streams"
    )

    print(
        "  -> joint RZF precoding"
    )

    print(
        "  -> physical post-RZF MRC"
    )

    print(
        "  -> SINR / MCS / TBLER"
    )

    print(
        "  -> per-UE and cell throughput"
    )

    print(
        "  -> per-UE and cell throughput"
    )


    print(
        "  -> serving-UE delivered throughput"
    )

    print(
        "  -> updated past-average throughput"
    )

    print(
        "  -> history ready for the next TTI"
    )

    print()
    print("IMPORTANT:")

    print(
        "- The controlled action callback is "
        "still NOT an RL policy."
    )

    print(
        "- SVD directions remain CSI/PMI-direction "
        "surrogates."
    )

    # print(
    #     "- Current throughput history and buffer "
    #     "values remain temporary placeholders."
    # )

    print(
        "- Initial throughput history and epsilon "
        "remain open-reproduction choices."
    )

    print(
        "- DL buffer values remain temporary "
        "placeholders."
    )

    print(
        "- epsilon = 0.95 is currently an "
        "open-reproduction parameter."
    )


    print("=" * 72)

    print()
    print("Interpretation:")

    print(
        "- PF TDS supplies the UE candidate set."
    )

    print(
        "- The 1LDS scheduler begins with an "
        "empty allocation."
    )

    print(
        "- One complete all-RBG decision is made "
        "per scheduler UE slot."
    )

    print(
        "- The state and action mask are rebuilt "
        "after every UE-slot allocation."
    )

    print(
        "- A rank-2 UE is selected once but "
        "represents two physical MIMO streams."
    )

    print(
        "- The controlled callback is only an "
        "integration-test policy, not RL."
    )

    # allocation_0 = build_empty_cell_allocation(
    #     num_user_slots=num_user_slots,
    #     num_rbgs=num_rbgs,
    #     device=device,
    # )

    # print()
    # print("=" * 72)
    # print("1LDS User-Slot 0 Decision")
    # print("=" * 72)

    # print()
    # print("Allocation before slot 0:")
    # print(
    #     allocation_0
    #     .candidate_by_user_slot
    #     .detach()
    #     .cpu()
    #     .tolist()
    # )


    # decision_0 = build_1lds_decision_data(
    #     allocation=allocation_0,
    #     user_slot_index=0,
    #     inputs=decision_inputs,
    #     config=state_config,
    # )

    # assert tuple(
    #     decision_0
    #     .state_data
    #     .state
    #     .shape
    # ) == (
    #     410,
    # )

    # assert tuple(
    #     decision_0
    #     .action_mask
    #     .shape
    # ) == (
    #     18,
    #     11,
    # )

    # assert torch.all(
    #     decision_0
    #     .spatial_features
    #     .allocated_rbg_count
    #     == 0
    # )

    # assert torch.all(
    #     decision_0
    #     .spatial_features
    #     .max_precoder_cross_correlation
    #     == 0.0
    # )

    # print(
    #     "state_0 shape:             "
    #     f"{tuple(decision_0.state_data.state.shape)}"
    # )

    # print(
    #     "action mask_0 shape:       "
    #     f"{tuple(decision_0.action_mask.shape)}"
    # )

    # print(
    #     "Allocated counts state_0:  "
    #     f"{decision_0.spatial_features.allocated_rbg_count.detach().cpu().tolist()}"
    # )

    # print(
    #     "Max correlation state_0:   "
    #     f"{float(decision_0.spatial_features.max_precoder_cross_correlation.max().item()):.6f}"
    # )

    # if num_valid_candidates <= 0:
    #     raise RuntimeError(
    #         "Chosen cell has no valid PF candidates."
    #     )

    # controlled_actions = (
    #     torch.arange(
    #         num_rbgs,
    #         device=device,
    #         dtype=torch.long,
    #     )
    #     % num_valid_candidates
    # )

    # # Deliberately leave the final RBG unallocated.
    # #
    # # Actor convention:
    # #     action == num_candidates
    # # means NO ALLOCATION.
    # controlled_actions[
    #     -1
    # ] = num_candidates

    # print()
    # print("Controlled slot-0 actor actions:")
    # print(
    #     controlled_actions
    #     .detach()
    #     .cpu()
    #     .tolist()
    # )

    # allocation_1 = apply_user_slot_actions(
    #     allocation=allocation_0,
    #     user_slot_index=0,
    #     actions=controlled_actions,
    #     num_candidates=num_candidates,
    #     candidate_valid_mask=(
    #         candidate_valid_mask
    #     ),
    # )

    # print()
    # print("Allocation after slot 0:")
    # print(
    #     allocation_1
    #     .candidate_by_user_slot
    #     .detach()
    #     .cpu()
    #     .tolist()
    # )

    # decision_1 = build_1lds_decision_data(
    #     allocation=allocation_1,
    #     user_slot_index=1,
    #     inputs=decision_inputs,
    #     config=state_config,
    # )

    # assert tuple(
    #     decision_1
    #     .state_data
    #     .state
    #     .shape
    # ) == (
    #     410,
    # )

    # assert tuple(
    #     decision_1
    #     .action_mask
    #     .shape
    # ) == (
    #     18,
    #     11,
    # )

    # assert not torch.equal(
    #     decision_0
    #     .state_data
    #     .state,
    #     decision_1
    #     .state_data
    #     .state,
    # )

    # num_allocated_rbgs = int(
    #     (
    #         allocation_1
    #         .candidate_by_user_slot[
    #             0,
    #             :,
    #         ]
    #         != NO_ALLOCATION
    #     )
    #     .sum()
    #     .item()
    # )

    # assert num_allocated_rbgs == (
    #     num_rbgs - 1
    # )

    # allocated_count_sum = int(
    #     decision_1
    #     .spatial_features
    #     .allocated_rbg_count
    #     .sum()
    #     .item()
    # )

    # assert allocated_count_sum == (
    #     num_allocated_rbgs
    # )

    # # The final RBG was deliberately left empty.
    # assert int(
    #     allocation_1
    #     .candidate_by_user_slot[
    #         0,
    #         -1,
    #     ]
    #     .item()
    # ) == NO_ALLOCATION

    # # Therefore no candidate has an already-scheduled
    # # pairing partner on the final RBG.
    # assert torch.all(
    #     decision_1
    #     .spatial_features
    #     .max_precoder_cross_correlation[
    #         :,
    #         -1,
    #     ]
    #     == 0.0
    # )

    # # NO ALLOCATION remains legal on every RBG.
    # assert torch.all(
    #     decision_1
    #     .action_mask[
    #         :,
    #         num_candidates,
    #     ]
    # )


    # for rbg_index in range(
    #     num_rbgs - 1
    # ):
    #     selected_candidate = int(
    #         allocation_1
    #         .candidate_by_user_slot[
    #             0,
    #             rbg_index,
    #         ]
    #         .item()
    #     )

    #     assert selected_candidate != (
    #         NO_ALLOCATION
    #     )

    #     assert not bool(
    #         decision_1
    #         .action_mask[
    #             rbg_index,
    #             selected_candidate,
    #         ]
    #         .item()
    #     )

    # print()
    # print("=" * 72)
    # print("1LDS User-Slot 1 Decision")
    # print("=" * 72)

    # state_1_counts = (
    #     decision_1
    #     .spatial_features
    #     .allocated_rbg_count
    #     .detach()
    #     .cpu()
    #     .tolist()
    # )

    # print(
    #     "state_1 shape:             "
    #     f"{tuple(decision_1.state_data.state.shape)}"
    # )

    # print(
    #     "action mask_1 shape:       "
    #     f"{tuple(decision_1.action_mask.shape)}"
    # )

    # print()
    # print("Allocated RBG counts in state_1:")
    # print(state_1_counts)

    # print()
    # print(
    #     "Per-candidate max correlation "
    #     "after slot 0:"
    # )

    # for candidate_index in range(
    #     num_candidates
    # ):
    #     if not bool(
    #         candidate_valid_mask[
    #             candidate_index
    #         ].item()
    #     ):
    #         continue

    #     correlation = (
    #         decision_1
    #         .spatial_features
    #         .max_precoder_cross_correlation[
    #             candidate_index,
    #             :,
    #         ]
    #     )

    #     maximum_value = float(
    #         correlation.max().item()
    #     )

    #     print(
    #         f"Candidate {candidate_index}: "
    #         f"{maximum_value:.6f}"
    #     )


    # if num_valid_candidates > 0:
    #     print()
    #     print(
    #         "Candidate-0 feature segment "
    #         "BEFORE slot-0 action:"
    #     )

    #     print_first_candidate_features(
    #         feature_segment=(
    #             decision_0
    #             .state_data
    #             .ue_feature_segments[
    #                 0
    #             ]
    #         ),
    #         num_rbgs=num_rbgs,
    #     )

    #     print()
    #     print(
    #         "Candidate-0 feature segment "
    #         "AFTER slot-0 action:"
    #     )

    #     print_first_candidate_features(
    #         feature_segment=(
    #             decision_1
    #             .state_data
    #             .ue_feature_segments[
    #                 0
    #             ]
    #         ),
    #         num_rbgs=num_rbgs,
    #     )

    # print()
    # print("=" * 72)
    # print(
    #     "REAL 1LDS STATE -> ACTION -> "
    #     "NEXT-STATE SANITY PASSED"
    # )
    # print("=" * 72)

    # print()
    # print("Interpretation:")

    # print(
    #     "- state_0 is built from an EMPTY "
    #     "deep-scheduler allocation."
    # )

    # print(
    #     "- controlled_actions stand in for "
    #     "the future actor output."
    # )

    # print(
    #     "- allocation slot 0 is updated from "
    #     "those actions."
    # )

    # print(
    #     "- state_1 is rebuilt from the updated "
    #     "allocation."
    # )

    # print(
    #     "- allocated-RBG count, correlation, "
    #     "and action mask evolve automatically."
    # )

    # print()
    # print("Reproduction notes:")

    # print(
    #     "- No PF-based FDS allocation initializes "
    #     "the deep scheduler."
    # )

    # print(
    #     "- SVD transmit directions remain "
    #     "PMI-direction surrogates."
    # )

    # print(
    #     "- Recommended rank remains an ideal-SVD "
    #     "RI surrogate."
    # )

    # print(
    #     "- DL buffer and initial throughput history "
    #     "remain temporary placeholders."
    # )



               

    # raw_features = OneLDSRawFeatures(
    #     past_average_throughput=(
    #         candidate_history
    #     ),
    #     rank=candidate_rank,
    #     allocated_rbg_count=(
    #         spatial_features
    #         .allocated_rbg_count
    #     ),
    #     dl_buffer=dl_buffer,
    #     wideband_cqi=(
    #         cqi_features
    #         .wideband_cqi
    #     ),
    #     subband_cqi=(
    #         cqi_features
    #         .subband_cqi
    #     ),
    #     max_precoder_cross_correlation=(
    #         spatial_features
    #         .max_precoder_cross_correlation
    #     ),
    #     candidate_valid_mask=(
    #         candidate_valid_mask
    #     ),
    # )

    # state_data = build_1lds_state(
    #     features=raw_features,
    #     config=state_config,
    # )

    # # ------------------------------------------------------------------
    # # State validation
    # # ------------------------------------------------------------------

    # expected_segment_shape = (
    #     num_candidates,
    #     41,
    # )

    # expected_state_shape = (
    #     410,
    # )

    # assert tuple(
    #     state_data
    #     .ue_feature_segments
    #     .shape
    # ) == expected_segment_shape

    # assert tuple(
    #     state_data
    #     .state
    #     .shape
    # ) == expected_state_shape

    # assert state_config.ue_feature_size == 41
    # assert state_config.state_size == 410
    # assert state_config.actor_output_size == 198

    # assert torch.isfinite(
    #     state_data.state
    # ).all()

    # invalid_candidate_mask = (
    #     ~candidate_valid_mask
    # )

    # if bool(
    #     invalid_candidate_mask.any().item()
    # ):
    #     invalid_segments = (
    #         state_data
    #         .ue_feature_segments[
    #             invalid_candidate_mask
    #         ]
    #     )

    #     assert torch.all(
    #         invalid_segments == 0.0
    #     )

    # # ------------------------------------------------------------------
    # # Existing allocation -> next 1LDS user-slot action mask
    # # ------------------------------------------------------------------

    # next_action_mask = (
    #     build_next_user_slot_action_mask(
    #         allocation=allocation,
    #         num_candidates=num_candidates,
    #         user_slot_index=1,
    #         candidate_valid_mask=(
    #             candidate_valid_mask
    #         ),
    #     )
    # )

    # assert tuple(
    #     next_action_mask.shape
    # ) == (
    #     num_rbgs,
    #     num_candidates + 1,
    # )

    # for rbg_index in range(
    #     num_rbgs
    # ):
    #     selected_candidate = int(
    #         slot_zero[
    #             rbg_index
    #         ]
    #         .item()
    #     )

    #     if selected_candidate != NO_ALLOCATION:
    #         assert not bool(
    #             next_action_mask[
    #                 rbg_index,
    #                 selected_candidate,
    #             ]
    #             .item()
    #         )

    #     # Final action is always NO-ALLOCATION.
    #     assert bool(
    #         next_action_mask[
    #             rbg_index,
    #             num_candidates,
    #         ]
    #         .item()
    #     )

    # # ------------------------------------------------------------------
    # # Human-readable diagnostics
    # # ------------------------------------------------------------------

    # rank_list = []

    # for candidate_index in range(
    #     num_candidates
    # ):
    #     if bool(
    #         candidate_valid_mask[
    #             candidate_index
    #         ]
    #         .item()
    #     ):
    #         rank_value = int(
    #             candidate_rank[
    #                 candidate_index
    #             ]
    #             .item()
    #         )

    #         rank_list.append(
    #             rank_value
    #         )
    #     else:
    #         rank_list.append(
    #             None
    #         )

    # print()
    # print("Candidate ranks:")
    # print(rank_list)

    # wideband_cqi_list = (
    #     cqi_features
    #     .wideband_cqi
    #     .detach()
    #     .cpu()
    #     .tolist()
    # )

    # print()
    # print("Candidate wideband CQIs:")
    # print(wideband_cqi_list)

    # if num_valid_candidates > 0:
    #     first_subband_cqi = (
    #         cqi_features
    #         .subband_cqi[
    #             0,
    #             :,
    #         ]
    #         .detach()
    #         .cpu()
    #         .tolist()
    #     )

    #     first_correlation = (
    #         spatial_features
    #         .max_precoder_cross_correlation[
    #             0,
    #             :,
    #         ]
    #         .detach()
    #         .cpu()
    #         .tolist()
    #     )

    #     print()
    #     print("Candidate-0 sub-band CQIs:")
    #     print(first_subband_cqi)

    #     print()
    #     print(
    #         "Candidate-0 max precoder "
    #         "cross-correlations:"
    #     )

    #     print(first_correlation)

    # print()
    # print("Valid-candidate correlation diagnostics:")

    # for candidate_index in range(
    #     num_candidates
    # ):
    #     if not bool(
    #         candidate_valid_mask[
    #             candidate_index
    #         ]
    #         .item()
    #     ):
    #         continue

    #     correlation_values = (
    #         spatial_features
    #         .max_precoder_cross_correlation[
    #             candidate_index,
    #             :,
    #         ]
    #         .detach()
    #         .cpu()
    #         .tolist()
    #     )

    #     maximum_correlation = float(
    #         spatial_features
    #         .max_precoder_cross_correlation[
    #             candidate_index,
    #             :,
    #         ]
    #         .max()
    #         .item()
    #     )

    #     print()
    #     print(
    #         f"Candidate {candidate_index}:"
    #     )

    #     print(
    #         "  max over RBGs = "
    #         f"{maximum_correlation:.6f}"
    #     )

    #     print(
    #         "  per-RBG values = "
    #         f"{correlation_values}"
    #     )

    # print()
    # print("=" * 72)
    # print("Constructed 1LDS State")
    # print("=" * 72)

    # segment_shape = tuple(
    #     state_data
    #     .ue_feature_segments
    #     .shape
    # )

    # state_shape = tuple(
    #     state_data
    #     .state
    #     .shape
    # )

    # state_min = float(
    #     state_data
    #     .state
    #     .min()
    #     .item()
    # )

    # state_max = float(
    #     state_data
    #     .state
    #     .max()
    #     .item()
    # )

    # state_is_finite = bool(
    #     torch.isfinite(
    #         state_data.state
    #     )
    #     .all()
    #     .item()
    # )

    # print(
    #     "UE feature segments:      "
    #     f"{segment_shape}"
    # )

    # print(
    #     "Flattened state:           "
    #     f"{state_shape}"
    # )

    # print(
    #     "State size:                "
    #     f"{state_data.state.numel()}"
    # )

    # print(
    #     "Expected actor outputs:    "
    #     f"{state_config.actor_output_size}"
    # )

    # print(
    #     "Next action-mask shape:    "
    #     f"{tuple(next_action_mask.shape)}"
    # )

    # print(
    #     "State minimum:             "
    #     f"{state_min:.6f}"
    # )

    # print(
    #     "State maximum:             "
    #     f"{state_max:.6f}"
    # )

    # print(
    #     "All state values finite:   "
    #     f"{state_is_finite}"
    # )

    # if num_valid_candidates > 0:
    #     print_first_candidate_features(
    #         feature_segment=(
    #             state_data
    #             .ue_feature_segments[
    #                 0
    #             ]
    #         ),
    #         num_rbgs=num_rbgs,
    #     )

    # print()
    # print("=" * 72)
    # print(
    #     "REAL 410-DIMENSIONAL "
    #     "1LDS STATE SANITY PASSED"
    # )
    # print("=" * 72)

    # print()
    # print(
    #     "IMPORTANT REPRODUCTION NOTES:"
    # )

    # print(
    #     "- PF-based initial FDS is an "
    #     "open-reproduction surrogate."
    # )

    # print(
    #     "- SVD transmit directions are "
    #     "PMI-direction surrogates."
    # )

    # print(
    #     "- Recommended rank is an ideal-SVD "
    #     "RI surrogate."
    # )

    # print(
    #     "- DL buffer and initial throughput "
    #     "history are temporary placeholders."
    # )

    # print(
    #     "- The 21-UE population is a "
    #     "memory-safe integration configuration."
    # )


if __name__ == "__main__":
    main()