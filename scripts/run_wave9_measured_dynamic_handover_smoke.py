from __future__ import annotations

from pathlib import Path

import torch

from oran_scheduler.rl.dynamic_handover_eval import (
    run_dynamic_multicell_ppo_evaluation,
)

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)

from oran_scheduler.rl.ppo_checkpoint import (
    load_ppo_model_checkpoint,
)

from oran_scheduler.rl.ppo_critic import (
    OneLDSPPOCritic,
    OneLDSPPOCriticConfig,
)

from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)

from oran_scheduler.rl.ppo_multicell_rollout import (
    OneLDSPPOMultiCellRolloutConfig,
    OneLDSPPOMultiCellRolloutController,
)

from oran_scheduler.rl.ppo_multistream_buffer import (
    PPOMultiStreamBufferConfig,
    PPOMultiStreamTransitionBuffer,
)

from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
)

from oran_scheduler.rl.ppo_sionna_chunked import (
    CellChunkedSionnaPPOInputProvider,
    ChunkedSionnaPPOConfig,
    build_chunked_sionna_ppo_context,
)

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)

from oran_scheduler.simulator.dynamic_handover_coordinator import (
    DynamicHandoverCoordinator,
)

from oran_scheduler.simulator.global_traffic_arrivals import (
    GlobalUETrafficArrivalProcess,
)

from oran_scheduler.simulator.global_ue_state import (
    GlobalUESchedulerStateRegistry,
)

from oran_scheduler.simulator.handover import (
    HandoverConfig,
    HandoverController,
)

from oran_scheduler.simulator.sionna_handover_measurement import (
    SionnaHandoverMeasurementConfig,
    SionnaPathlossHandoverMeasurementProvider,
)

from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)

from oran_scheduler.simulator.traffic import (
    build_training_ftp3_config,
)

from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)


DEVICE = torch.device(
    "cuda:0"
)

SEED = 1234

NUM_TTIS = 3

NUM_STREAMS = 2

NUM_CANDIDATES = 10

NUM_RBGS = 18

NUM_USER_SLOTS = 4

TTI_DURATION_S = 0.5e-3

FULL_BUFFER_STATE_BITS = 12_000.0

INITIAL_AVERAGE_THROUGHPUT_BPS = 1.0e6

THROUGHPUT_FORGETTING_FACTOR = 0.9

#
# Controlled integration condition.
#
# We choose a UE whose measured preferred selected
# cell is at least this much stronger than the other
# selected cell.
#
MIN_CONTROLLED_MARGIN_DB = 6.0


CHECKPOINT_PATH = Path(
    "~/oran-deep-scheduler/"
    "experiments/checkpoints/"
    "paper_1lds_ppo_500tti.pt"
).expanduser()


def _db_ratio(
    stronger: torch.Tensor,
    weaker: torch.Tensor,
) -> float:

    return float(
        (
            10.0
            * torch.log10(
                stronger
                / weaker
            )
        ).item()
    )


def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    torch.cuda.set_device(
        DEVICE
    )

    torch.manual_seed(
        SEED
    )

    torch.cuda.reset_peak_memory_stats(
        DEVICE
    )

    print("=" * 78)
    print(
        "WAVE-9 SIONNA-MEASURED DYNAMIC "
        "HANDOVER + PPO SMOKE"
    )
    print("=" * 78)

    print(
        f"GPU: {torch.cuda.get_device_name(DEVICE)}"
    )

    print(
        "Scheduler: frozen pretrained PPO"
    )

    print(
        "Scheduling PHY: real chunked Sionna"
    )

    print(
        "Handover measurement: "
        "Sionna-derived channel-averaged link strength"
    )

    print(
        "Handover scope: two selected cells"
    )

    print(
        "Controlled condition: one UE starts on "
        "the weaker selected cell"
    )

    print(
        "IMPORTANT: this is NOT claimed to be "
        "3GPP A3/RSRP handover."
    )

    print()


    # ==========================================================
    # REAL TEMPORAL SIONNA CONTEXT
    # ==========================================================

    context = (
        build_chunked_sionna_ppo_context(
            config=(
                ChunkedSionnaPPOConfig(
                    num_training_cells=(
                        NUM_STREAMS
                    ),

                    num_ut_per_sector=2,

                    num_rbs=18,

                    num_rbgs=18,

                    subcarriers_per_rb=12,

                    ue_microbatch_size=1,

                    identity_stable_ue_channel_rng=True,

                    topology_seed=42,

                    association_channel_seed=1000,

                    mimo_channel_seed=2000,

                    device="cuda:0",

                    ut_speed_kmh=30.0,

                    temporal_radio_mode=(
                        "velocity_window"
                    ),

                    temporal_window_ttis=(
                        NUM_TTIS
                    ),

                    tti_duration_s=(
                        TTI_DURATION_S
                    ),

                    temporal_max_displacement_m=20.0,
                )
            )
        )
    )

    selected_cells = tuple(
        int(cell)
        for cell
        in context.selected_cell_indices
    )

    if len(selected_cells) != 2:
        raise RuntimeError(
            "Wave-9 smoke requires exactly "
            "two selected cells."
        )

    cell_0 = selected_cells[0]
    cell_1 = selected_cells[1]

    cohort = torch.cat(
        context.global_ue_indices_by_stream,
        dim=0,
    )

    num_ues = int(
        cohort.numel()
    )

    print(
        f"Selected cells: {selected_cells}"
    )

    print(
        f"Closed cohort size: {num_ues}"
    )


    # ==========================================================
    # REAL SIONNA HANDOVER MEASUREMENT
    # ==========================================================

    measurement_provider = (
        SionnaPathlossHandoverMeasurementProvider(
            initial_topology=(
                context.topology
            ),

            topology_config=(
                context.topology_config
            ),

            global_ue_indices=(
                cohort
            ),

            config=(
                SionnaHandoverMeasurementConfig(
                    num_rbs=18,

                    subcarriers_per_rb=12,

                    tti_duration_s=(
                        TTI_DURATION_S
                    ),

                    channel_seed=9100,

                    max_horizontal_displacement_m=(
                        20.0
                    ),

                    device="cuda:0",
                )
            ),
        )
    )


    #
    # Same-TTI call is safe because Wave 8 explicitly
    # established idempotence.
    #
    measurement_0 = (
        measurement_provider(
            0
        )
    )

    expected_shape = (
        num_ues,
        context.num_cells,
    )

    if tuple(
        measurement_0.shape
    ) != expected_shape:
        raise RuntimeError(
            "Unexpected handover measurement shape."
        )

    selected_cell_tensor = torch.tensor(
        selected_cells,
        dtype=torch.long,
        device=measurement_0.device,
    )

    selected_power_0 = (
        measurement_0.index_select(
            dim=1,
            index=(
                selected_cell_tensor
            ),
        )
    )

    preferred_slot_0 = torch.argmax(
        selected_power_0,
        dim=1,
    )

    preferred_cell_0 = (
        selected_cell_tensor[
            preferred_slot_0
        ]
    )


    # ==========================================================
    # CONTROLLED INITIAL ASSOCIATION
    #
    # Start everybody on their measured preferred
    # selected cell EXCEPT one UE.
    #
    # For that UE:
    #
    #     serving cell
    #         =
    #     weaker selected cell
    #
    # while the handover controller sees the untouched
    # Sionna measurement.
    #
    # This guarantees a real-measurement-driven
    # reassociation integration test without pretending
    # that 1 ms of motion naturally crosses a cell.
    # ==========================================================

    preferred_counts = {
        cell_0: int(
            torch.count_nonzero(
                preferred_cell_0
                == cell_0
            ).item()
        ),

        cell_1: int(
            torch.count_nonzero(
                preferred_cell_0
                == cell_1
            ).item()
        ),
    }

    print(
        "Measured TTI-0 preferred-cell counts: "
        f"{preferred_counts}"
    )

    if (
        preferred_counts[cell_0] < 1
        or preferred_counts[cell_1] < 1
    ):
        raise RuntimeError(
            "This topology seed does not populate "
            "both selected cells under the measured "
            "two-cell association."
        )

    candidate_rows = []

    for row in range(
        num_ues
    ):

        destination_cell = int(
            preferred_cell_0[
                row
            ].item()
        )

        destination_slot = int(
            preferred_slot_0[
                row
            ].item()
        )

        source_slot = (
            1
            - destination_slot
        )

        source_cell = int(
            selected_cell_tensor[
                source_slot
            ].item()
        )

        #
        # Destination must not become empty when this
        # UE is temporarily moved out of it.
        #
        if (
            preferred_counts[
                destination_cell
            ]
            < 2
        ):
            continue

        destination_power = (
            selected_power_0[
                row,
                destination_slot,
            ]
        )

        source_power = (
            selected_power_0[
                row,
                source_slot,
            ]
        )

        margin_db = _db_ratio(
            destination_power,
            source_power,
        )

        if (
            margin_db
            >= MIN_CONTROLLED_MARGIN_DB
        ):
            candidate_rows.append(
                (
                    margin_db,
                    row,
                    source_cell,
                    destination_cell,
                )
            )


    if not candidate_rows:
        raise RuntimeError(
            "Could not find a UE with sufficient "
            "measured selected-cell margin for the "
            "controlled Wave-9 handover."
        )


    (
        forced_margin_db,
        forced_row,
        source_cell,
        destination_cell,
    ) = max(
        candidate_rows,
        key=lambda item: item[0],
    )

    forced_ue = int(
        cohort[
            forced_row
        ].item()
    )


    controlled_serving = (
        preferred_cell_0
        .detach()
        .clone()
    )

    controlled_serving[
        forced_row
    ] = source_cell


    controlled_counts = {
        cell_0: int(
            torch.count_nonzero(
                controlled_serving
                == cell_0
            ).item()
        ),

        cell_1: int(
            torch.count_nonzero(
                controlled_serving
                == cell_1
            ).item()
        ),
    }

    if any(
        count < 1
        for count
        in controlled_counts.values()
    ):
        raise RuntimeError(
            "Controlled serving setup produced "
            "an empty selected cell."
        )

    if any(
        count > NUM_CANDIDATES
        for count
        in controlled_counts.values()
    ):
        raise RuntimeError(
            "Controlled smoke exceeds the "
            "10-candidate architecture limit."
        )


    source_stream = (
        selected_cells.index(
            source_cell
        )
    )

    destination_stream = (
        selected_cells.index(
            destination_cell
        )
    )


    print(
        f"Controlled UE: {forced_ue}"
    )

    print(
        "Controlled initial serving cell: "
        f"{source_cell}"
    )

    print(
        "Measured preferred destination: "
        f"{destination_cell}"
    )

    print(
        "Measured TTI-0 margin: "
        f"{forced_margin_db:.3f} dB"
    )

    print(
        "Controlled starting populations: "
        f"{controlled_counts}"
    )


    # ==========================================================
    # GLOBAL PERSISTENT UE STATE
    #
    # Full-buffer traffic only for this integration
    # smoke. Packet continuity was already separately
    # validated by Wave 7.
    # ==========================================================

    full_buffer_mask = torch.ones(
        (
            num_ues,
        ),
        dtype=torch.bool,
        device=DEVICE,
    )

    buffer_bits = torch.full(
        (
            num_ues,
        ),
        fill_value=(
            FULL_BUFFER_STATE_BITS
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    average_throughput = torch.full(
        (
            num_ues,
        ),
        fill_value=(
            INITIAL_AVERAGE_THROUGHPUT_BPS
        ),
        dtype=torch.float32,
        device=DEVICE,
    )


    registry = (
        GlobalUESchedulerStateRegistry(
            global_ue_indices=(
                cohort
            ),

            serving_bs=(
                controlled_serving
            ),

            average_throughput_bps=(
                average_throughput
            ),

            buffer_bits=(
                buffer_bits
            ),

            full_buffer_mask=(
                full_buffer_mask
            ),
        )
    )


    ftp3_config = (
        build_training_ftp3_config(
            tti_duration_s=(
                TTI_DURATION_S
            )
        )
    )


    global_arrivals = (
        GlobalUETrafficArrivalProcess(
            global_ue_indices=(
                cohort
            ),

            full_buffer_mask=(
                full_buffer_mask
            ),

            ftp3_config=(
                ftp3_config
            ),

            seed=SEED,
        )
    )


    # ==========================================================
    # HANDOVER CONTROLLER
    #
    # 1 dB hysteresis
    # 2 consecutive TTIs
    #
    # Our controlled UE starts with >=6 dB measured
    # advantage toward the destination.
    # ==========================================================

    handover_controller = (
        HandoverController(
            initial_serving_bs=(
                controlled_serving
            ),

            num_bs=(
                context.num_cells
            ),

            config=(
                HandoverConfig(
                    hysteresis_db=1.0,

                    time_to_trigger_ttis=2,
                )
            ),
        )
    )


    coordinator = (
        DynamicHandoverCoordinator(
            registry=registry,

            handover_controller=(
                handover_controller
            ),

            selected_cell_indices=(
                selected_cells
            ),

            tds_config=(
                PFTimeDomainConfig(
                    num_candidates=(
                        NUM_CANDIDATES
                    )
                )
            ),

            throughput_forgetting_factor=(
                THROUGHPUT_FORGETTING_FACTOR
            ),

            ftp3_config=(
                ftp3_config
            ),

            full_buffer_state_bits=(
                FULL_BUFFER_STATE_BITS
            ),

            traffic_arrival_process=(
                global_arrivals
            ),

            local_traffic_seed_base=5000,
        )
    )


    # ==========================================================
    # ACTUAL HANDOVER MEASUREMENT INPUT
    #
    # For this 2-cell integration smoke only, mask
    # the other 19 BSs from handover candidacy.
    #
    # The power values for the two selected cells are
    # untouched Sionna-derived values.
    #
    # Final network-scale experiments can remove this
    # two-cell restriction.
    # ==========================================================

    def measured_selected_cell_link_power(
        tti_index: int,
    ) -> torch.Tensor:

        raw = (
            measurement_provider(
                tti_index
            )
        )

        masked = torch.full_like(
            raw,
            fill_value=(
                torch.finfo(
                    raw.dtype
                ).tiny
            ),
        )

        masked[
            :,
            selected_cell_tensor,
        ] = raw[
            :,
            selected_cell_tensor,
        ]


        forced_destination_power = (
            raw[
                forced_row,
                destination_cell,
            ]
        )

        forced_source_power = (
            raw[
                forced_row,
                source_cell,
            ]
        )

        margin_db = _db_ratio(
            forced_destination_power,
            forced_source_power,
        )

        print(
            f"TTI {tti_index}: measured UE "
            f"{forced_ue} destination/source "
            f"margin = {margin_db:.3f} dB"
        )

        return masked


    # ==========================================================
    # DYNAMIC REAL SIONNA SCHEDULING INPUT
    # ==========================================================

    radio_provider = (
        CellChunkedSionnaPPOInputProvider(
            context=context,

            global_ue_indices_provider=(
                coordinator
                .membership_provider
            ),
        )
    )


    # ==========================================================
    # FROZEN PRETRAINED PPO
    # ==========================================================

    state_config = (
        OneLDSStateConfig(
            throughput_normalization_bps=(
                100.0e6
            ),

            buffer_normalization=(
                FULL_BUFFER_STATE_BITS
            ),

            subband_cqi_normalization=15.0,

            num_candidates=(
                NUM_CANDIDATES
            ),

            num_rbgs=(
                NUM_RBGS
            ),

            max_rank=2,
        )
    )


    actor = (
        OneLDSPPOActor(
            OneLDSPPOActorConfig(
                state_size=(
                    state_config.state_size
                ),

                hidden_size=32,

                num_rbgs=(
                    NUM_RBGS
                ),

                num_actions_per_rbg=(
                    NUM_CANDIDATES
                    + 1
                ),
            )
        )
        .to(
            DEVICE
        )
    )


    critic = (
        OneLDSPPOCritic(
            OneLDSPPOCriticConfig(
                state_size=(
                    state_config.state_size
                ),

                hidden_size=32,
            )
        )
        .to(
            DEVICE
        )
    )


    checkpoint = (
        load_ppo_model_checkpoint(
            path=(
                CHECKPOINT_PATH
            ),

            actor=actor,

            map_location=(
                DEVICE
            ),
        )
    )


    actor.eval()

    critic.eval()

    for parameter in actor.parameters():
        parameter.requires_grad_(
            False
        )

    for parameter in critic.parameters():
        parameter.requires_grad_(
            False
        )


    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        "Checkpoint TTI: "
        f"{checkpoint.get('tti_index')}"
    )


    transition_buffer = (
        PPOMultiStreamTransitionBuffer(
            config=(
                PPOMultiStreamBufferConfig(
                    num_streams=(
                        NUM_STREAMS
                    ),

                    update_size=128,
                )
            )
        )
    )


    rollout_controllers = tuple(
        OneLDSPPOMultiCellRolloutController(
            stream_id=stream_index,

            actor=actor,

            critic=critic,

            transition_buffer=(
                transition_buffer
            ),

            config=(
                OneLDSPPOMultiCellRolloutConfig(
                    num_user_slots=(
                        NUM_USER_SLOTS
                    )
                )
            ),
        )

        for stream_index
        in range(
            NUM_STREAMS
        )
    )


    # ==========================================================
    # INTEGRATION ASSERTIONS
    # ==========================================================

    saw_forced_handover = False

    saw_source_without_forced = False

    saw_destination_with_forced = False

    saw_destination_candidate = False


    def transition_observer(
        tti_index,
        transition,
    ) -> None:

        nonlocal saw_forced_handover
        nonlocal saw_source_without_forced
        nonlocal saw_destination_with_forced

        moved = (
            transition
            .moved_global_ue_indices
        )

        forced_moved = bool(
            torch.any(
                moved
                == forced_ue
            ).item()
        )

        if tti_index == 0 and forced_moved:
            raise RuntimeError(
                "TTT=2 handover occurred too early."
            )

        if not forced_moved:
            return

        if tti_index != 1:
            raise RuntimeError(
                "Controlled UE handover occurred "
                "at unexpected TTI."
            )

        saw_forced_handover = True

        source_state = (
            transition
            .local_states[
                source_stream
            ]
        )

        destination_state = (
            transition
            .local_states[
                destination_stream
            ]
        )

        saw_source_without_forced = (
            not bool(
                torch.any(
                    source_state
                    .global_ue_indices
                    == forced_ue
                ).item()
            )
        )

        saw_destination_with_forced = (
            bool(
                torch.any(
                    destination_state
                    .global_ue_indices
                    == forced_ue
                ).item()
            )
        )

        print(
            "REAL-MEASUREMENT HANDOVER: "
            f"TTI={tti_index}, "
            f"UE={forced_ue}, "
            f"{source_cell}->{destination_cell}"
        )


    def observer(
        tti_index,
        stream_index,
        result,
        transition,
    ) -> None:

        del transition

        nonlocal saw_destination_candidate

        real_cell = (
            selected_cells[
                stream_index
            ]
        )

        serving_ids = (
            result
            .scheduler_observation
            .serving_global_ue_indices
        )

        candidate_ids = (
            result
            .prepared
            .candidate_global_ue_indices[
                result
                .prepared
                .candidate_valid_mask
            ]
        )

        if (
            tti_index >= 1
            and real_cell
            == destination_cell
        ):
            if bool(
                torch.any(
                    candidate_ids
                    == forced_ue
                ).item()
            ):
                saw_destination_candidate = True

        delivered_mbps = float(
            result
            .traffic_service
            .delivered_rate_bps
            .sum()
            .item()
            / 1.0e6
        )

        print(
            f"TTI {tti_index}, "
            f"stream {stream_index}, "
            f"cell {real_cell}: "
            f"UEs={int(serving_ids.numel())}, "
            f"candidates={int(candidate_ids.numel())}, "
            f"delivered={delivered_mbps:.3f} Mbps"
        )


    # ==========================================================
    # RUN
    # ==========================================================

    result = (
        run_dynamic_multicell_ppo_evaluation(
            start_tti_index=0,

            num_ttis=(
                NUM_TTIS
            ),

            radio_input_provider=(
                radio_provider
            ),

            #
            # THIS is the Wave-9 integration.
            #
            handover_link_power_provider=(
                measured_selected_cell_link_power
            ),

            coordinator=(
                coordinator
            ),

            rollout_controllers=(
                rollout_controllers
            ),

            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode=(
                        "data_available_only"
                    )
                )
            ),

            state_config=(
                state_config
            ),

            greedy_config=(
                PPOGreedySearchConfig()
            ),

            reward_config=(
                PPORewardConfig(
                    geometric_mean_normalizer_bps=(
                        100.0e6
                    )
                )
            ),

            reward_population=(
                "candidates"
            ),

            transition_observer=(
                transition_observer
            ),

            observer=observer,

            device=DEVICE,
        )
    )


    # ==========================================================
    # FINAL SCIENTIFIC CHECKS
    # ==========================================================

    final_state = (
        result.final_global_state
    )

    matches = torch.nonzero(
        final_state
        .global_ue_indices
        == forced_ue,
        as_tuple=False,
    ).flatten()

    if int(
        matches.numel()
    ) != 1:
        raise RuntimeError(
            "Final forced-UE lookup failed."
        )

    final_row = int(
        matches[
            0
        ].item()
    )

    final_cell = int(
        final_state
        .serving_bs[
            final_row
        ].item()
    )


    if final_cell != destination_cell:
        raise RuntimeError(
            "Controlled UE did not finish on "
            "its Sionna-measured preferred cell."
        )

    if not saw_forced_handover:
        raise RuntimeError(
            "Real measurement did not trigger "
            "the controlled handover."
        )

    if not saw_source_without_forced:
        raise RuntimeError(
            "Source cell retained the handed-over UE."
        )

    if not saw_destination_with_forced:
        raise RuntimeError(
            "Destination cell did not receive "
            "the handed-over UE."
        )

    if not saw_destination_candidate:
        raise RuntimeError(
            "Handed-over UE never appeared in "
            "destination PF candidates."
        )

    if len(
        transition_buffer
    ) != 0:
        raise RuntimeError(
            "Evaluation unexpectedly collected "
            "PPO transitions."
        )

    if any(
        controller
        .has_unresolved_tti_boundary
        for controller
        in rollout_controllers
    ):
        raise RuntimeError(
            "Evaluation left unresolved PPO "
            "trajectory state."
        )


    torch.cuda.synchronize(
        DEVICE
    )

    peak_mib = (
        torch.cuda.max_memory_allocated(
            DEVICE
        )
        / (
            1024.0
            ** 2
        )
    )


    print()
    print("=" * 78)

    print(
        "Measured controlled UE: "
        f"{forced_ue}"
    )

    print(
        "Controlled route: "
        f"{source_cell} -> {destination_cell}"
    )

    print(
        "Initial measured margin: "
        f"{forced_margin_db:.3f} dB"
    )

    print(
        "Completed handovers: "
        f"{result.num_completed_handovers}"
    )

    print(
        "Forced UE final cell: "
        f"{final_cell}"
    )

    print(
        "Destination PF candidate observed: "
        f"{saw_destination_candidate}"
    )

    print(
        "PPO transitions collected: "
        f"{len(transition_buffer)}"
    )

    print(
        "Peak allocated GPU: "
        f"{peak_mib:.1f} MiB"
    )

    print("=" * 78)

    print()
    print(
        "WAVE9_MEASURED_DYNAMIC_HANDOVER_SMOKE_PASS"
    )


if __name__ == "__main__":
    main()
