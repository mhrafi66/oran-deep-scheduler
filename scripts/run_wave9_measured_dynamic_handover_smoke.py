from __future__ import annotations

import math
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

SPEED_KMH = 30.0

HANDOVER_HYSTERESIS_DB = 1.0

HANDOVER_TTT_TTIS = 2

#
# We deliberately require a strong REAL measured
# advantage before constructing the controlled
# initial serving mismatch.
#
MIN_MEASURED_ADVANTAGE_DB = 6.0

INITIAL_AVERAGE_THROUGHPUT_BPS = (
    1.0e6
)

THROUGHPUT_FORGETTING_FACTOR = (
    0.9
)

FULL_BUFFER_STATE_BITS = (
    12_000.0
)

CHECKPOINT_PATH = Path(
    "~/oran-deep-scheduler/"
    "experiments/checkpoints/"
    "paper_1lds_ppo_500tti.pt"
).expanduser()


def build_global_state(
    *,
    context,
    ftp3_config,
):
    """
    Build the closed two-cell UE cohort used by the
    final measured-handover integration smoke.
    """

    global_ids = torch.cat(
        context.global_ue_indices_by_stream,
        dim=0,
    )

    serving_parts = []

    for (
        cell_index,
        ue_ids,
    ) in zip(
        context.selected_cell_indices,
        context.global_ue_indices_by_stream,
        strict=True,
    ):

        serving_parts.append(
            torch.full(
                ue_ids.shape,
                fill_value=(
                    int(cell_index)
                ),
                dtype=torch.long,
                device=DEVICE,
            )
        )

    serving_bs = torch.cat(
        serving_parts,
        dim=0,
    )

    num_ues = int(
        global_ids.numel()
    )

    #
    # Deterministic mixed FB / FTP population.
    #
    full_buffer_mask = (
        torch.arange(
            num_ues,
            device=DEVICE,
        )
        % 2
        == 0
    )

    ftp_bits = float(
        ftp3_config.packet_size_bits
    )

    buffer_bits = torch.where(
        full_buffer_mask,

        torch.full(
            (
                num_ues,
            ),
            fill_value=(
                FULL_BUFFER_STATE_BITS
            ),
            dtype=torch.float32,
            device=DEVICE,
        ),

        torch.full(
            (
                num_ues,
            ),
            fill_value=ftp_bits,
            dtype=torch.float32,
            device=DEVICE,
        ),
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
                global_ids
            ),

            serving_bs=(
                serving_bs
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

    return (
        registry,
        global_ids,
        serving_bs,
        full_buffer_mask,
    )


def select_controlled_mismatch(
    *,
    measurement: torch.Tensor,
    cohort_global_ids: torch.Tensor,
    serving_bs: torch.Tensor,
    selected_cell_indices: tuple[int, ...],
):
    """
    Find a UE whose REAL Sionna-derived handover
    measurement strongly prefers its ordinary
    serving cell over the OTHER selected cell.

    We then intentionally initialize that UE on the
    weaker selected cell.

    Therefore the subsequent handover itself is
    driven by REAL Sionna measurement values rather
    than a fabricated link-power trigger.

    This mismatch is a controlled integration-test
    initial condition, not a claim about naturally
    occurring handover frequency.
    """

    if len(
        selected_cell_indices
    ) != 2:
        raise ValueError(
            "Final integration smoke requires "
            "exactly two selected cells."
        )

    cell_a = int(
        selected_cell_indices[
            0
        ]
    )

    cell_b = int(
        selected_cell_indices[
            1
        ]
    )

    candidates = []

    for row in range(
        int(
            cohort_global_ids.numel()
        )
    ):

        destination = int(
            serving_bs[
                row
            ].item()
        )

        if destination not in {
            cell_a,
            cell_b,
        }:
            continue

        source = (
            cell_b
            if destination == cell_a
            else cell_a
        )

        destination_power = float(
            measurement[
                row,
                destination,
            ].item()
        )

        source_power = float(
            measurement[
                row,
                source,
            ].item()
        )

        if (
            destination_power <= 0.0
            or source_power <= 0.0
        ):
            continue

        advantage_db = (
            10.0
            * math.log10(
                destination_power
                / source_power
            )
        )

        candidates.append(
            (
                advantage_db,
                row,
                source,
                destination,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No candidate UE found for "
            "controlled measured handover."
        )

    candidates.sort(
        reverse=True,
        key=lambda item: item[0],
    )

    (
        advantage_db,
        row,
        source,
        destination,
    ) = candidates[
        0
    ]

    if (
        advantage_db
        < MIN_MEASURED_ADVANTAGE_DB
    ):
        raise RuntimeError(
            "No UE has a sufficiently strong "
            "real measured destination advantage. "
            f"Best was {advantage_db:.3f} dB; "
            f"required "
            f"{MIN_MEASURED_ADVANTAGE_DB:.3f} dB."
        )

    global_ue = int(
        cohort_global_ids[
            row
        ].item()
    )

    return (
        row,
        global_ue,
        source,
        destination,
        advantage_db,
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
        "WAVE-9 SIONNA-MEASURED "
        "DYNAMIC PPO HANDOVER"
    )

    print("=" * 78)

    print(
        "Measurement:"
    )

    print(
        "  moving geometry"
    )

    print(
        "  -> lightweight Sionna channel"
    )

    print(
        "  -> channel-averaged pathloss"
    )

    print(
        "  -> inverse pathloss link strength"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is OPEN-REPRODUCTION handover "
        "measurement."
    )

    print(
        "It is NOT claimed to implement "
        "standardized 3GPP A3/RSRP filtering."
    )

    print()


    # ==========================================================
    # REAL TEMPORAL SCHEDULING CONTEXT
    # ==========================================================

    context = (
        build_chunked_sionna_ppo_context(
            config=(
                ChunkedSionnaPPOConfig(
                    num_training_cells=(
                        NUM_STREAMS
                    ),

                    #
                    # Capability/integration scale.
                    #
                    num_ut_per_sector=2,

                    num_rbs=18,

                    num_rbgs=18,

                    subcarriers_per_rb=12,

                    #
                    # Required for persistent UE-owned
                    # channel RNG across reassociation.
                    #
                    ue_microbatch_size=1,

                    identity_stable_ue_channel_rng=(
                        True
                    ),

                    topology_seed=42,

                    association_channel_seed=1000,

                    mimo_channel_seed=2000,

                    device="cuda:0",

                    ut_speed_kmh=(
                        SPEED_KMH
                    ),

                    temporal_radio_mode=(
                        "velocity_window"
                    ),

                    temporal_window_ttis=(
                        NUM_TTIS
                    ),

                    tti_duration_s=(
                        TTI_DURATION_S
                    ),

                    temporal_max_displacement_m=(
                        20.0
                    ),
                )
            )
        )
    )

    selected_cells = tuple(
        int(cell)
        for cell
        in context.selected_cell_indices
    )

    if len(
        selected_cells
    ) != NUM_STREAMS:
        raise RuntimeError(
            "Unexpected selected-cell count."
        )

    cohort = torch.cat(
        context.global_ue_indices_by_stream,
        dim=0,
    )

    print(
        "Selected scheduling cells: "
        f"{selected_cells}"
    )

    print(
        "Closed cohort size: "
        f"{int(cohort.numel())}"
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


    # ==========================================================
    # GLOBAL TRAFFIC / PF STATE
    # ==========================================================

    ftp3_config = (
        build_training_ftp3_config(
            tti_duration_s=(
                TTI_DURATION_S
            )
        )
    )

    (
        registry,
        cohort_global_ids,
        original_serving_bs,
        full_buffer_mask,
    ) = build_global_state(
        context=context,

        ftp3_config=(
            ftp3_config
        ),
    )

    if not torch.equal(
        cohort,
        cohort_global_ids,
    ):
        raise RuntimeError(
            "Measurement cohort and global "
            "scheduler registry are misaligned."
        )


    # ==========================================================
    # FIND A REAL MEASURED LINK ADVANTAGE
    # ==========================================================

    measurement_0 = (
        measurement_provider(
            0
        )
    )

    (
        forced_row,
        forced_ue,
        source_cell,
        destination_cell,
        initial_advantage_db,
    ) = select_controlled_mismatch(
        measurement=(
            measurement_0
        ),

        cohort_global_ids=(
            cohort_global_ids
        ),

        serving_bs=(
            original_serving_bs
        ),

        selected_cell_indices=(
            selected_cells
        ),
    )

    print()

    print(
        "Controlled UE: "
        f"{forced_ue}"
    )

    print(
        "Controlled initial mismatch: "
        f"{source_cell} -> "
        f"{destination_cell}"
    )

    print(
        "Real measured destination advantage: "
        f"{initial_advantage_db:.3f} dB"
    )


    # ==========================================================
    # CREATE ONLY THE INITIAL MISMATCH
    # ==========================================================

    mismatched_serving_bs = (
        original_serving_bs
        .detach()
        .clone()
    )

    mismatched_serving_bs[
        forced_row
    ] = (
        source_cell
    )

    mismatch_mask = torch.zeros_like(
        mismatched_serving_bs,
        dtype=torch.bool,
    )

    mismatch_mask[
        forced_row
    ] = True

    registry.apply_serving_bs_update(
        new_serving_bs=(
            mismatched_serving_bs
        ),

        handover_mask=(
            mismatch_mask
        ),
    )

    initial_dynamic_state = (
        registry.snapshot()
    )

    #
    # Dynamic local-state bridge currently requires
    # every selected cell to remain non-empty.
    #
    for cell in selected_cells:

        count = int(
            torch.count_nonzero(
                initial_dynamic_state
                .serving_bs
                == cell
            ).item()
        )

        if count < 1:
            raise RuntimeError(
                "Controlled mismatch produced "
                f"empty selected cell {cell}."
            )

        if count > NUM_CANDIDATES:
            raise RuntimeError(
                "Integration smoke population "
                "exceeds fixed PPO candidate width."
            )


    # ==========================================================
    # HYSTERESIS + TTT HANDOVER STATE MACHINE
    # ==========================================================

    handover_controller = (
        HandoverController(
            initial_serving_bs=(
                initial_dynamic_state
                .serving_bs
            ),

            num_bs=(
                context.num_cells
            ),

            config=(
                HandoverConfig(
                    hysteresis_db=(
                        HANDOVER_HYSTERESIS_DB
                    ),

                    time_to_trigger_ttis=(
                        HANDOVER_TTT_TTIS
                    ),
                )
            ),
        )
    )


    # ==========================================================
    # GLOBAL IDENTITY-OWNED ARRIVAL PROCESS
    # ==========================================================

    global_arrivals = (
        GlobalUETrafficArrivalProcess(
            global_ue_indices=(
                cohort_global_ids
            ),

            full_buffer_mask=(
                full_buffer_mask
            ),

            ftp3_config=(
                ftp3_config
            ),

            seed=(
                SEED
            ),
        )
    )


    # ==========================================================
    # DYNAMIC GLOBAL -> CELL-LOCAL STATE BRIDGE
    # ==========================================================

    coordinator = (
        DynamicHandoverCoordinator(
            registry=(
                registry
            ),

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

            local_traffic_seed_base=(
                5000
            ),
        )
    )


    # ==========================================================
    # DYNAMIC REAL SIONNA SCHEDULING INPUT PROVIDER
    # ==========================================================

    radio_provider = (
        CellChunkedSionnaPPOInputProvider(
            context=(
                context
            ),

            global_ue_indices_provider=(
                coordinator
                .membership_provider
            ),
        )
    )


    # ==========================================================
    # FINAL HANDOVER MEASUREMENT WRAPPER
    #
    # The measurement itself is REAL Sionna-derived.
    #
    # For THIS TWO-CELL integration smoke only,
    # non-selected BSs are excluded from handover
    # candidacy. Otherwise a UE could correctly move
    # to a third cell whose scheduler we are not
    # executing in this two-stream smoke.
    #
    # Publication/full-network experiments must not
    # describe this as unrestricted 21-cell HO.
    # ==========================================================

    selected_cell_mask = torch.zeros(
        context.num_cells,
        dtype=torch.bool,
        device=DEVICE,
    )

    for cell in selected_cells:
        selected_cell_mask[
            cell
        ] = True


    def handover_link_power_provider(
        tti_index: int,
    ) -> torch.Tensor:

        measured = (
            measurement_provider(
                tti_index
            )
            .detach()
            .clone()
        )

        if tuple(
            measured.shape
        ) != (
            int(
                cohort_global_ids
                .numel()
            ),
            context.num_cells,
        ):
            raise RuntimeError(
                "Measured handover tensor has "
                "unexpected shape."
            )

        #
        # Restrict only the DECISION SET.
        #
        # The values for the two selected physical
        # cells remain untouched real measurements.
        #
        measured[
            :,
            ~selected_cell_mask,
        ] = torch.finfo(
            measured.dtype
        ).tiny

        destination_power = float(
            measured[
                forced_row,
                destination_cell,
            ].item()
        )

        source_power = float(
            measured[
                forced_row,
                source_cell,
            ].item()
        )

        advantage_db = (
            10.0
            * math.log10(
                destination_power
                / source_power
            )
        )

        print(
            f"TTI {tti_index}: "
            f"forced UE measured "
            f"destination advantage = "
            f"{advantage_db:.3f} dB"
        )

        return measured


    # ==========================================================
    # FROZEN PPO POLICY
    # ==========================================================

    state_config = OneLDSStateConfig(
        throughput_normalization_bps=(
            100.0e6
        ),

        buffer_normalization=(
            FULL_BUFFER_STATE_BITS
        ),

        subband_cqi_normalization=(
            15.0
        ),

        num_candidates=(
            NUM_CANDIDATES
        ),

        num_rbgs=(
            NUM_RBGS
        ),

        max_rank=2,
    )

    actor = (
        OneLDSPPOActor(
            OneLDSPPOActorConfig(
                state_size=(
                    state_config
                    .state_size
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
                    state_config
                    .state_size
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

            actor=(
                actor
            ),

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
        "Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        "Checkpoint TTI: "
        f"{checkpoint.get('tti_index')}"
    )


    # ==========================================================
    # EVALUATION-ONLY PPO CONTROLLERS
    # ==========================================================

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
            stream_id=(
                stream_index
            ),

            actor=(
                actor
            ),

            critic=(
                critic
            ),

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
    # OBSERVATION / PASS CONDITIONS
    # ==========================================================

    saw_forced_handover = False

    saw_source_without_ue = False

    saw_destination_with_ue = False

    forced_handover_tti = None


    def observer(
        tti_index,
        stream_index,
        result,
        transition,
    ) -> None:

        nonlocal saw_forced_handover
        nonlocal saw_source_without_ue
        nonlocal saw_destination_with_ue
        nonlocal forced_handover_tti

        real_cell = int(
            selected_cells[
                stream_index
            ]
        )

        serving_ids = (
            result
            .scheduler_observation
            .serving_global_ue_indices
        )

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

        if forced_moved:

            saw_forced_handover = (
                True
            )

            forced_handover_tti = (
                tti_index
            )

            print(
                "FORCED UE HANDOVER OBSERVED: "
                f"UE={forced_ue}, "
                f"TTI={tti_index}, "
                f"{source_cell}"
                f" -> "
                f"{destination_cell}"
            )

        current_forced_cell = int(
            transition
            .handover_result
            .serving_bs[
                forced_row
            ].item()
        )

        if (
            current_forced_cell
            == destination_cell
        ):

            if (
                real_cell
                == source_cell
                and not bool(
                    torch.any(
                        serving_ids
                        == forced_ue
                    ).item()
                )
            ):
                saw_source_without_ue = (
                    True
                )

            if (
                real_cell
                == destination_cell
                and bool(
                    torch.any(
                        serving_ids
                        == forced_ue
                    ).item()
                )
            ):
                saw_destination_with_ue = (
                    True
                )

        delivered_mbps = float(
            result
            .traffic_service
            .delivered_rate_bps
            .sum()
            .item()
            / 1.0e6
        )

        print(
            f"TTI {tti_index} "
            f"stream {stream_index} "
            f"cell {real_cell}: "
            f"serving_ues="
            f"{int(serving_ids.numel())}, "
            f"delivered="
            f"{delivered_mbps:.3f} Mbps"
        )


    # ==========================================================
    # RUN REAL MEASUREMENT -> HO -> SIONNA -> PPO
    # ==========================================================

    run_result = (
        run_dynamic_multicell_ppo_evaluation(
            start_tti_index=0,

            num_ttis=(
                NUM_TTIS
            ),

            radio_input_provider=(
                radio_provider
            ),

            handover_link_power_provider=(
                handover_link_power_provider
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

            observer=(
                observer
            ),

            device=(
                DEVICE
            ),
        )
    )


    # ==========================================================
    # FINAL SCIENTIFIC CONSISTENCY CHECKS
    # ==========================================================

    final_state = (
        run_result
        .final_global_state
    )

    final_matches = torch.nonzero(
        final_state
        .global_ue_indices
        == forced_ue,
        as_tuple=False,
    ).flatten()

    if int(
        final_matches.numel()
    ) != 1:
        raise RuntimeError(
            "Final controlled UE lookup failed."
        )

    final_row = int(
        final_matches[
            0
        ].item()
    )

    final_cell = int(
        final_state
        .serving_bs[
            final_row
        ].item()
    )

    if not saw_forced_handover:
        raise RuntimeError(
            "Real Sionna measurement never "
            "triggered the controlled UE handover."
        )

    if (
        final_cell
        != destination_cell
    ):
        raise RuntimeError(
            "Controlled UE did not finish at "
            "the real measured destination cell."
        )

    if not saw_source_without_ue:
        raise RuntimeError(
            "Source scheduler population did not "
            "lose the handed-over UE."
        )

    if not saw_destination_with_ue:
        raise RuntimeError(
            "Destination scheduler population did "
            "not gain the handed-over UE."
        )

    if len(
        transition_buffer
    ) != 0:
        raise RuntimeError(
            "Evaluation-only measured handover "
            "unexpectedly collected PPO samples."
        )

    if any(
        controller
        .has_unresolved_tti_boundary
        for controller
        in rollout_controllers
    ):
        raise RuntimeError(
            "Evaluation-only measured handover "
            "left PPO temporal boundaries."
        )


    # ==========================================================
    # SUMMARY
    # ==========================================================

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
        "Controlled UE: "
        f"{forced_ue}"
    )

    print(
        "Initial mismatched cell: "
        f"{source_cell}"
    )

    print(
        "Real measured destination: "
        f"{destination_cell}"
    )

    print(
        "Initial measured advantage: "
        f"{initial_advantage_db:.3f} dB"
    )

    print(
        "Observed handover TTI: "
        f"{forced_handover_tti}"
    )

    print(
        "Completed handovers total: "
        f"{run_result.num_completed_handovers}"
    )

    print(
        "Source lost UE: "
        f"{saw_source_without_ue}"
    )

    print(
        "Destination gained UE: "
        f"{saw_destination_with_ue}"
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
