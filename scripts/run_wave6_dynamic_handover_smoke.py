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

NUM_TTIS = 4

NUM_STREAMS = 2

NUM_CANDIDATES = 10

NUM_RBGS = 18

NUM_USER_SLOTS = 4

INITIAL_AVERAGE_THROUGHPUT_BPS = (
    1.0e6
)

THROUGHPUT_FORGETTING_FACTOR = (
    0.9
)

FULL_BUFFER_STATE_BITS = (
    12_000.0
)

TTI_DURATION_S = (
    0.5e-3
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
    Build one CLOSED two-cell UE cohort.

    Only UEs initially associated with the two
    selected smoke cells participate in this first
    end-to-end dynamic-association experiment.
    """

    global_ids = torch.cat(
        context
        .global_ue_indices_by_stream,
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
                fill_value=cell_index,
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
    # Deterministic 50/50-ish mixed traffic
    # assignment in persistent global cohort order.
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

    registry = GlobalUESchedulerStateRegistry(
        global_ue_indices=global_ids,

        serving_bs=serving_bs,

        average_throughput_bps=(
            average_throughput
        ),

        buffer_bits=buffer_bits,

        full_buffer_mask=(
            full_buffer_mask
        ),
    )

    return (
        registry,
        global_ids,
        serving_bs,
        full_buffer_mask,
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
        "WAVE-6 REAL SIONNA DYNAMIC HANDOVER SMOKE"
    )
    print("=" * 78)

    print(
        f"GPU: {torch.cuda.get_device_name(DEVICE)}"
    )

    print(
        "Mode: evaluation-only frozen PPO"
    )

    print(
        "Handover trigger: synthetic controlled "
        "measurement"
    )

    print(
        "Scheduling PHY: real chunked Sionna"
    )

    print(
        "Traffic continuity: global aggregate state"
    )

    print(
        "Packet FIFO handover: NOT YET integrated"
    )

    print()


    # ==========================================================
    # MEMORY-SAFE TEMPORAL SIONNA CONTEXT
    # ==========================================================

    context = (
        build_chunked_sionna_ppo_context(
            config=(
                ChunkedSionnaPPOConfig(
                    num_training_cells=(
                        NUM_STREAMS
                    ),

                    #
                    # CAPABILITY-SMOKE SCALE ONLY.
                    #
                    num_ut_per_sector=2,

                    num_rbs=18,

                    num_rbgs=18,

                    subcarriers_per_rb=12,

                    #
                    # Dynamic handover mode requires
                    # one persistent UE identity per
                    # radio microbatch.
                    #
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

                    temporal_max_displacement_m=(
                        20.0
                    ),
                )
            )
        )
    )

    print(
        "Selected physical cells: "
        f"{context.selected_cell_indices}"
    )

    print(
        "Initial UE counts: "
        f"{tuple(int(x.numel()) for x in context.global_ue_indices_by_stream)}"
    )

    #
    # We want this tiny smoke population to remain
    # entirely inside the <=10 PF candidate set.
    #
    if any(
        int(ids.numel()) > NUM_CANDIDATES - 1
        for ids in (
            context.global_ue_indices_by_stream
        )
    ):
        raise RuntimeError(
            "Smoke population is too large to "
            "guarantee the handed-over UE remains "
            "inside the 10-candidate PF set."
        )


    # ==========================================================
    # GLOBAL TRAFFIC / PF / ASSOCIATION STATE
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
        initial_serving_bs,
        full_buffer_mask,
    ) = build_global_state(
        context=context,
        ftp3_config=ftp3_config,
    )

    print(
        f"Closed cohort UEs: {int(cohort_global_ids.numel())}"
    )


    # ==========================================================
    # FORCE ONE UE FROM STREAM 0 -> STREAM 1
    # ==========================================================

    source_cell = (
        context
        .selected_cell_indices[
            0
        ]
    )

    destination_cell = (
        context
        .selected_cell_indices[
            1
        ]
    )

    source_ues = (
        context
        .global_ue_indices_by_stream[
            0
        ]
    )

    if int(
        source_ues.numel()
    ) < 2:
        raise RuntimeError(
            "Source smoke cell must contain at "
            "least two UEs."
        )

    forced_ue = int(
        source_ues[
            0
        ].item()
    )

    forced_row_matches = torch.nonzero(
        cohort_global_ids
        == forced_ue,
        as_tuple=False,
    ).flatten()

    if int(
        forced_row_matches.numel()
    ) != 1:
        raise RuntimeError(
            "Forced UE identity lookup failed."
        )

    forced_row = int(
        forced_row_matches[
            0
        ].item()
    )

    print(
        "Forced handover UE: "
        f"{forced_ue}"
    )

    print(
        "Controlled route: "
        f"{source_cell} -> {destination_cell}"
    )


    # ==========================================================
    # HANDOVER CONTROLLER
    #
    # t=1 target qualifies once.
    # t=2 target qualifies again -> handover.
    # ==========================================================

    handover_controller = (
        HandoverController(
            initial_serving_bs=(
                initial_serving_bs
            ),

            num_bs=(
                context.num_cells
            ),

            config=(
                HandoverConfig(
                    hysteresis_db=3.0,

                    time_to_trigger_ttis=2,
                )
            ),
        )
    )


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

            seed=SEED,
        )
    )


    coordinator = (
        DynamicHandoverCoordinator(
            registry=registry,

            handover_controller=(
                handover_controller
            ),

            selected_cell_indices=(
                context
                .selected_cell_indices
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

            ftp3_config=ftp3_config,

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
    # DYNAMIC REAL SIONNA PROVIDER
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
    # FROZEN PAPER-SHAPED PPO ACTOR
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

    actor = OneLDSPPOActor(
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
    ).to(
        DEVICE
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=(
                state_config.state_size
            ),

            hidden_size=32,
        )
    ).to(
        DEVICE
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

    for parameter in actor.parameters():
        parameter.requires_grad_(
            False
        )

    critic.eval()

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
    # EVALUATION-ONLY ROLLOUT CONTROLLERS
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
    # CONTROLLED HANDOVER MEASUREMENT
    #
    # This is intentionally NOT the Sionna-derived
    # handover trigger yet.
    #
    # All non-target UEs strongly prefer their
    # initial cell.
    #
    # Forced UE:
    #
    #   TTI 0  -> source stronger
    #   TTI 1  -> destination stronger, TTT=1
    #   TTI 2  -> destination stronger, HO
    #   TTI 3  -> destination remains stronger
    # ==========================================================

    def handover_link_power_provider(
        tti_index: int,
    ) -> torch.Tensor:

        link_power = torch.ones(
            (
                int(
                    cohort_global_ids
                    .numel()
                ),
                context.num_cells,
            ),
            dtype=torch.float32,
            device=DEVICE,
        )

        rows = torch.arange(
            int(
                cohort_global_ids
                .numel()
            ),
            device=DEVICE,
        )

        link_power[
            rows,
            initial_serving_bs,
        ] = 100.0

        if tti_index >= 1:

            link_power[
                forced_row,
                source_cell,
            ] = 1.0

            link_power[
                forced_row,
                destination_cell,
            ] = 100.0

        return link_power


    saw_handover = False

    saw_source_without_ue = False

    saw_destination_with_ue = False

    saw_destination_candidate = False


    def observer(
        tti_index,
        stream_index,
        result,
        transition,
    ) -> None:

        nonlocal saw_handover

        nonlocal saw_source_without_ue

        nonlocal saw_destination_with_ue

        nonlocal saw_destination_candidate

        real_cell = (
            context
            .selected_cell_indices[
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

        allocated_candidate = (
            result
            .schedule
            .allocation
            .candidate_by_user_slot
            .reshape(
                -1
            )
        )

        valid_alloc = (
            allocated_candidate
            >= 0
        )

        allocated_candidate = (
            allocated_candidate[
                valid_alloc
            ]
        )

        if int(
            allocated_candidate.numel()
        ) > 0:

            scheduled_ids = (
                result
                .prepared
                .candidate_global_ue_indices[
                    allocated_candidate
                ]
            )

        else:

            scheduled_ids = torch.empty(
                (
                    0,
                ),
                dtype=torch.long,
                device=DEVICE,
            )

        moved = (
            transition
            .moved_global_ue_indices
        )

        if bool(
            torch.any(
                moved
                == forced_ue
            ).item()
        ):
            saw_handover = True

        if tti_index >= 2:

            if (
                real_cell
                == source_cell
            ):

                if not bool(
                    torch.any(
                        serving_ids
                        == forced_ue
                    ).item()
                ):
                    saw_source_without_ue = (
                        True
                    )

            if (
                real_cell
                == destination_cell
            ):

                if bool(
                    torch.any(
                        serving_ids
                        == forced_ue
                    ).item()
                ):
                    saw_destination_with_ue = (
                        True
                    )

                if bool(
                    torch.any(
                        candidate_ids
                        == forced_ue
                    ).item()
                ):
                    saw_destination_candidate = (
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

        scheduled_forced = bool(
            torch.any(
                scheduled_ids
                == forced_ue
            ).item()
        )

        print(
            f"TTI {tti_index} "
            f"stream {stream_index} "
            f"cell {real_cell}: "
            f"UEs={int(serving_ids.numel())}, "
            f"candidates={int(candidate_ids.numel())}, "
            f"delivered={delivered_mbps:.3f} Mbps, "
            f"forced_ue_scheduled={scheduled_forced}"
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

            observer=observer,

            device=DEVICE,
        )
    )


    # ==========================================================
    # SCIENTIFIC INTEGRITY CHECKS
    # ==========================================================

    final_state = (
        result.final_global_state
    )

    final_forced_row = torch.nonzero(
        final_state.global_ue_indices
        == forced_ue,
        as_tuple=False,
    ).flatten()

    if int(
        final_forced_row.numel()
    ) != 1:
        raise RuntimeError(
            "Final forced-UE lookup failed."
        )

    final_forced_row = int(
        final_forced_row[
            0
        ].item()
    )

    if (
        int(
            final_state
            .serving_bs[
                final_forced_row
            ].item()
        )
        != destination_cell
    ):
        raise RuntimeError(
            "Forced UE did not finish at "
            "destination cell."
        )

    if (
        result.num_completed_handovers
        != 1
    ):
        raise RuntimeError(
            "Expected exactly one completed "
            "handover."
        )

    if not saw_handover:
        raise RuntimeError(
            "Observer did not see forced handover."
        )

    if not saw_source_without_ue:
        raise RuntimeError(
            "Source cell did not lose forced UE."
        )

    if not saw_destination_with_ue:
        raise RuntimeError(
            "Destination cell did not gain "
            "forced UE."
        )

    if not saw_destination_candidate:
        raise RuntimeError(
            "Handed-over UE never appeared in "
            "destination PF candidate population."
        )

    if len(
        transition_buffer
    ) != 0:
        raise RuntimeError(
            "Evaluation-only run unexpectedly "
            "collected PPO transitions."
        )

    if any(
        controller
        .has_unresolved_tti_boundary
        for controller in (
            rollout_controllers
        )
    ):
        raise RuntimeError(
            "Evaluation-only run left PPO "
            "trajectory boundaries."
        )


    # ==========================================================
    # MEMORY / SUMMARY
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
        "Completed handovers: "
        f"{result.num_completed_handovers}"
    )

    print(
        "Forced UE final cell: "
        f"{destination_cell}"
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
        "Destination PF candidate: "
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
        "WAVE6_DYNAMIC_HANDOVER_SMOKE_PASS"
    )


if __name__ == "__main__":
    main()
