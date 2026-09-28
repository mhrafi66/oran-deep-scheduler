from __future__ import annotations

import time

import torch

import csv
from pathlib import Path

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)
from oran_scheduler.rl.ppo_critic import (
    OneLDSPPOCritic,
    OneLDSPPOCriticConfig,
)
from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
)
from oran_scheduler.rl.ppo_multicell_rollout import (
    OneLDSPPOMultiCellRolloutConfig,
    OneLDSPPOMultiCellRolloutController,
)
from oran_scheduler.rl.ppo_multicell_training_runner import (
    run_multicell_ppo_training,
)
from oran_scheduler.rl.ppo_multistream_buffer import (
    PPOMultiStreamBufferConfig,
    PPOMultiStreamTransitionBuffer,
)
from oran_scheduler.rl.ppo_multistream_training import (
    PPOMultiStreamTrainingCoordinator,
)
from oran_scheduler.rl.ppo_multistream_update import (
    PPOMultiStreamUpdateConfig,
)
from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
)
from oran_scheduler.rl.ppo_sionna_chunked import (
    CellChunkedSionnaPPOInputProvider,
    ChunkedSionnaPPOConfig,
    build_chunked_sionna_ppo_context,
    build_chunked_state_managers,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingRunnerConfig,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizerConfig,
    create_ppo_optimizers,
)

from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)
from oran_scheduler.simulator.traffic import (
    TrafficBufferManager,
    build_training_ftp3_config,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)

from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertBufferConfig,
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_expert_guidance import (
    PPOExpertGuidanceConfig,
)
from oran_scheduler.rl.ppo_expert_training import (
    PPOExpertGuidanceCoordinator,
)
from oran_scheduler.rl.ppo_pf_expert import (
    PPOPFExpertConfig,
)

from oran_scheduler.rl.ppo_training_metrics import (
    PPOTrainingMetricsObserver,
)

from oran_scheduler.rl.ppo_checkpoint import (
    save_ppo_model_checkpoint,
)

from oran_scheduler.utils.perf_timing import (
    print_perf_timings,
    reset_perf_timings,
)

METRICS_PATH = Path(
    "experiments/"
    "perf_gpu_parallel_training_metrics.csv"
)

KPI_METRICS_PATH = Path(
    "experiments/"
    "perf_gpu_parallel_kpi_metrics.csv"
)

CHECKPOINT_PATH = Path(
    "experiments/checkpoints/"
    "ppo_1lds_perf_gpu_parallel.pt"
)

DEVICE = torch.device(
    "cuda:0"
)

SEED = 1234


# PAPER-SPECIFIED scheduler dimensions.
NUM_CANDIDATES = 10

NUM_RBGS = 18

NUM_USER_SLOTS = 4


# # PAPER-SPECIFIED full centralized training network.
# NUM_TRAINING_CELLS = 21

# TEMPORARY KINGSPEAK DEVELOPMENT CONFIG.
#
# OPEN-REPRODUCTION ENGINEERING:
# Execute only two PPO cell streams while retaining
# the complete 420-UE / 21-cell global topology.
NUM_TRAINING_CELLS = 21

#
# OPEN-REPRODUCTION ENGINEERING.
#
# Maximum number of serving UEs passed to one
# paper-array Sionna generation call.
#
UE_MICROBATCH_SIZE = 4


# Scalability probe only:
#
# advance exactly one paper-sized warm-up TTI.
NUM_TTIS = 1


# --------------------------------------------------------------
# TEST-SCALE PPO UPDATE BOUNDARY.
#
# PAPER:
#     M = 128
#
# This smoke run intentionally uses:
#
#     2 cells
#       x
#     4 user slots
#       =
#     8 joint transitions
#
# so one complete optimizer update occurs quickly.
#
# This is NOT the final reproduction setting.
# --------------------------------------------------------------

# # PAPER-SPECIFIED.
# #
# # This value is inactive during the warm-up-only
# # scalability run because no PPO experience is
# # collected before TTI 100.
# UPDATE_SIZE = 128

# TEST-SCALE ONLY.
#
# 2 cells x 4 user slots = 8 transitions at the
# first cross-TTI update boundary.
#
# Paper uses M = 128.
UPDATE_SIZE = 128

# --------------------------------------------------------------
# OPEN-REPRODUCTION PARAMETERS.
# --------------------------------------------------------------

INITIAL_AVERAGE_THROUGHPUT_BPS = (
    1.0e6
)

THROUGHPUT_FORGETTING_FACTOR = (
    0.9
)

THROUGHPUT_NORMALIZATION_BPS = (
    100.0e6
)

FULL_BUFFER_STATE_BITS = (
    12_000.0
)

BUFFER_NORMALIZATION_BITS = (
    FULL_BUFFER_STATE_BITS
)

CQI_NORMALIZATION = 15.0

GEOMETRIC_MEAN_NORMALIZER_BPS = (
    100.0e6
)

GAE_LAMBDA = 0.9

ENTROPY_COEFFICIENT = 0.0


# ==============================================================
# PF EXPERT / TEACHER 2
# ==============================================================

# PAPER-SPECIFIED.
EXPERT_BUFFER_CAPACITY = 4000


# PAPER-UNSPECIFIED.
#
# Test-scale choice for this real-Sionna integration.
EXPERT_BATCH_SIZE = 8


# PAPER-UNSPECIFIED.
#
# Keep explicit for later sensitivity experiments.
EXPERT_GUIDANCE_WEIGHT = 1.0


# ==============================================================
# CANDIDATE PERMUTATION
# ==============================================================

# PAPER:
#     candidate-order permutation is used.
#
# PAPER-UNSPECIFIED:
#     numerical N_Pi.
#
# Test-scale choice:
NUM_CANDIDATE_PERMUTATIONS = 1

INCLUDE_ORIGINAL_SAMPLE = True


PPO_AUGMENTATION_SEED = 1111

EXPERT_AUGMENTATION_SEED = 2222

# 30-kHz normal-CP NR slot.
#
# The current RateConfig also uses 0.5 ms.
# Keep this explicit rather than burying traffic
# timing inside TrafficBufferManager.
TTI_DURATION_S = 0.5e-3


# # PAPER-SPECIFIED:
# #
# # normal agent-sample collection begins after
# # the initial 100 TTIs.
# FIRST_COLLECTION_TTI_INDEX = 0


# TEST-SCALE ONLY.
#
# Start collection immediately so this short
# real-Sionna run proves an actual PPO/JSD update.
#
# Paper uses first_collection_tti_index = 100.
FIRST_COLLECTION_TTI_INDEX = 0

def module_parameter_norm(
    module: torch.nn.Module,
) -> float:
    with torch.no_grad():
        total = torch.zeros(
            (),
            dtype=torch.float64,
            device=DEVICE,
        )

        for parameter in module.parameters():
            total += (
                parameter
                .detach()
                .to(
                    dtype=torch.float64
                )
                .square()
                .sum()
            )

        return float(
            torch.sqrt(
                total
            ).item()
        )


def build_mixed_training_traffic_managers(
    serving_global_ue_indices_by_stream,
) -> tuple[
    tuple[
        TrafficBufferManager,
        ...,
    ],
    tuple[
        torch.Tensor,
        ...,
    ],
]:
    """
    Build the paper's mixed training traffic at
    integration-test scale.

    PAPER-SPECIFIED:
        50% Full Buffer
        50% FTP3

        FTP3:
            1500 bytes / packet
            500 packets / second

    TEMPORARY SMOKE-RUN CHOICE:
        We alternate FB / FTP3 across REAL serving
        UEs in global stream order.

TEMPORARY SMOKE-RUN CHOICE:
    Alternate FB / FTP3 across all REAL serving UEs
    of the selected PPO streams.

    The assignment counter spans streams, which keeps
    the overall selected-UE population approximately
    50% Full Buffer and 50% FTP3.
            3 FB
            3 FTP3

        exactly 50 / 50.

    TEMPORARY INITIALIZATION:
        Every FTP3 UE begins with one packet.

        The final paper run will instead use the
        paper's 100-TTI warm-up before PPO collection.

        We prefill here because this two-TTI smoke
        run intentionally starts collection at TTI 0.
    """

    ftp3_config = (
        build_training_ftp3_config(
            tti_duration_s=(
                TTI_DURATION_S
            ),
        )
    )

    managers: list[
        TrafficBufferManager
    ] = []

    full_buffer_masks: list[
        torch.Tensor
    ] = []

    #
    # This counter spans ALL selected PPO cells.
    #
    # Therefore the 50/50 assignment is global,
    # rather than independently rounding within
    # each small three-UE cell.
    #
    real_ue_ordinal = 0

    for (
        stream_index,
        global_ue_indices,
    ) in enumerate(
        serving_global_ue_indices_by_stream
    ):
        valid_mask = torch.ones(
            global_ue_indices.shape,
            dtype=torch.bool,
            device=(
                global_ue_indices.device
            ),
        )

        full_buffer_mask = (
            torch.zeros_like(
                valid_mask,
                dtype=torch.bool,
            )
        )

        for serving_slot_index in range(
            valid_mask.shape[0]
        ):
            if not bool(
                valid_mask[
                    serving_slot_index
                ].item()
            ):
                continue

            #
            # Even global real-UE ordinal:
            #     Full Buffer
            #
            # Odd:
            #     FTP3
            #
            if (
                real_ue_ordinal
                % 2
                == 0
            ):
                full_buffer_mask[
                    serving_slot_index
                ] = True

            real_ue_ordinal += 1

        ftp3_mask = (
            valid_mask
            & (~full_buffer_mask)
        )

        #
        # Because the smoke run skips the real
        # 100-TTI warm-up, begin each FTP3 UE with
        # one packet so the queue-limited delivery
        # path is definitely exercised.
        #
        # initial_ftp_buffer_bits = torch.where(
        #     ftp3_mask,
        #     torch.full(
        #         valid_mask.shape,
        #         fill_value=(
        #             ftp3_config
        #             .packet_size_bits
        #         ),
        #         dtype=torch.float32,
        #         device=valid_mask.device,
        #     ),
        #     torch.zeros(
        #         valid_mask.shape,
        #         dtype=torch.float32,
        #         device=valid_mask.device,
        #     ),
        # )

        # manager = TrafficBufferManager(
        #     full_buffer_mask=(
        #         full_buffer_mask
        #     ),
        #     ftp3_config=(
        #         ftp3_config
        #     ),
        #     full_buffer_state_bits=(
        #         FULL_BUFFER_STATE_BITS
        #     ),
        #     initial_ftp_buffer_bits=(
        #         initial_ftp_buffer_bits
        #     ),
        #     seed=(
        #         SEED
        #         + stream_index
        #     ),
        # )

        manager = TrafficBufferManager(
            full_buffer_mask=(
                full_buffer_mask
            ),
            ftp3_config=(
                ftp3_config
            ),
            full_buffer_state_bits=(
                FULL_BUFFER_STATE_BITS
            ),
            initial_ftp_buffer_bits=None,
            seed=(
                SEED
                + stream_index
            ),
        )

        managers.append(
            manager
        )

        full_buffer_masks.append(
            full_buffer_mask
        )

    return (
        tuple(
            managers
        ),
        tuple(
            full_buffer_masks
        ),
    )

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This real-Sionna integration script "
            "requires a CUDA GPU."
        )

    torch.manual_seed(
        SEED
    )

    print()
    print("=" * 72)
    print(
        "REAL SIONNA -> CENTRALIZED 1LDS-PPO SMOKE RUN"
    )
    print("=" * 72)

    print(
        f"Device:                 {DEVICE}"
    )

    print(
        "GPU:                    "
        f"{torch.cuda.get_device_name(DEVICE)}"
    )

    print(
        f"PPO streams:            "
        f"{NUM_TRAINING_CELLS}"
    )

    print(
        f"User slots / cell:      "
        f"{NUM_USER_SLOTS}"
    )

    print(
        f"RBGs:                   "
        f"{NUM_RBGS}"
    )

    print(
        "UE microbatch size:      "
        f"{UE_MICROBATCH_SIZE} "
        "(open-reproduction)"
    )

    # print(
    #     f"Smoke update size:      "
    #     f"{UPDATE_SIZE}"
    # )

    print(
        "PPO update size M:      "
        f"{UPDATE_SIZE}"
    )

    print(
        "Expert guidance:        enabled during collection"
    )

    print(
        "Expert buffer:          "
        f"{EXPERT_BUFFER_CAPACITY}"
    )

    print(
        "Expert mini-batch:      "
        f"{EXPERT_BATCH_SIZE} "
        "(open-reproduction)"
    )

    # print(
    #     "Permutation:            "
    #     "original + "
    #     f"{NUM_CANDIDATE_PERMUTATIONS}"
    # )

    print(
        "Permutation:            enabled at PPO update"
    )
    print(
        "Collection starts TTI:  "
        f"{FIRST_COLLECTION_TTI_INDEX}"
    )

    # print(
    #     "Run type:               "
    #     "21-cell warm-up scalability"
    # )

    print(
        "Run type:               "
        "2-cell real-Sionna repeated-learning run"
    )

    print(
        "Radio evolution:        refreshed each TTI"
    )

    print(
        "Radio temporal model:   independent "
        "realizations"
    )

    print(
        "Topology/association:   fixed "
        "(temporary)"
    )

    print(
        "Traffic:                "
        "50% FB + 50% FTP3"
    )

    print("=" * 72)
    print()

    # snapshot_start = (
    #     time.perf_counter()
    # )

    # ==========================================================
    # PAPER-SCALE TRAINING POPULATION
    #
    # 21 cells
    # x
    # 20 generated UEs / sector
    # =
    # 420 UEs
    #
    # Only NUM_TRAINING_CELLS streams are executed
    # in THIS first chunked smoke run.
    # ==========================================================

    context_start = (
        time.perf_counter()
    )

    context = (
        build_chunked_sionna_ppo_context(
            config=(
                ChunkedSionnaPPOConfig(
                    num_training_cells=(
                        NUM_TRAINING_CELLS
                    ),

                    #
                    # PAPER-SPECIFIED.
                    #
                    num_ut_per_sector=20,

                    num_rbs=18,

                    num_rbgs=18,

                    subcarriers_per_rb=12,

                    ue_microbatch_size=(
                        UE_MICROBATCH_SIZE
                    ),

                    topology_seed=42,

                    association_channel_seed=1000,

                    mimo_channel_seed=2000,

                    device="cuda:0",
                )
            )
        )
    )

    torch.cuda.synchronize(
        DEVICE
    )

    context_elapsed = (
        time.perf_counter()
        - context_start
    )

    print(
        "Complete topology UEs:   "
        f"{context.num_global_ues}"
    )

    print(
        "Complete topology cells: "
        f"{context.num_cells}"
    )

    print(
        "Selected PPO cells:      "
        f"{context.selected_cell_indices}"
    )

    test_to_print_temp = f"{tuple( int(indices.numel()) for indices in context.global_ue_indices_by_stream )}"
    print(
        "UEs in selected cells:   "
        f"{test_to_print_temp}"
    )

    print(
        "Context build time:      "
        f"{context_elapsed:.3f} s"
    )

    print()


    # ==========================================================
    # CHUNKED PAPER-MIMO PROVIDER
    # ==========================================================

    radio_input_provider = (
        CellChunkedSionnaPPOInputProvider(
            context=context
        )
    )

    #
    # Generate TTI 0 now because persistent traffic
    # and PF-history managers need the serving UE
    # identity vectors.
    #
    # radio_input_provider.prepare_tti(
    #     0
    # )

    # print(
    #     "Initial chunk observations:"
    # )

    # for (
    #     stream_index,
    #     observation,
    # ) in enumerate(
    #     radio_input_provider
    #     .current_observations
    # ):
    #     real_cell = (
    #         context
    #         .selected_cell_indices[
    #             stream_index
    #         ]
    #     )

    #     print(
    #         f"  stream {stream_index}: "
    #         f"real cell {real_cell}, "
    #         f"{int(observation.serving_ue_valid_mask.sum().item())} UEs"
    #     )

    #     print(
    #         "    precoder directions: "
    #         f"{tuple(observation.precoder_directions.shape)}"
    #     )

    # print()


    # ==========================================================
    # PERSISTENT PF HISTORY
    # ==========================================================

    state_managers = (
        build_chunked_state_managers(
            context=context,

            initial_average_throughput_bps=(
                INITIAL_AVERAGE_THROUGHPUT_BPS
            ),

            throughput_forgetting_factor=(
                THROUGHPUT_FORGETTING_FACTOR
            ),

            num_candidates=(
                NUM_CANDIDATES
            ),
        )
    )


    input_provider = (
        radio_input_provider
    )

    # torch.cuda.synchronize(
    #     DEVICE
    # )

    # snapshot_elapsed = (
    #     time.perf_counter()
    #     - snapshot_start
    # )

    # print(
    #     "Real channel shape:      "
    #     f"{tuple(snapshot.h_freq.shape)}"
    # )

    # print(
    #     "Selected real cells:     "
    #     f"{snapshot.selected_cell_indices}"
    # )

    # print(
    #     "Associated UEs/stream:   "
    #     f"{snapshot.selected_num_ues_per_cell}"
    # )

    # print(
    #     "Snapshot build time:     "
    #     f"{snapshot_elapsed:.3f} s"
    # )

    # print()

    # for stream_index, observation in enumerate(
    #     snapshot.observations
    # ):
    #     print(
    #         f"Stream {stream_index}:"
    #     )

    #     print(
    #         "  real cell:            "
    #         f"{snapshot.selected_cell_indices[stream_index]}"
    #     )

    #     print(
    #         "  serving layout:       "
    #         f"{tuple(observation.serving_global_ue_indices.shape)}"
    #     )

    #     print(
    #         "  real serving UEs:     "
    #         f"{int(observation.serving_ue_valid_mask.sum().item())}"
    #     )

    #     print(
    #         "  subband CQI:          "
    #         f"{tuple(observation.subband_cqi.shape)}"
    #     )

    #     print(
    #         "  precoder directions:  "
    #         f"{tuple(observation.precoder_directions.shape)}"
    #     )

    #     print()


    # # ==========================================================
    # # PERSISTENT CELL STATE
    # # ==========================================================

    # state_managers = (
    #     build_snapshot_state_managers(
    #         snapshot=snapshot,
    #         initial_average_throughput_bps=(
    #             INITIAL_AVERAGE_THROUGHPUT_BPS
    #         ),
    #         throughput_forgetting_factor=(
    #             THROUGHPUT_FORGETTING_FACTOR
    #         ),
    #         num_candidates=(
    #             NUM_CANDIDATES
    #         ),
    #     )
    # )

    (
        traffic_managers,
        full_buffer_masks,
    ) = (
        build_mixed_training_traffic_managers(
            context
            .global_ue_indices_by_stream
        )
    )

    num_full_buffer_ues = sum(
        int(
            full_buffer_mask
            .sum()
            .item()
        )
        for full_buffer_mask
        in full_buffer_masks
    )

    num_real_ues = sum(
        int(
            global_ue_indices
            .numel()
        )

        for global_ue_indices
        in (
            context
            .global_ue_indices_by_stream
        )
    )

    num_ftp3_ues = (
        num_real_ues
        - num_full_buffer_ues
    )

    print(
        "Training traffic:"
    )

    print(
        "  Full Buffer UEs:       "
        f"{num_full_buffer_ues}"
    )

    print(
        "  FTP3 UEs:              "
        f"{num_ftp3_ues}"
    )

    print(
        "  FB fraction:           "
        f"{num_full_buffer_ues / num_real_ues:.3f}"
    )

    print()

    # ==========================================================
    # NETWORK-LEVEL PPO KPI OBSERVER
    #
    # Collects actual delivered throughput, long-term
    # throughput statistics, reward geometric mean, and
    # greedy/PF reward diagnostics from every active cell.
    #
    # This retains only small CPU-side metric tensors;
    # it does NOT retain the large Sionna PHY tensors.
    # ==========================================================

    kpi_observer = (
        PPOTrainingMetricsObserver(
            num_cells=(
                NUM_TRAINING_CELLS
            ),
            csv_path=(
                KPI_METRICS_PATH
            ),
        )
    )





    # radio_input_provider = (
    #     IndependentSionnaPPOTTIInputProvider(
    #         initial_snapshot=(
    #             snapshot
    #         ),
    #         initial_tti_index=0,
    #     )
    # )

    # input_provider = (
    #     radio_input_provider
    # )

    # # ==========================================================
    # # DROP THE EXTRA LOCAL REFERENCE
    # # ==========================================================
    # #
    # # radio_input_provider now owns the initial snapshot.
    # #
    # # We no longer need the separate local variable
    # # `snapshot`.
    # #
    # # At TTI 1, when radio_input_provider replaces its
    # # current snapshot with the refreshed one, the old
    # # large H tensor can then become eligible for release.
    # #
    # del snapshot

    # ==========================================================
    # PAPER-SHAPED 1LDS STATE
    #
    # 10 x (5 + 2*18) = 410
    # ==========================================================

    state_config = OneLDSStateConfig(
        throughput_normalization_bps=(
            THROUGHPUT_NORMALIZATION_BPS
        ),
        buffer_normalization=(
            BUFFER_NORMALIZATION_BITS
        ),
        subband_cqi_normalization=(
            CQI_NORMALIZATION
        ),
        num_candidates=(
            NUM_CANDIDATES
        ),
        num_rbgs=(
            NUM_RBGS
        ),
        max_rank=2,
    )

    assert (
        state_config.state_size
        == 410
    )


    # ==========================================================
    # PAPER-SHAPED PPO NETWORKS
    # ==========================================================

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

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )


    # ==========================================================
    # CENTRALIZED MULTI-CELL ROLLOUT BUFFER
    # ==========================================================

    transition_buffer = (
        PPOMultiStreamTransitionBuffer(
            config=(
                PPOMultiStreamBufferConfig(
                    num_streams=(
                        NUM_TRAINING_CELLS
                    ),
                    update_size=(
                        UPDATE_SIZE
                    ),
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
                    ),
                )
            ),
        )
        for stream_index
        in range(
            NUM_TRAINING_CELLS
        )
    )


    # ==========================================================
    # PAPER PPO EXPERT BUFFER
    # ==========================================================

    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=(
                    EXPERT_BUFFER_CAPACITY
                ),
                sampling_with_replacement=False,
            )
        )
    )


    # ==========================================================
    # TEACHER 2:
    # PF EXPERT -> JSD ACTOR UPDATE
    # ==========================================================

    expert_guidance = (
        PPOExpertGuidanceCoordinator(
            expert_buffer=(
                expert_buffer
            ),
            guidance_config=(
                PPOExpertGuidanceConfig(
                    batch_size=(
                        EXPERT_BATCH_SIZE
                    ),
                    guidance_weight=(
                        EXPERT_GUIDANCE_WEIGHT
                    ),
                    divergence_mode=(
                        "true_jsd"
                    ),
                )
            ),
        )
    )


    # ==========================================================
    # CANDIDATE PERMUTATION
    # ==========================================================

    augmentation_config = (
        PPOCandidateAugmentationConfig(
            num_permutations=(
                NUM_CANDIDATE_PERMUTATIONS
            ),
            include_original=(
                INCLUDE_ORIGINAL_SAMPLE
            ),
        )
    )


    ppo_augmentation_generator = (
        torch.Generator(
            device="cpu"
        )
    )

    ppo_augmentation_generator.manual_seed(
        PPO_AUGMENTATION_SEED
    )


    expert_augmentation_generator = (
        torch.Generator(
            device="cpu"
        )
    )

    expert_augmentation_generator.manual_seed(
        EXPERT_AUGMENTATION_SEED
    )

    # ==========================================================
    # CENTRAL LEARNER
    #
    # PPO candidate permutation + PF expert/JSD
    # are both enabled.
    # ==========================================================

    centralized_training = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=(
                transition_buffer
            ),

            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode=(
                        # "require_exact"
                        "use_all_when_reached"
                    ),
                )
            ),

            gae_config=(
                PPOGAEConfig(
                    gae_lambda=(
                        GAE_LAMBDA
                    ),
                )
            ),

            loss_config=(
                PPOLossConfig(
                    entropy_coefficient=(
                        ENTROPY_COEFFICIENT
                    ),
                )
            ),

            #
            # Teacher 1 representation augmentation.
            #
            candidate_augmentation_config=(
                augmentation_config
            ),

            candidate_augmentation_generator=(
                ppo_augmentation_generator
            ),

            #
            # Teacher 2.
            #
            expert_guidance_coordinator=(
                expert_guidance
            ),
        )
    )

    initial_actor_norm = (
        module_parameter_norm(
            actor
        )
    )

    initial_critic_norm = (
        module_parameter_norm(
            critic
        )
    )



    # ==========================================================
    # RUN TWO REAL-SIONNA PPO TTIs
    # ==========================================================

    total_start = (
        time.perf_counter()
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_fieldnames = [
        "tti",
        "elapsed_s",
        "peak_gpu_mib",
        "buffer_size",
        "ppo_updates_this_tti",
        "total_ppo_updates",
        "expert_demos_added",
        "expert_buffer_size",
        "jsd_updates_this_tti",
        "mean_td_rate_mbps",
        "mean_wideband_cqi",
        "actor_norm",
        "critic_norm",
        "actor_loss",
        "critic_loss",
        "mean_ppo_ratio",
        "mean_advantage",
        "raw_jsd_loss",
        "weighted_jsd_loss",
    ]

    with METRICS_PATH.open(
        "w",
        newline="",
    ) as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=metrics_fieldnames,
        )
        writer.writeheader()

    for tti_index in range(
        NUM_TTIS
    ):
        updates_before = (
            centralized_training
            .num_updates
        )

        expert_updates_before = (
            centralized_training
            .num_expert_guidance_updates
        )

        expert_buffer_before = len(
            expert_buffer
        )

        torch.cuda.reset_peak_memory_stats(
            DEVICE
        )

        tti_start = (
            time.perf_counter()
        )
        reset_perf_timings()
        result = run_multicell_ppo_training(
            start_tti_index=(
                tti_index
            ),

            #
            # One TTI at a time only so we can
            # inspect boundary behavior.
            #
            num_ttis=1,

            input_provider=(
                input_provider
            ),

            traffic_managers=(
                traffic_managers
            ),

            state_managers=(
                state_managers
            ),

            rollout_controllers=(
                rollout_controllers
            ),

            actor=actor,

            critic=critic,

            optimizers=optimizers,

            centralized_training=(
                centralized_training
            ),

            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode=(
                        "data_available_only"
                    ),
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
                        GEOMETRIC_MEAN_NORMALIZER_BPS
                    ),
                )
            ),

            #
            # PRIMARY CURRENT REPRODUCTION
            # INTERPRETATION.
            #
            reward_population=(
                "candidates"
            ),

            #
            # TEST-SCALE ONLY.
            #
            # Paper:
            #     TTIs 0..99 warm up
            #     collection starts at 100.
            #
            # We collect from TTI 0 only so this
            # two-TTI integration run can prove that
            # an optimizer update really occurs.
            #
            # runner_config=(
            #     PPOTrainingRunnerConfig(
            #         first_collection_tti_index=0,
            #     )
            # ),

            runner_config=(
                PPOTrainingRunnerConfig(
                    first_collection_tti_index=(
                        FIRST_COLLECTION_TTI_INDEX
                    ),
                )
            ),

            #
            # Teacher 2 disabled at this stage.
            #
            expert_buffer=(
                expert_buffer
            ),

            pf_expert_config=(
                PPOPFExpertConfig()
            ),

            expert_candidate_augmentation_config=(
                augmentation_config
            ),

            expert_candidate_augmentation_generator=(
                expert_augmentation_generator
            ),

            reward_reduction="mean",

            cell_result_observer=(
                kpi_observer
            ),

            device=DEVICE,
        )

        torch.cuda.synchronize(
            DEVICE
        )

        print_perf_timings()

        print(
            "  chunked TTI builds:   "
            f"{radio_input_provider.num_tti_builds}"
        )

        print(
            "  cell chunk builds:    "
            f"{radio_input_provider.num_stream_builds}"
        )

        print(
            "Channel generations:     "
            f"{radio_input_provider.num_channel_generations}"
        )

        print(
            "Channel topology resets: "
            f"{radio_input_provider.num_channel_topology_resets}"
        )

        current_observations = (
            radio_input_provider
            .current_observations
        )

        example_observation = (
            current_observations[
                0
            ]
        )

        example_valid_mask = (
            example_observation
            .serving_ue_valid_mask
        )

        mean_td_rate_mbps = float(
            example_observation
            .td_instantaneous_rate_bps[
                example_valid_mask
            ]
            .mean()
            .item()
            / 1.0e6
        )

        mean_wideband_cqi = float(
            example_observation
            .wideband_cqi[
                example_valid_mask
            ]
            .mean()
            .item()
        )

        print(
            "  stream-0 mean TD rate:"
            f" {mean_td_rate_mbps:.3f} Mbps"
        )

        print(
            "  stream-0 mean WB CQI: "
            f"{mean_wideband_cqi:.3f}"
        )

        elapsed = (
            time.perf_counter()
            - tti_start
        )

        updates_after = (
            centralized_training
            .num_updates
        )

        expert_updates_after = (
            centralized_training
            .num_expert_guidance_updates
        )

        peak_memory_mib = (
            torch.cuda
            .max_memory_allocated(
                DEVICE
            )
            / (
                1024.0
                * 1024.0
            )
        )

        print(
            "-" * 72
        )

        print(
            f"TTI {tti_index}"
        )

        print(
            "  elapsed:              "
            f"{elapsed:.3f} s"
        )

        print(
            "  centralized buffer:   "
            f"{result.final_transition_buffer_size}"
        )

        print(
            "  PPO updates this TTI: "
            f"{updates_after - updates_before}"
        )

        print(
            "  total PPO updates:    "
            f"{updates_after}"
        )


        print(
            "  expert demos added:  "
            f"{len(expert_buffer) - expert_buffer_before}"
        )

        print(
            "  expert buffer size:  "
            f"{len(expert_buffer)}"
        )

        print(
            "  JSD updates this TTI:"
            f" {expert_updates_after - expert_updates_before}"
        )
        print(
            "  unresolved boundaries:"
            f" {result.unresolved_boundary_by_cell}"
        )

        print(
            "  peak allocated GPU:   "
            f"{peak_memory_mib:.1f} MiB"
        )

    # print(
    #     "Cell chunk builds:       "
    #     f"{radio_input_provider.num_stream_builds}"
    # )

        # print(
        #     "  radio channel seed:   "
        #     f"{current_radio.channel_seed_used}"
        # )

        # print(
        #     "  channel refreshes:    "
        #     f"{radio_input_provider.num_channel_refreshes}"
        # )

        print(
            "  chunked TTI builds:   "
            f"{radio_input_provider.num_tti_builds}"
        )

        print(
            "  stream-0 mean TD rate:"
            f" {mean_td_rate_mbps:.3f} Mbps"
        )

        print(
            "  stream-0 mean WB CQI: "
            f"{mean_wideband_cqi:.3f}"
        )

        print(
            "  stream-0 mean TD rate:"
            f" {mean_td_rate_mbps:.3f} Mbps"
        )

        print(
            "  stream-0 mean WB CQI: "
            f"{mean_wideband_cqi:.3f}"
        )


        actor_loss_value = None
        critic_loss_value = None
        mean_ppo_ratio_value = None
        mean_advantage_value = None
        raw_jsd_loss_value = None
        weighted_jsd_loss_value = None


        if (
            updates_after
            > updates_before
        ):
            update = (
                centralized_training
                .last_update
            )

            if update is None:
                raise RuntimeError(
                    "PPO update occurred without "
                    "stored diagnostics."
                )


            actor_loss_value = float(
                update
                .ppo_update
                .actor_loss
                .item()
            )

            critic_loss_value = float(
                update
                .ppo_update
                .critic_loss
                .item()
            )

            mean_ppo_ratio_value = float(
                update
                .ppo_update
                .mean_probability_ratio
                .item()
            )

            mean_advantage_value = float(
                update
                .ppo_update
                .mean_advantage
                .item()
            )



            print()

            print(
                "  >>> CENTRAL PPO UPDATE"
            )

            print(
                "      real transitions:   "
                f"{update.num_real_transitions}"
            )

            print(
                "      optimizer samples:  "
                f"{update.num_optimizer_samples}"
            )

            print(
                "      actor loss:         "
                f"{float(update.ppo_update.actor_loss.item()): .6f}"
            )

            print(
                "      critic loss:        "
                f"{float(update.ppo_update.critic_loss.item()): .6f}"
            )

            print(
                "      mean PPO ratio:     "
                f"{float(update.ppo_update.mean_probability_ratio.item()): .6f}"
            )

            print(
                "      mean advantage:     "
                f"{float(update.ppo_update.mean_advantage.item()): .6f}"
            )

            if (
                update.expert_update
                is None
            ):
                raise RuntimeError(
                    "Real-Sionna PPO update occurred "
                    "without the enabled expert "
                    "guidance update."
                )

            expert_loss = (
                update
                .expert_update
                .guidance_update
                .loss_data
            )

            raw_jsd_loss_value = float(
                expert_loss.raw_loss.item()
            )

            weighted_jsd_loss_value = float(
                expert_loss.weighted_loss.item()
            )

            print()

            print(
                "  >>> PF EXPERT / JSD UPDATE"
            )

            print(
                "      raw JSD loss:       "
                f"{float(expert_loss.raw_loss.item()): .6f}"
            )

            print(
                "      weighted JSD loss:  "
                f"{float(expert_loss.weighted_loss.item()): .6f}"
            )   
        print()
        current_actor_norm = module_parameter_norm(
            actor
        )

        current_critic_norm = module_parameter_norm(
            critic
        )


        metrics_row = {
            "tti": tti_index,
            "elapsed_s": elapsed,
            "peak_gpu_mib": peak_memory_mib,
            "buffer_size": (
                result.final_transition_buffer_size
            ),
            "ppo_updates_this_tti": (
                updates_after - updates_before
            ),
            "total_ppo_updates": updates_after,
            "expert_demos_added": (
                len(expert_buffer)
                - expert_buffer_before
            ),
            "expert_buffer_size": len(
                expert_buffer
            ),
            "jsd_updates_this_tti": (
                expert_updates_after
                - expert_updates_before
            ),
            "mean_td_rate_mbps": (
                mean_td_rate_mbps
            ),
            "mean_wideband_cqi": (
                mean_wideband_cqi
            ),
            "actor_norm": (
                current_actor_norm
            ),
            "critic_norm": (
                current_critic_norm
            ),
            "actor_loss": (
                actor_loss_value
            ),
            "critic_loss": (
                critic_loss_value
            ),
            "mean_ppo_ratio": (
                mean_ppo_ratio_value
            ),
            "mean_advantage": (
                mean_advantage_value
            ),
            "raw_jsd_loss": (
                raw_jsd_loss_value
            ),
            "weighted_jsd_loss": (
                weighted_jsd_loss_value
            ),
        }


        with METRICS_PATH.open(
            "a",
            newline="",
        ) as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=metrics_fieldnames,
            )
            writer.writerow(
                metrics_row
            )


        # --------------------------------------------------
        # DURABLE PER-TTI MODEL CHECKPOINT
        #
        # Save to a temporary file first, then atomically
        # replace the latest checkpoint. Therefore a Slurm
        # timeout during a later TTI cannot destroy the last
        # successfully completed checkpoint.
        #
        # NOTE:
        # This preserves model/optimizer state only.
        # It is NOT an exact simulator-state resume point.
        # --------------------------------------------------

        temporary_checkpoint_path = (
            CHECKPOINT_PATH.with_name(
                CHECKPOINT_PATH.stem
                + ".tmp"
                + CHECKPOINT_PATH.suffix
            )
        )

        save_ppo_model_checkpoint(
            path=temporary_checkpoint_path,
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            tti_index=tti_index,
            num_ppo_updates=(
                centralized_training.num_updates
            ),
            metadata={
                "num_training_cells": (
                    NUM_TRAINING_CELLS
                ),
                "num_user_slots": (
                    NUM_USER_SLOTS
                ),
                "num_rbgs": NUM_RBGS,
                "update_size": UPDATE_SIZE,
                "first_collection_tti": (
                    FIRST_COLLECTION_TTI_INDEX
                ),
                "ue_microbatch_size": (
                    UE_MICROBATCH_SIZE
                ),
                "run_type": (
                    "accelerated_full_scale_training"
                ),
                "sample_semantics": (
                    "one joint PPO transition per "
                    "cell/user-layer; 18 RBG branches "
                    "per transition"
                ),
            },
        )

        temporary_checkpoint_path.replace(
            CHECKPOINT_PATH
        )

        print(
            "  latest checkpoint:     "
            f"{CHECKPOINT_PATH}"
        )


    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    final_actor_norm = (
        module_parameter_norm(
            actor
        )
    )

    final_critic_norm = (
        module_parameter_norm(
            critic
        )
    )

    print("=" * 72)
    print(
        "REAL-SIONNA PPO SMOKE SUMMARY"
    )
    print("=" * 72)

    print(
        "Total execution time:    "
        f"{total_elapsed:.3f} s"
    )

    print(
        "Global training population: "
        "420 UEs"
    )

    print(
        "Paper MIMO storage:       "
        "serving-cell chunks"
    )

    print(
        "BS links per chunk:       "
        "all 21"
    )

    print(
        "PPO streams this smoke:   "
        f"{NUM_TRAINING_CELLS}"
    )

    print(
        "Radio evolution:          "
        "new chunked realization / TTI"
    )

    print(
        "PPO optimizer updates:   "
        f"{centralized_training.num_updates}"
    )

    print(
        "Actor norm before:       "
        f"{initial_actor_norm:.8f}"
    )

    print(
        "Actor norm after:        "
        f"{final_actor_norm:.8f}"
    )

    print(
        "Critic norm before:      "
        f"{initial_critic_norm:.8f}"
    )

    print(
        "Critic norm after:       "
        f"{final_critic_norm:.8f}"
    )

    print(
        "Cell chunk builds:       "
        f"{radio_input_provider.num_stream_builds}"
    )

    print()

    # if (
    #     centralized_training
    #     .num_updates
    #     != 1
    # ):
    #     raise RuntimeError(
    #         "Expected exactly one centralized PPO "
    #         "update in the two-TTI smoke run."
    #     )

    # if (
    #     centralized_training
    #     .num_updates
    #     != 0
    # ):
    #     raise RuntimeError(
    #         "Warm-up scalability run unexpectedly "
    #         "performed a PPO optimizer update."
    #     )



    # if (
    #     centralized_training
    #     .num_expert_guidance_updates
    #     != 1
    # ):
    #     raise RuntimeError(
    #         "Expected exactly one PF-expert/JSD "
    #         "update in the two-TTI smoke run."
    #     )

    # if (
    #     centralized_training
    #     .num_expert_guidance_updates
    #     != 0
    # ):
    #     raise RuntimeError(
    #         "Warm-up scalability run unexpectedly "
    #         "performed a PF-expert/JSD update."
    #     )


    # if len(
    #     expert_buffer
    # ) == 0:
    #     raise RuntimeError(
    #         "PF expert did not populate D_expert."
    #     )

    # if len(
    #     expert_buffer
    # ) != 0:
    #     raise RuntimeError(
    #         "PF expert demonstrations were collected "
    #         "during pre-collection warm-up."
    #     )


    # last_update = (
    #     centralized_training
    #     .last_update
    # )

    # if last_update is None:
    #     raise RuntimeError(
    #         "Missing final PPO update diagnostics."
    #     )


    # expected_optimizer_samples = (
    #     UPDATE_SIZE
    #     * (
    #         int(
    #             INCLUDE_ORIGINAL_SAMPLE
    #         )
    #         + NUM_CANDIDATE_PERMUTATIONS
    #     )
    # )

    # if (
    #     last_update
    #     .num_optimizer_samples
    #     != expected_optimizer_samples
    # ):
    #     raise RuntimeError(
    #         "Candidate augmentation produced the "
    #         "wrong PPO optimizer sample count."
    #     )


    # if (
    #     last_update.expert_update
    #     is None
    # ):
    #     raise RuntimeError(
    #         "Expected a Teacher-2 update."
    #     )


    # if (
    #     centralized_training
    #     .last_update
    #     is not None
    # ):
    #     raise RuntimeError(
    #         "Warm-up scalability run unexpectedly "
    #         "stored optimizer-update diagnostics."
    #     )

    # if (
    #     transition_buffer.num_transitions
    #     != 0
    # ):
    #     raise RuntimeError(
    #         "Agent transitions were collected during "
    #         "pre-collection warm-up."
    #     )

    # if (
    #     len(transition_buffer)
    #     != 0
    # ):
    #     raise RuntimeError(
    #         "Agent transitions were collected during "
    #         "pre-collection warm-up."
    #     )



    # ==============================================================
    # FULL-SCALE TRAINING
    #
    # The old 2-cell/M=8 repeated-learning assertions
    # are intentionally omitted here.
    # ==============================================================

    # ==============================================================
    # TEST-SCALE REAL-LEARNING VALIDATION
    #
    # OPEN-REPRODUCTION INTEGRATION TEST:
    #
    #   2 centralized streams
    #   4 1LDS user slots / stream
    #   collection from TTI 0
    #   update_size = 8
    #
    # TTI 0 produces the within-TTI transitions.
    # At the start of TTI 1, the cross-TTI boundaries are
    # resolved and the centralized buffer reaches 8.
    #
    # We therefore expect exactly one PPO update and one
    # PF-expert/JSD update.
    # ==============================================================

    # if (
    #     centralized_training.num_updates
    #     != 1
    # ):
    #     raise RuntimeError(
    #         "Expected exactly one centralized PPO "
    #         "update in the two-TTI learning smoke."
    #     )


    # if (
    #     centralized_training
    #     .num_expert_guidance_updates
    #     != 1
    # ):
    #     raise RuntimeError(
    #         "Expected exactly one PF-expert/JSD "
    #         "update in the two-TTI learning smoke."
    #     )


    # if len(
    #     expert_buffer
    # ) == 0:
    #     raise RuntimeError(
    #         "PF expert did not populate D_expert."
    #     )


    # last_update = (
    #     centralized_training.last_update
    # )

    # if last_update is None:
    #     raise RuntimeError(
    #         "Missing final PPO update diagnostics."
    #     )


    # expected_optimizer_samples = (
    #     UPDATE_SIZE
    #     * (
    #         int(
    #             INCLUDE_ORIGINAL_SAMPLE
    #         )
    #         + NUM_CANDIDATE_PERMUTATIONS
    #     )
    # )

    # if (
    #     last_update.num_optimizer_samples
    #     != expected_optimizer_samples
    # ):
    #     raise RuntimeError(
    #         "Candidate augmentation produced the "
    #         "wrong PPO optimizer sample count."
    #     )


    # if (
    #     last_update.expert_update
    #     is None
    # ):
    #     raise RuntimeError(
    #         "Expected a PF-expert/JSD update."
    #     )


    if not torch.isfinite(
        torch.tensor(
            [
                final_actor_norm,
                final_critic_norm,
            ],
            dtype=torch.float64,
        )
    ).all():
        raise RuntimeError(
            "Network parameters became non-finite."
        )

    # expected_channel_refreshes = (
    #     NUM_TTIS
    #     - 1
    # )

    # if (
    #     radio_input_provider
    #     .num_channel_refreshes
    #     != expected_channel_refreshes
    # ):
    #     raise RuntimeError(
    #         "Unexpected number of Sionna channel "
    #         "refreshes."
    #     )

    expected_stream_builds = (
        NUM_TTIS
        * NUM_TRAINING_CELLS
    )

    if (
        radio_input_provider
        .num_stream_builds
        != expected_stream_builds
    ):
        raise RuntimeError(
            "Unexpected number of lazy cell "
            "Sionna chunk builds."
        )

    expected_tti_builds = (
        NUM_TTIS
    )

    if (
        radio_input_provider
        .num_tti_builds
        != expected_tti_builds
    ):
        raise RuntimeError(
            "Unexpected number of chunked Sionna "
            "TTI builds."
        )

    print(
        "REAL SIONNA -> CENTRALIZED PPO "
        "INTEGRATION PASSED"
    )

    checkpoint_path = (
        save_ppo_model_checkpoint(
            path=CHECKPOINT_PATH,
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            tti_index=(
                NUM_TTIS - 1
            ),
            num_ppo_updates=(
                centralized_training
                .num_updates
            ),
            metadata={
                "num_training_cells": (
                    NUM_TRAINING_CELLS
                ),
                "num_user_slots": (
                    NUM_USER_SLOTS
                ),
                "num_rbgs": (
                    NUM_RBGS
                ),
                "update_size": (
                    UPDATE_SIZE
                ),
                "first_collection_tti": (
                    FIRST_COLLECTION_TTI_INDEX
                ),
                "ue_microbatch_size": (
                    UE_MICROBATCH_SIZE
                ),
                "topology_ues": (
                    context
                    .num_global_ues
                ),
                "sample_semantics": (
                    "one joint transition per "
                    "cell/user-layer; "
                    "18 RBG branches per transition"
                ),
            },
        )
    )

    print(
        "Expert-guidance updates: "
        f"{centralized_training.num_expert_guidance_updates}"
    )

    print(
        "Expert buffer size:      "
        f"{len(expert_buffer)}"
    )

    # print(
    #     "Optimizer rows/update:   "
    #     f"{last_update.num_optimizer_samples}"
    # )

    print(
        "Collection start TTI:    "
        f"{FIRST_COLLECTION_TTI_INDEX}"
    )

    # print(
    #     "Warm-up PPO updates:     "
    #     f"{centralized_training.num_updates}"
    # )

    print(
        "Total PPO updates:       "
        f"{centralized_training.num_updates}"
    )

    # print(
    #     "Warm-up expert updates:  "
    #     f"{centralized_training.num_expert_guidance_updates}"
    # )

    print(
        "Total expert updates:    "
        f"{centralized_training.num_expert_guidance_updates}"
    )

    print(
        "Chunked TTI builds:      "
        f"{radio_input_provider.num_tti_builds}"
    )

    print(
        "Metrics CSV:             "
        f"{METRICS_PATH}"
    )

    print(
        "KPI metrics CSV:         "
        f"{KPI_METRICS_PATH}"
    )

    print(
        "Reference checkpoint:    "
        f"{checkpoint_path}"
    )


    print("=" * 72)


if __name__ == "__main__":
    main()



