from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)

from oran_scheduler.rl.ppo_checkpoint import (
    load_ppo_model_checkpoint,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda:0")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this stress test."
    )


# ============================================================
# 1LDS DIMENSIONS
# ============================================================

NUM_CANDIDATES = 10
NUM_RBGS = 18

FEATURES_PER_CANDIDATE = (
    5
    + 2 * NUM_RBGS
)

STATE_SIZE = (
    NUM_CANDIDATES
    * FEATURES_PER_CANDIDATE
)

NUM_ACTIONS = (
    NUM_CANDIDATES
    + 1
)


# ============================================================
# TEST SCALE
# ============================================================

NUM_TEST_STATES = 8192
PERMUTATIONS_PER_STATE = 8

RANDOM_SEED = 20260912


RESULT_PATH = Path(
    "experiments/stress_tests/"
    "03_confidence_vs_permutation_summary.csv"
)

BIN_RESULT_PATH = Path(
    "experiments/stress_tests/"
    "03_confidence_vs_permutation_bins.csv"
)


CHECKPOINT_CANDIDATES = (
    Path(
        "experiments/checkpoints/"
        "ppo_1lds_notchpeak_tti6_update3.pt"
    ),
    Path(
        "experiments/checkpoints/"
        "ppo_1lds_notchpeak_latest.pt"
    ),
)


# ============================================================
# CHECKPOINT
# ============================================================

def resolve_checkpoint() -> Path:

    for path in CHECKPOINT_CANDIDATES:

        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find PPO checkpoint."
    )


# ============================================================
# CONTROLLED NORMALIZED STATES
#
# Same state distribution used by Tests 01 and 02.
# ============================================================

def build_states(
    *,
    generator: torch.Generator,
) -> torch.Tensor:

    segments = torch.zeros(
        (
            NUM_TEST_STATES,
            NUM_CANDIDATES,
            FEATURES_PER_CANDIDATE,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )


    # Past throughput
    segments[:, :, 0] = (
        0.05
        +
        0.90
        * torch.rand(
            (
                NUM_TEST_STATES,
                NUM_CANDIDATES,
            ),
            device=DEVICE,
            generator=generator,
        )
    )


    # Rank
    rank = torch.randint(
        low=1,
        high=3,
        size=(
            NUM_TEST_STATES,
            NUM_CANDIDATES,
        ),
        device=DEVICE,
        generator=generator,
    )

    segments[:, :, 1] = (
        rank.float()
        / 2.0
    )


    # Already allocated RBGs
    allocated = torch.randint(
        low=0,
        high=NUM_RBGS + 1,
        size=(
            NUM_TEST_STATES,
            NUM_CANDIDATES,
        ),
        device=DEVICE,
        generator=generator,
    )

    segments[:, :, 2] = (
        allocated.float()
        / float(NUM_RBGS)
    )


    # Buffer
    finite_buffer = torch.rand(
        (
            NUM_TEST_STATES,
            NUM_CANDIDATES,
        ),
        device=DEVICE,
        generator=generator,
    )

    full_buffer_mask = (
        torch.rand(
            (
                NUM_TEST_STATES,
                NUM_CANDIDATES,
            ),
            device=DEVICE,
            generator=generator,
        )
        < 0.5
    )

    segments[:, :, 3] = torch.where(
        full_buffer_mask,
        torch.ones_like(
            finite_buffer
        ),
        finite_buffer,
    )


    # Wideband CQI
    wb_cqi = torch.rand(
        (
            NUM_TEST_STATES,
            NUM_CANDIDATES,
        ),
        device=DEVICE,
        generator=generator,
    )

    segments[:, :, 4] = wb_cqi


    # Subband CQI
    subband_cqi = (
        wb_cqi.unsqueeze(-1)
        +
        0.15
        * torch.randn(
            (
                NUM_TEST_STATES,
                NUM_CANDIDATES,
                NUM_RBGS,
            ),
            device=DEVICE,
            generator=generator,
        )
    ).clamp(
        min=0.0,
        max=1.0,
    )

    segments[
        :,
        :,
        5:
        5 + NUM_RBGS,
    ] = subband_cqi


    # Spatial cross-correlation
    segments[
        :,
        :,
        5 + NUM_RBGS:
        5 + 2 * NUM_RBGS,
    ] = torch.rand(
        (
            NUM_TEST_STATES,
            NUM_CANDIDATES,
            NUM_RBGS,
        ),
        device=DEVICE,
        generator=generator,
    )


    return segments


# ============================================================
# PEARSON CORRELATION
# ============================================================

def pearson_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:

    x = x.float()
    y = y.float()

    x_centered = (
        x - x.mean()
    )

    y_centered = (
        y - y.mean()
    )

    numerator = (
        x_centered
        * y_centered
    ).sum()

    denominator = torch.sqrt(
        (
            x_centered.square().sum()
        )
        *
        (
            y_centered.square().sum()
        )
    )

    return (
        numerator
        / denominator.clamp_min(
            1.0e-12
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()

    print("=" * 72)

    print(
        "STRESS TEST 03: "
        "CONFIDENCE VS CANDIDATE-PERMUTATION INSTABILITY"
    )

    print("=" * 72)


    checkpoint_path = resolve_checkpoint()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    print(
        f"Device:                   {DEVICE}"
    )

    print(
        "GPU:                      "
        f"{torch.cuda.get_device_name(DEVICE)}"
    )

    print(
        "Checkpoint:               "
        f"{checkpoint_path}"
    )

    print(
        "Checkpoint TTI:           "
        f"{checkpoint.get('tti_index')}"
    )

    print(
        "Checkpoint PPO updates:   "
        f"{checkpoint.get('num_ppo_updates')}"
    )

    print(
        "States:                   "
        f"{NUM_TEST_STATES}"
    )

    print(
        "Permutations / state:     "
        f"{PERMUTATIONS_PER_STATE}"
    )


    # ========================================================
    # ACTOR
    # ========================================================

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=STATE_SIZE,
            hidden_size=32,
            num_rbgs=NUM_RBGS,
            num_actions_per_rbg=(
                NUM_ACTIONS
            ),
        )
    ).to(
        DEVICE
    )


    load_ppo_model_checkpoint(
        path=checkpoint_path,
        actor=actor,
        map_location=DEVICE,
    )

    actor.eval()


    # ========================================================
    # RANDOM GENERATOR
    # ========================================================

    generator = torch.Generator(
        device=DEVICE
    )

    generator.manual_seed(
        RANDOM_SEED
    )


    torch.cuda.reset_peak_memory_stats(
        DEVICE
    )

    torch.cuda.synchronize(
        DEVICE
    )

    start_time = time.perf_counter()


    # ========================================================
    # ORIGINAL STATES
    # ========================================================

    segments = build_states(
        generator=generator
    )

    original_states = segments.reshape(
        NUM_TEST_STATES,
        STATE_SIZE,
    )


    # ========================================================
    # ORIGINAL ACTOR OUTPUT
    # ========================================================

    with torch.inference_mode():

        original_logits = actor(
            original_states
        )

        original_probabilities = (
            torch.softmax(
                original_logits,
                dim=-1,
            )
        )


    # ========================================================
    # ORIGINAL CONFIDENCE
    # ========================================================

    top2_values, _ = torch.topk(
        original_probabilities,
        k=2,
        dim=-1,
    )

    confidence_margin = (
        top2_values[..., 0]
        -
        top2_values[..., 1]
    )


    original_argmax = (
        original_probabilities.argmax(
            dim=-1
        )
    )


    # ========================================================
    # RANDOM CANDIDATE PERMUTATIONS
    # ========================================================

    random_keys = torch.rand(
        (
            NUM_TEST_STATES,
            PERMUTATIONS_PER_STATE,
            NUM_CANDIDATES,
        ),
        device=DEVICE,
        generator=generator,
    )


    new_to_old = torch.argsort(
        random_keys,
        dim=-1,
    )


    # Inverse permutation:
    #
    # old_to_new[..., old_candidate]
    # gives its location in the permuted input.
    old_to_new = torch.empty_like(
        new_to_old
    )


    new_positions = torch.arange(
        NUM_CANDIDATES,
        dtype=torch.long,
        device=DEVICE,
    )

    new_positions = (
        new_positions
        .view(
            1,
            1,
            NUM_CANDIDATES,
        )
        .expand_as(
            new_to_old
        )
    )


    old_to_new.scatter_(
        dim=-1,
        index=new_to_old,
        src=new_positions,
    )


    # ========================================================
    # PERMUTE COMPLETE UE FEATURE SEGMENTS
    # ========================================================

    expanded_segments = (
        segments
        .unsqueeze(1)
        .expand(
            NUM_TEST_STATES,
            PERMUTATIONS_PER_STATE,
            NUM_CANDIDATES,
            FEATURES_PER_CANDIDATE,
        )
    )


    gather_index = (
        new_to_old
        .unsqueeze(-1)
        .expand(
            NUM_TEST_STATES,
            PERMUTATIONS_PER_STATE,
            NUM_CANDIDATES,
            FEATURES_PER_CANDIDATE,
        )
    )


    permuted_segments = torch.gather(
        expanded_segments,
        dim=2,
        index=gather_index,
    )


    permuted_states = (
        permuted_segments.reshape(
            (
                NUM_TEST_STATES
                * PERMUTATIONS_PER_STATE
            ),
            STATE_SIZE,
        )
    )


    # ========================================================
    # PERMUTED ACTOR OUTPUT
    # ========================================================

    with torch.inference_mode():

        permuted_logits = actor(
            permuted_states
        )

        permuted_probabilities = (
            torch.softmax(
                permuted_logits,
                dim=-1,
            )
        )


    permuted_probabilities = (
        permuted_probabilities.reshape(
            NUM_TEST_STATES,
            PERMUTATIONS_PER_STATE,
            NUM_RBGS,
            NUM_ACTIONS,
        )
    )


    # ========================================================
    # RESTORE PHYSICAL CANDIDATE IDENTITY
    # ========================================================

    candidate_probabilities = (
        permuted_probabilities[
            ...,
            :NUM_CANDIDATES
        ]
    )


    restore_index = (
        old_to_new
        .unsqueeze(2)
        .expand(
            NUM_TEST_STATES,
            PERMUTATIONS_PER_STATE,
            NUM_RBGS,
            NUM_CANDIDATES,
        )
    )


    restored_candidate_probabilities = (
        torch.gather(
            candidate_probabilities,
            dim=-1,
            index=restore_index,
        )
    )


    no_allocation_probability = (
        permuted_probabilities[
            ...,
            NUM_CANDIDATES:
            NUM_CANDIDATES + 1
        ]
    )


    restored_probabilities = torch.cat(
        (
            restored_candidate_probabilities,
            no_allocation_probability,
        ),
        dim=-1,
    )


    # ========================================================
    # DECISION STABILITY
    # ========================================================

    restored_argmax = (
        restored_probabilities.argmax(
            dim=-1
        )
    )


    reference_argmax = (
        original_argmax
        .unsqueeze(1)
        .expand(
            NUM_TEST_STATES,
            PERMUTATIONS_PER_STATE,
            NUM_RBGS,
        )
    )


    agreement = (
        restored_argmax
        == reference_argmax
    )


    # --------------------------------------------------------
    # For each ORIGINAL state/RBG:
    #
    # fraction of candidate permutations for which
    # the physical winner stays unchanged.
    #
    # Shape:
    # [state, RBG]
    # --------------------------------------------------------

    stability_fraction = (
        agreement.float().mean(
            dim=1
        )
    )


    flattened_margin = (
        confidence_margin.flatten()
    )

    flattened_stability = (
        stability_fraction.flatten()
    )


    # ========================================================
    # OVERALL RELATIONSHIP
    # ========================================================

    correlation = (
        pearson_correlation(
            flattened_margin,
            flattened_stability,
        )
    )


    overall_stability = (
        flattened_stability.mean()
    )


    # ========================================================
    # CONFIDENCE BINS
    # ========================================================

    bin_specs = (
        (
            "margin < 0.002",
            0.0,
            0.002,
        ),
        (
            "0.002 <= margin < 0.005",
            0.002,
            0.005,
        ),
        (
            "0.005 <= margin < 0.010",
            0.005,
            0.010,
        ),
        (
            "0.010 <= margin < 0.020",
            0.010,
            0.020,
        ),
        (
            "margin >= 0.020",
            0.020,
            None,
        ),
    )


    bin_rows = []


    for (
        label,
        lower,
        upper,
    ) in bin_specs:

        if upper is None:

            mask = (
                flattened_margin
                >= lower
            )

        elif lower == 0.0:

            mask = (
                flattened_margin
                < upper
            )

        else:

            mask = (
                (
                    flattened_margin
                    >= lower
                )
                &
                (
                    flattened_margin
                    < upper
                )
            )


        count = int(
            mask.sum().item()
        )


        if count > 0:

            mean_margin = float(
                flattened_margin[
                    mask
                ].mean().item()
            )

            mean_stability = float(
                flattened_stability[
                    mask
                ].mean().item()
            )

        else:

            mean_margin = float("nan")
            mean_stability = float("nan")


        bin_rows.append(
            {
                "confidence_bin":
                    label,

                "num_original_rbg_decisions":
                    count,

                "fraction_of_all_decisions":
                    (
                        count
                        /
                        flattened_margin.numel()
                    ),

                "mean_margin":
                    mean_margin,

                "mean_permutation_stability":
                    mean_stability,

                "mean_permutation_stability_percent":
                    (
                        100.0
                        * mean_stability
                    ),
            }
        )


    # ========================================================
    # SIMPLE LOW/HIGH CONFIDENCE SPLITS
    # ========================================================

    low_mask = (
        flattened_margin
        < 0.01
    )

    higher_mask = (
        flattened_margin
        >= 0.01
    )

    strong_mask = (
        flattened_margin
        >= 0.02
    )


    low_stability = (
        flattened_stability[
            low_mask
        ].mean()
    )


    higher_stability = (
        flattened_stability[
            higher_mask
        ].mean()
    )


    if strong_mask.any():

        strong_stability = (
            flattened_stability[
                strong_mask
            ].mean()
        )

    else:

        strong_stability = torch.tensor(
            float("nan"),
            device=DEVICE,
        )


    # ========================================================
    # TIMING / MEMORY
    # ========================================================

    torch.cuda.synchronize(
        DEVICE
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )


    peak_memory_mib = (
        torch.cuda.max_memory_allocated(
            DEVICE
        )
        / (1024.0 * 1024.0)
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()

    print("-" * 72)

    print(
        "OVERALL CONFIDENCE -> STABILITY RELATIONSHIP"
    )

    print("-" * 72)


    print(
        "Original RBG decisions:   "
        f"{flattened_margin.numel()}"
    )

    print(
        "Permutation comparisons:  "
        f"{agreement.numel()}"
    )

    print()

    print(
        "Mean confidence margin:   "
        f"{flattened_margin.mean().item():.6f}"
    )

    print(
        "Mean permutation stability:"
        f" {100.0 * overall_stability.item():.2f}%"
    )

    print(
        "Pearson corr(margin,"
        " stability): "
        f"{correlation.item():.6f}"
    )


    print()

    print("-" * 72)

    print(
        "STABILITY BY ORIGINAL CONFIDENCE"
    )

    print("-" * 72)


    for row in bin_rows:

        print(
            f"{row['confidence_bin']:<27}"
            f" n={row['num_original_rbg_decisions']:>7}"
            f"   stability="
            f"{row['mean_permutation_stability_percent']:>6.2f}%"
        )


    print()

    print(
        "Margin < 0.01 stability:  "
        f"{100.0 * low_stability.item():.2f}%"
    )

    print(
        "Margin >= 0.01 stability: "
        f"{100.0 * higher_stability.item():.2f}%"
    )

    print(
        "Margin >= 0.02 stability: "
        f"{100.0 * strong_stability.item():.2f}%"
    )


    print()

    print(
        "Elapsed:                  "
        f"{elapsed:.3f} s"
    )

    print(
        "Peak GPU memory:          "
        f"{peak_memory_mib:.1f} MiB"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary_row = {
        "checkpoint":
            str(checkpoint_path),

        "checkpoint_tti":
            checkpoint.get(
                "tti_index"
            ),

        "ppo_updates":
            checkpoint.get(
                "num_ppo_updates"
            ),

        "device":
            str(DEVICE),

        "gpu":
            torch.cuda.get_device_name(
                DEVICE
            ),

        "num_states":
            NUM_TEST_STATES,

        "permutations_per_state":
            PERMUTATIONS_PER_STATE,

        "num_original_rbg_decisions":
            flattened_margin.numel(),

        "num_permutation_comparisons":
            agreement.numel(),

        "mean_confidence_margin":
            float(
                flattened_margin
                .mean()
                .item()
            ),

        "mean_permutation_stability_percent":
            float(
                100.0
                * overall_stability.item()
            ),

        "pearson_margin_stability":
            float(
                correlation.item()
            ),

        "margin_lt_0p01_stability_percent":
            float(
                100.0
                * low_stability.item()
            ),

        "margin_ge_0p01_stability_percent":
            float(
                100.0
                * higher_stability.item()
            ),

        "margin_ge_0p02_stability_percent":
            float(
                100.0
                * strong_stability.item()
            ),

        "elapsed_s":
            elapsed,

        "peak_gpu_memory_mib":
            peak_memory_mib,
    }


    with RESULT_PATH.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_row.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            summary_row
        )


    with BIN_RESULT_PATH.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                bin_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            bin_rows
        )


    print()

    print(
        "Saved summary:            "
        f"{RESULT_PATH}"
    )

    print(
        "Saved confidence bins:    "
        f"{BIN_RESULT_PATH}"
    )

    print()

    print("=" * 72)

    print(
        "STRESS TEST 03 COMPLETE"
    )

    print("=" * 72)

    print()


if __name__ == "__main__":
    main()
