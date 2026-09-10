from __future__ import annotations

import csv
import math
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
# DIMENSIONS
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

NUM_TEST_STATES = 131072

RANDOM_SEED = 20260911


RESULT_PATH = Path(
    "experiments/stress_tests/"
    "02_policy_confidence_summary.csv"
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


    # --------------------------------------------------------
    # Past average throughput
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Rank: normalized rank 1 or 2
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Allocated RBG count
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Buffer
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Wideband CQI
    # --------------------------------------------------------

    wb_cqi = torch.rand(
        (
            NUM_TEST_STATES,
            NUM_CANDIDATES,
        ),
        device=DEVICE,
        generator=generator,
    )

    segments[:, :, 4] = (
        wb_cqi
    )


    # --------------------------------------------------------
    # Subband CQI
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Cross-correlation
    # --------------------------------------------------------

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


    return segments.reshape(
        NUM_TEST_STATES,
        STATE_SIZE,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()

    print("=" * 72)

    print(
        "STRESS TEST 02: "
        "POLICY CONFIDENCE / ACTION AMBIGUITY"
    )

    print("=" * 72)


    checkpoint_path = (
        resolve_checkpoint()
    )


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
        "Number of states:         "
        f"{NUM_TEST_STATES}"
    )

    print(
        "RBG decisions:            "
        f"{NUM_TEST_STATES * NUM_RBGS}"
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
    # RNG
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
    # BUILD STATES
    # ========================================================

    states = build_states(
        generator=generator
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.inference_mode():

        logits = actor(
            states
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )


    # ========================================================
    # TOP-1 / TOP-2
    # ========================================================

    top2_values, top2_indices = torch.topk(
        probabilities,
        k=2,
        dim=-1,
    )


    top1_probability = (
        top2_values[..., 0]
    )

    top2_probability = (
        top2_values[..., 1]
    )

    probability_margin = (
        top1_probability
        - top2_probability
    )


    argmax_action = (
        top2_indices[..., 0]
    )


    # ========================================================
    # ENTROPY
    # ========================================================

    eps = 1.0e-12

    entropy = -(
        probabilities
        * torch.log(
            probabilities.clamp_min(
                eps
            )
        )
    ).sum(
        dim=-1
    )


    maximum_entropy = math.log(
        NUM_ACTIONS
    )


    normalized_entropy = (
        entropy
        / maximum_entropy
    )


    # ========================================================
    # CONFIDENCE THRESHOLDS
    # ========================================================

    margin_lt_001 = (
        probability_margin
        < 0.01
    ).float().mean()


    margin_lt_002 = (
        probability_margin
        < 0.02
    ).float().mean()


    margin_lt_005 = (
        probability_margin
        < 0.05
    ).float().mean()


    top1_lt_015 = (
        top1_probability
        < 0.15
    ).float().mean()


    top1_lt_020 = (
        top1_probability
        < 0.20
    ).float().mean()


    normalized_entropy_gt_090 = (
        normalized_entropy
        > 0.90
    ).float().mean()


    normalized_entropy_gt_095 = (
        normalized_entropy
        > 0.95
    ).float().mean()


    normalized_entropy_gt_099 = (
        normalized_entropy
        > 0.99
    ).float().mean()


    # --------------------------------------------------------
    # Action index 10 = no allocation
    # --------------------------------------------------------

    no_allocation_argmax = (
        argmax_action
        == NUM_CANDIDATES
    ).float().mean()


    # ========================================================
    # QUANTILES
    # ========================================================

    flattened_margin = (
        probability_margin.flatten()
    )

    flattened_top1 = (
        top1_probability.flatten()
    )

    flattened_entropy = (
        normalized_entropy.flatten()
    )


    margin_q05 = torch.quantile(
        flattened_margin,
        0.05,
    )

    margin_q50 = torch.quantile(
        flattened_margin,
        0.50,
    )

    margin_q95 = torch.quantile(
        flattened_margin,
        0.95,
    )


    top1_q05 = torch.quantile(
        flattened_top1,
        0.05,
    )

    top1_q50 = torch.quantile(
        flattened_top1,
        0.50,
    )

    top1_q95 = torch.quantile(
        flattened_top1,
        0.95,
    )


    entropy_q05 = torch.quantile(
        flattened_entropy,
        0.05,
    )

    entropy_q50 = torch.quantile(
        flattened_entropy,
        0.50,
    )

    entropy_q95 = torch.quantile(
        flattened_entropy,
        0.95,
    )


    # ========================================================
    # TIMING
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
    # RESULTS
    # ========================================================

    results = {

        "mean_top1_probability":
            float(
                top1_probability
                .mean()
                .item()
            ),

        "median_top1_probability":
            float(
                top1_q50.item()
            ),

        "p05_top1_probability":
            float(
                top1_q05.item()
            ),

        "p95_top1_probability":
            float(
                top1_q95.item()
            ),

        "mean_probability_margin":
            float(
                probability_margin
                .mean()
                .item()
            ),

        "median_probability_margin":
            float(
                margin_q50.item()
            ),

        "p05_probability_margin":
            float(
                margin_q05.item()
            ),

        "p95_probability_margin":
            float(
                margin_q95.item()
            ),

        "mean_normalized_entropy":
            float(
                normalized_entropy
                .mean()
                .item()
            ),

        "median_normalized_entropy":
            float(
                entropy_q50.item()
            ),

        "p05_normalized_entropy":
            float(
                entropy_q05.item()
            ),

        "p95_normalized_entropy":
            float(
                entropy_q95.item()
            ),

        "margin_lt_0p01_percent":
            float(
                100.0
                * margin_lt_001.item()
            ),

        "margin_lt_0p02_percent":
            float(
                100.0
                * margin_lt_002.item()
            ),

        "margin_lt_0p05_percent":
            float(
                100.0
                * margin_lt_005.item()
            ),

        "top1_lt_0p15_percent":
            float(
                100.0
                * top1_lt_015.item()
            ),

        "top1_lt_0p20_percent":
            float(
                100.0
                * top1_lt_020.item()
            ),

        "normalized_entropy_gt_0p90_percent":
            float(
                100.0
                * normalized_entropy_gt_090.item()
            ),

        "normalized_entropy_gt_0p95_percent":
            float(
                100.0
                * normalized_entropy_gt_095.item()
            ),

        "normalized_entropy_gt_0p99_percent":
            float(
                100.0
                * normalized_entropy_gt_099.item()
            ),

        "no_allocation_argmax_percent":
            float(
                100.0
                * no_allocation_argmax.item()
            ),
    }


    # ========================================================
    # PRINT
    # ========================================================

    print()

    print("-" * 72)

    print(
        "TOP-ACTION CONFIDENCE"
    )

    print("-" * 72)


    print(
        "Mean top-1 probability:   "
        f"{results['mean_top1_probability']:.6f}"
    )

    print(
        "Median top-1 probability: "
        f"{results['median_top1_probability']:.6f}"
    )

    print(
        "5th %-ile top-1 prob.:    "
        f"{results['p05_top1_probability']:.6f}"
    )

    print(
        "95th %-ile top-1 prob.:   "
        f"{results['p95_top1_probability']:.6f}"
    )


    print()

    print(
        "Mean top1-top2 margin:    "
        f"{results['mean_probability_margin']:.6f}"
    )

    print(
        "Median margin:            "
        f"{results['median_probability_margin']:.6f}"
    )

    print(
        "5th %-ile margin:         "
        f"{results['p05_probability_margin']:.6f}"
    )

    print(
        "95th %-ile margin:        "
        f"{results['p95_probability_margin']:.6f}"
    )


    print()

    print(
        "Mean normalized entropy: "
        f"{results['mean_normalized_entropy']:.6f}"
    )

    print(
        "Median norm. entropy:     "
        f"{results['median_normalized_entropy']:.6f}"
    )

    print(
        "5th %-ile norm. entropy:  "
        f"{results['p05_normalized_entropy']:.6f}"
    )

    print(
        "95th %-ile norm. entropy: "
        f"{results['p95_normalized_entropy']:.6f}"
    )


    print()

    print("-" * 72)

    print(
        "AMBIGUOUS-DECISION FRACTIONS"
    )

    print("-" * 72)


    print(
        "Margin < 0.01:            "
        f"{results['margin_lt_0p01_percent']:.2f}%"
    )

    print(
        "Margin < 0.02:            "
        f"{results['margin_lt_0p02_percent']:.2f}%"
    )

    print(
        "Margin < 0.05:            "
        f"{results['margin_lt_0p05_percent']:.2f}%"
    )

    print(
        "Top-1 probability < .15: "
        f"{results['top1_lt_0p15_percent']:.2f}%"
    )

    print(
        "Top-1 probability < .20: "
        f"{results['top1_lt_0p20_percent']:.2f}%"
    )

    print(
        "Norm. entropy > .90:     "
        f"{results['normalized_entropy_gt_0p90_percent']:.2f}%"
    )

    print(
        "Norm. entropy > .95:     "
        f"{results['normalized_entropy_gt_0p95_percent']:.2f}%"
    )

    print(
        "Norm. entropy > .99:     "
        f"{results['normalized_entropy_gt_0p99_percent']:.2f}%"
    )

    print(
        "No-allocation is argmax: "
        f"{results['no_allocation_argmax_percent']:.2f}%"
    )


    print()

    print(
        "Uniform-policy reference:"
    )

    print(
        "  probability/action =    "
        f"{1.0 / NUM_ACTIONS:.6f}"
    )

    print(
        "  normalized entropy =    "
        "1.000000"
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
    # CSV
    # ========================================================

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    row = {

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

        "num_rbg_decisions":
            (
                NUM_TEST_STATES
                * NUM_RBGS
            ),

        **results,

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
                row.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            row
        )


    print()

    print(
        "Saved summary:            "
        f"{RESULT_PATH}"
    )

    print()

    print("=" * 72)

    print(
        "STRESS TEST 02 COMPLETE"
    )

    print("=" * 72)

    print()


if __name__ == "__main__":
    main()
