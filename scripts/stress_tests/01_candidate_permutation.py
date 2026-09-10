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
        "This stress test is configured to use cuda:0, "
        "but CUDA is not available."
    )


# ============================================================
# PAPER / REPRODUCTION DIMENSIONS
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
#
# This is intentionally much larger than the previous script
# so the allocated GPU is actually exercised.
#
# 8192 states
# x 8 candidate permutations/state
# =
# 65,536 permutation comparisons
#
# Each comparison has 18 RBG branches.
# ============================================================

NUM_TEST_STATES = 8192

PERMUTATIONS_PER_STATE = 8

RANDOM_SEED = 20260910


RESULT_PATH = Path(
    "experiments/stress_tests/"
    "01_candidate_permutation_summary.csv"
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
        "Could not find either expected PPO checkpoint."
    )


# ============================================================
# SYNTHETIC BUT PHYSICALLY PLAUSIBLE NORMALIZED STATES
#
# Candidate feature layout:
#
#   0     past average throughput
#   1     rank
#   2     allocated RBG count
#   3     DL buffer
#   4     wideband CQI
#   5:23  subband CQI for 18 RBGs
#   23:41 spatial cross-correlation for 18 RBGs
#
# This matches the current 1LDS representation:
#
#   10 x (5 + 2*18) = 410
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
    # 0. Past averaged throughput
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
    # 1. Rank
    #
    # rank 1 -> normalized 0.5
    # rank 2 -> normalized 1.0
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
    # 2. Already allocated RBG count
    # --------------------------------------------------------

    allocated_rbgs = torch.randint(
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
        allocated_rbgs.float()
        / float(NUM_RBGS)
    )


    # --------------------------------------------------------
    # 3. DL buffer
    #
    # Rough synthetic mix:
    # ~50% full-buffer-like candidates,
    # others finite-buffer.
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
    # 4. Wideband CQI
    # --------------------------------------------------------

    wideband_cqi = torch.rand(
        (
            NUM_TEST_STATES,
            NUM_CANDIDATES,
        ),
        device=DEVICE,
        generator=generator,
    )

    segments[:, :, 4] = (
        wideband_cqi
    )


    # --------------------------------------------------------
    # 5:23. Subband CQI
    #
    # Correlated with WB CQI rather than completely random.
    # --------------------------------------------------------

    subband_noise = (
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
    )

    subband_cqi = (
        wideband_cqi.unsqueeze(-1)
        + subband_noise
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
    # 23:41. Spatial cross-correlation
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


    return segments


# ============================================================
# JENSEN-SHANNON DIVERGENCE
# ============================================================

def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:

    eps = 1.0e-12

    p = p.clamp_min(
        eps
    )

    q = q.clamp_min(
        eps
    )

    midpoint = (
        0.5
        * (
            p
            + q
        )
    )

    kl_p = (
        p
        * (
            torch.log(p)
            - torch.log(midpoint)
        )
    ).sum(
        dim=-1
    )

    kl_q = (
        q
        * (
            torch.log(q)
            - torch.log(midpoint)
        )
    ).sum(
        dim=-1
    )

    return (
        0.5
        * kl_p
        +
        0.5
        * kl_q
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()

    print("=" * 72)

    print(
        "STRESS TEST 01: "
        "CANDIDATE PERMUTATION ROBUSTNESS"
    )

    print("=" * 72)

    print(
        f"Device:                   "
        f"{DEVICE}"
    )

    print(
        "GPU:                      "
        f"{torch.cuda.get_device_name(DEVICE)}"
    )


    # --------------------------------------------------------
    # Deterministic CUDA random generator
    # --------------------------------------------------------

    generator = torch.Generator(
        device=DEVICE
    )

    generator.manual_seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


    # --------------------------------------------------------
    # Load actor
    # --------------------------------------------------------

    checkpoint_path = (
        resolve_checkpoint()
    )

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

    checkpoint = (
        load_ppo_model_checkpoint(
            path=checkpoint_path,
            actor=actor,
            map_location=DEVICE,
        )
    )

    actor.eval()


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
        "Permutations / state:     "
        f"{PERMUTATIONS_PER_STATE}"
    )


    # --------------------------------------------------------
    # Reset memory statistics
    # --------------------------------------------------------

    torch.cuda.reset_peak_memory_stats(
        DEVICE
    )

    torch.cuda.synchronize(
        DEVICE
    )

    start_time = (
        time.perf_counter()
    )


    # ========================================================
    # BUILD ORIGINAL STATES
    # ========================================================

    segments = build_states(
        generator=generator
    )

    original_states = (
        segments.reshape(
            NUM_TEST_STATES,
            STATE_SIZE,
        )
    )


    # ========================================================
    # BUILD RANDOM PERMUTATIONS ON GPU
    #
    # new_to_old[s, p, j]
    #
    # tells us which OLD candidate is placed in NEW slot j.
    #
    # argsort(random numbers) gives a random permutation.
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


    # ========================================================
    # COMPUTE INVERSE PERMUTATION
    #
    # old_to_new[s,p,old] = new slot
    # ========================================================

    old_to_new = torch.empty_like(
        new_to_old
    )

    new_indices = torch.arange(
        NUM_CANDIDATES,
        dtype=torch.long,
        device=DEVICE,
    )

    new_indices = (
        new_indices
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
        src=new_indices,
    )


    # ========================================================
    # PERMUTE COMPLETE 41-FEATURE CANDIDATE SEGMENTS
    #
    # Original:
    #
    # [state, candidate, feature]
    #
    # Expanded:
    #
    # [state, permutation, candidate, feature]
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

    gather_indices = (
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
        index=gather_indices,
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
    # ACTOR INFERENCE — GPU
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


        permuted_logits = actor(
            permuted_states
        )

        permuted_probabilities = (
            torch.softmax(
                permuted_logits,
                dim=-1,
            )
        )


    # --------------------------------------------------------
    # Shape:
    #
    # [state, permutation, RBG, action]
    # --------------------------------------------------------

    permuted_probabilities = (
        permuted_probabilities.reshape(
            NUM_TEST_STATES,
            PERMUTATIONS_PER_STATE,
            NUM_RBGS,
            NUM_ACTIONS,
        )
    )


    # ========================================================
    # RESTORE OUTPUT ACTION COLUMNS TO ORIGINAL PHYSICAL UE
    # ORDER
    #
    # Candidate action columns move.
    #
    # Action 10 = NO ALLOCATION and does NOT move.
    # ========================================================

    permuted_candidate_probs = (
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

    restored_candidate_probs = (
        torch.gather(
            permuted_candidate_probs,
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
            restored_candidate_probs,
            no_allocation_probability,
        ),
        dim=-1,
    )


    # ========================================================
    # ORIGINAL PROBABILITIES REPEATED FOR EVERY PERMUTATION
    # ========================================================

    reference_probabilities = (
        original_probabilities
        .unsqueeze(1)
        .expand(
            NUM_TEST_STATES,
            PERMUTATIONS_PER_STATE,
            NUM_RBGS,
            NUM_ACTIONS,
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

    probability_difference = (
        reference_probabilities
        - restored_probabilities
    ).abs()


    # --------------------------------------------------------
    # Total variation distance
    #
    # TV(P,Q) = 0.5 * sum |P-Q|
    # --------------------------------------------------------

    tv_distance = (
        0.5
        * probability_difference.sum(
            dim=-1
        )
    )


    jsd = js_divergence(
        reference_probabilities,
        restored_probabilities,
    )


    original_argmax = (
        reference_probabilities.argmax(
            dim=-1
        )
    )

    restored_argmax = (
        restored_probabilities.argmax(
            dim=-1
        )
    )

    agreement = (
        original_argmax
        == restored_argmax
    )


    # --------------------------------------------------------
    # Does the COMPLETE 18-RBG greedy decision stay identical?
    # --------------------------------------------------------

    whole_decision_agreement = (
        agreement.all(
            dim=-1
        )
    )


    # --------------------------------------------------------
    # Max probability shift for each state/permutation pair
    # --------------------------------------------------------

    max_probability_difference = (
        probability_difference
        .amax(
            dim=(-1, -2)
        )
    )


    # ========================================================
    # REDUCE ON GPU
    # ========================================================

    mean_tv = (
        tv_distance.mean()
    )

    p95_tv = torch.quantile(
        tv_distance.flatten(),
        0.95,
    )

    max_tv = (
        tv_distance.max()
    )


    mean_jsd = (
        jsd.mean()
    )

    p95_jsd = torch.quantile(
        jsd.flatten(),
        0.95,
    )

    max_jsd = (
        jsd.max()
    )


    mean_max_probability_difference = (
        max_probability_difference.mean()
    )

    worst_probability_difference = (
        max_probability_difference.max()
    )


    rbg_argmax_agreement = (
        agreement.float().mean()
        * 100.0
    )

    whole_18rbg_agreement = (
        whole_decision_agreement
        .float()
        .mean()
        * 100.0
    )


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
        / (
            1024.0
            * 1024.0
        )
    )


    # ========================================================
    # COPY ONLY FINAL SCALARS TO CPU
    # ========================================================

    results = {
        "mean_tv_distance":
            float(mean_tv.item()),

        "p95_tv_distance":
            float(p95_tv.item()),

        "max_tv_distance":
            float(max_tv.item()),

        "mean_jsd":
            float(mean_jsd.item()),

        "p95_jsd":
            float(p95_jsd.item()),

        "max_jsd":
            float(max_jsd.item()),

        "mean_max_probability_difference":
            float(
                mean_max_probability_difference.item()
            ),

        "worst_probability_difference":
            float(
                worst_probability_difference.item()
            ),

        "rbg_argmax_agreement_percent":
            float(
                rbg_argmax_agreement.item()
            ),

        "whole_18rbg_agreement_percent":
            float(
                whole_18rbg_agreement.item()
            ),
    }


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print("-" * 72)

    print(
        "RESULTS AFTER RESTORING "
        "PHYSICAL CANDIDATE IDENTITIES"
    )

    print("-" * 72)


    print(
        "State/permutation pairs:  "
        f"{NUM_TEST_STATES * PERMUTATIONS_PER_STATE}"
    )

    print(
        "RBG comparisons:          "
        f"{NUM_TEST_STATES * PERMUTATIONS_PER_STATE * NUM_RBGS}"
    )


    print()

    print(
        "Mean TV distance:         "
        f"{results['mean_tv_distance']:.8f}"
    )

    print(
        "95th %-ile TV distance:  "
        f"{results['p95_tv_distance']:.8f}"
    )

    print(
        "Worst TV distance:        "
        f"{results['max_tv_distance']:.8f}"
    )


    print()

    print(
        "Mean JSD:                 "
        f"{results['mean_jsd']:.8f}"
    )

    print(
        "95th %-ile JSD:           "
        f"{results['p95_jsd']:.8f}"
    )

    print(
        "Worst JSD:                "
        f"{results['max_jsd']:.8f}"
    )


    print()

    print(
        "Mean max |prob diff|:     "
        f"{results['mean_max_probability_difference']:.8f}"
    )

    print(
        "Worst max |prob diff|:    "
        f"{results['worst_probability_difference']:.8f}"
    )


    print()

    print(
        "RBG argmax agreement:     "
        f"{results['rbg_argmax_agreement_percent']:.2f}%"
    )

    print(
        "Whole 18-RBG agreement:   "
        f"{results['whole_18rbg_agreement_percent']:.2f}%"
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


    print()

    print(
        "Ideal permutation-equivariant behavior:"
    )

    print(
        "  TV = 0"
    )

    print(
        "  JSD = 0"
    )

    print(
        "  argmax agreement = 100%"
    )


    # ========================================================
    # SAVE SUMMARY CSV
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

        "num_test_states":
            NUM_TEST_STATES,

        "permutations_per_state":
            PERMUTATIONS_PER_STATE,

        "state_permutation_pairs":
            (
                NUM_TEST_STATES
                * PERMUTATIONS_PER_STATE
            ),

        "rbg_comparisons":
            (
                NUM_TEST_STATES
                * PERMUTATIONS_PER_STATE
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
        "STRESS TEST 01 COMPLETE"
    )

    print("=" * 72)

    print()


if __name__ == "__main__":
    main()
