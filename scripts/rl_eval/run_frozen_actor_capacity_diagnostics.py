from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import time
from pathlib import Path

import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)
from oran_scheduler.rl.ppo_checkpoint import (
    load_ppo_model_checkpoint,
)


# ============================================================
# Reuse exactly the Set-1 controlled state bank and
# policy-comparison machinery.
#
# Set 2 already follows this same pattern.
# ============================================================

BASE_PATH = Path(__file__).with_name(
    "run_frozen_actor_diagnostics.py"
)

spec = importlib.util.spec_from_file_location(
    "rl_eval_base",
    BASE_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not load {BASE_PATH}"
    )

base = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    base
)


K = base.NUM_CANDIDATES
M = base.NUM_RBGS
F = base.FEATURES_PER_CANDIDATE
A = base.NUM_ACTIONS
S = base.STATE_SIZE

H = 32


# ============================================================
# LOCKED RL SET 3
#
# 6 evaluation families
# x
# 5 controlled state-bank seeds
# =
# 30 Slurm tasks
# ============================================================

EVALUATIONS = (
    "representation_capacity",
    "neuron_ablation",
    "parameter_noise",
    "compression",
    "quantization",
    "normalization_contract",
)

N_REPLICATES = 5

BASE_SEED = 202609240


DORMANCY_THRESHOLDS = (
    1e-4,
    1e-3,
    1e-2,
)


GROUP_ABLATION_SIZES = (
    1,
    2,
    4,
    8,
)

GROUP_ABLATION_REPEATS = 8


PARAMETER_NOISE_EPSILONS = (
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
)

PARAMETER_NOISE_REPEATS = 3


PRUNE_FRACTIONS = (
    0.10,
    0.25,
    0.50,
    0.75,
)

RANDOM_PRUNE_REPEATS = 3


LOW_RANK_VALUES = (
    24,
    16,
    8,
    4,
)


QUANT_BITS = (
    8,
    6,
    4,
    3,
)


NORMALIZATION_SCALES = (
    0.90,
    0.95,
    0.99,
    1.01,
    1.05,
    1.10,
)


FEATURE_GROUPS = {
    "past_throughput": (
        0,
        1,
    ),

    "buffer": (
        3,
        4,
    ),

    "wideband_cqi": (
        4,
        5,
    ),

    "subband_cqi": (
        5,
        5 + M,
    ),

    "cross_correlation": (
        5 + M,
        5 + 2 * M,
    ),
}


# ============================================================
# CLI / TASK MAPPING
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-id",
        type=int,
        default=int(
            os.getenv(
                "RL_EVAL3_TASK_ID",
                "0",
            )
        ),
    )

    return parser.parse_args()


def task_spec(
    task_id: int,
) -> tuple[
    str,
    int,
    int,
]:

    total = (
        len(EVALUATIONS)
        * N_REPLICATES
    )

    if not (
        0
        <= task_id
        < total
    ):
        raise ValueError(
            "task-id must be in "
            f"[0, {total - 1}]"
        )

    family = EVALUATIONS[
        task_id
        // N_REPLICATES
    ]

    replicate = (
        task_id
        % N_REPLICATES
    )

    seed = (
        BASE_SEED
        + replicate
    )

    return (
        family,
        replicate,
        seed,
    )


# ============================================================
# GENERIC HELPERS
# ============================================================

def resolve_checkpoint() -> Path:

    default = (
        Path.home()
        / "oran-deep-scheduler"
        / "experiments"
        / "checkpoints"
        / "paper_1lds_ppo_500tti.pt"
    )

    path = Path(
        os.getenv(
            "PPO_CHECKPOINT",
            str(default),
        )
    ).expanduser()

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    return path


def l2_norm(
    module: nn.Module,
) -> float:

    return math.sqrt(
        sum(
            float(
                parameter
                .detach()
                .pow(2)
                .sum()
                .item()
            )

            for parameter
            in module.parameters()
        )
    )


def mean_dict(
    items: list[
        dict[
            str,
            float,
        ]
    ],
) -> dict[
    str,
    float,
]:

    if not items:
        raise ValueError(
            "items cannot be empty"
        )

    return {
        key: float(
            sum(
                item[key]
                for item in items
            )
            / len(items)
        )

        for key
        in items[0]
    }


def compare(
    reference,
    changed,
) -> dict[
    str,
    float,
]:

    output = (
        base.compare_policies(
            reference=reference,
            changed=changed,
        )
    )

    return {
        **output,

        "rbg_flip_pct":
            (
                100.0
                - output[
                    "rbg_agreement_pct"
                ]
            ),

        "whole_schedule_change_pct":
            (
                100.0
                - output[
                    "whole_18rbg_agreement_pct"
                ]
            ),
    }


def outputs_from_logits(
    *,
    logits: torch.Tensor,
    action_mask: torch.Tensor,
):

    logits = logits.float()

    logits = logits.masked_fill(
        ~action_mask,
        torch.finfo(
            logits.dtype
        ).min,
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    top2 = probabilities.topk(
        k=2,
        dim=-1,
    ).values

    tiny = torch.finfo(
        probabilities.dtype
    ).tiny

    entropy = -(
        probabilities
        *
        probabilities
        .clamp_min(
            tiny
        )
        .log()
    ).sum(
        dim=-1
    )

    return {
        "probabilities":
            probabilities,

        "actions":
            probabilities.argmax(
                dim=-1
            ),

        "margin":
            (
                top2[
                    ...,
                    0,
                ]
                -
                top2[
                    ...,
                    1,
                ]
            ),

        "normalized_entropy":
            (
                entropy
                / math.log(
                    A
                )
            ),
    }


@torch.no_grad()
def hidden_activations(
    actor: OneLDSPPOActor,
    states: torch.Tensor,
):

    h1 = (
        actor.network[1](
            actor.network[0](
                states
            )
        )
    )

    h2 = (
        actor.network[3](
            actor.network[2](
                h1
            )
        )
    )

    return (
        h1,
        h2,
    )


@torch.no_grad()
def ablated_outputs(
    *,
    actor,
    states,
    action_mask,
    layer: int,
    indices: torch.Tensor,
):

    h1 = (
        actor.network[1](
            actor.network[0](
                states
            )
        )
    )

    if layer == 1:

        h1 = h1.clone()

        h1[
            :,
            indices,
        ] = 0.0


    h2 = (
        actor.network[3](
            actor.network[2](
                h1
            )
        )
    )

    if layer == 2:

        h2 = h2.clone()

        h2[
            :,
            indices,
        ] = 0.0


    if layer not in (
        1,
        2,
    ):
        raise ValueError(
            layer
        )


    logits = (
        actor.network[4](
            h2
        )
        .reshape(
            states.shape[0],
            M,
            A,
        )
    )

    return outputs_from_logits(
        logits=logits,
        action_mask=action_mask,
    )


# ============================================================
# REPRESENTATION METRICS
# ============================================================

def representation_metrics(
    x: torch.Tensor,
) -> dict[
    str,
    float,
]:

    x = (
        x
        .detach()
        .float()
    )

    mean_abs = (
        x.abs()
        .mean(
            dim=0
        )
    )

    normalized_mean_abs = (
        mean_abs
        /
        mean_abs
        .mean()
        .clamp_min(
            1e-12
        )
    )


    centered = (
        x
        -
        x.mean(
            dim=0,
            keepdim=True,
        )
    )


    singular_values = (
        torch.linalg.svdvals(
            centered
        )
    )


    p = (
        singular_values
        /
        singular_values
        .sum()
        .clamp_min(
            1e-12
        )
    )


    s2 = (
        singular_values
        .square()
    )


    std = centered.std(
        dim=0,
        unbiased=True,
    )

    valid = (
        std
        > 1e-8
    )


    if int(
        valid.sum().item()
    ) >= 2:

        z = (
            centered[
                :,
                valid,
            ]
            /
            std[
                valid
            ]
        )

        corr = (
            z.T
            @ z
            /
            float(
                max(
                    x.shape[0] - 1,
                    1,
                )
            )
        )

        offdiag = (
            ~torch.eye(
                corr.shape[0],
                dtype=torch.bool,
                device=corr.device,
            )
        )

        mean_abs_corr = float(
            corr[
                offdiag
            ]
            .abs()
            .mean()
            .item()
        )

    else:

        mean_abs_corr = (
            math.nan
        )


    output = {
        "mean_activation":
            float(
                x.mean().item()
            ),

        "mean_abs_activation":
            float(
                x.abs().mean().item()
            ),

        "zero_activation_pct":
            (
                100.0
                *
                float(
                    (
                        x
                        == 0
                    )
                    .float()
                    .mean()
                    .item()
                )
            ),

        "dead_neuron_pct":
            (
                100.0
                *
                float(
                    (
                        mean_abs
                        == 0
                    )
                    .float()
                    .mean()
                    .item()
                )
            ),

        "effective_rank":
            float(
                torch.exp(
                    -(
                        p
                        *
                        p.clamp_min(
                            1e-12
                        )
                        .log()
                    )
                    .sum()
                )
                .item()
            ),

        "stable_rank":
            float(
                (
                    s2.sum()
                    /
                    s2.max()
                    .clamp_min(
                        1e-12
                    )
                )
                .item()
            ),

        "participation_ratio":
            float(
                (
                    s2.sum()
                    .square()
                    /
                    s2.square()
                    .sum()
                    .clamp_min(
                        1e-12
                    )
                )
                .item()
            ),

        "mean_abs_offdiag_correlation":
            mean_abs_corr,

        "smallest_normalized_mean_abs":
            float(
                normalized_mean_abs
                .min()
                .item()
            ),

        "largest_normalized_mean_abs":
            float(
                normalized_mean_abs
                .max()
                .item()
            ),
    }


    for tau in DORMANCY_THRESHOLDS:

        token = (
            f"{tau:.0e}"
            .replace(
                "-",
                "m",
            )
            .replace(
                "+",
                "p",
            )
        )

        output[
            f"dormant_pct_tau_{token}"
        ] = (
            100.0
            *
            float(
                (
                    normalized_mean_abs
                    <= tau
                )
                .float()
                .mean()
                .item()
            )
        )


    return output


# ============================================================
# EXPERIMENT 1
# REPRESENTATION CAPACITY
# ============================================================

def eval_representation(
    actor,
    states,
):

    h1, h2 = hidden_activations(
        actor,
        states,
    )

    return [
        {
            "condition":
                "hidden_layer_1",

            "layer":
                1,

            **representation_metrics(
                h1
            ),
        },

        {
            "condition":
                "hidden_layer_2",

            "layer":
                2,

            **representation_metrics(
                h2
            ),
        },
    ]


# ============================================================
# EXPERIMENT 2
# HIDDEN-NEURON ABLATION
# ============================================================

def eval_neuron_ablation(
    actor,
    states,
    action_mask,
    reference,
    generator,
):

    rows = []


    # --------------------------------------------------------
    # Every neuron individually.
    # --------------------------------------------------------

    for layer in (
        1,
        2,
    ):

        for neuron in range(
            H
        ):

            indices = torch.tensor(
                [
                    neuron,
                ],
                device=states.device,
            )

            changed = ablated_outputs(
                actor=actor,
                states=states,
                action_mask=action_mask,
                layer=layer,
                indices=indices,
            )

            rows.append(
                {
                    "condition":
                        (
                            f"layer{layer}_"
                            f"neuron_{neuron:02d}"
                        ),

                    "ablation_type":
                        "single_neuron",

                    "layer":
                        layer,

                    "num_ablated_neurons":
                        1,

                    "neuron_index":
                        neuron,

                    "repeat_count":
                        1,

                    **compare(
                        reference,
                        changed,
                    ),
                }
            )


    # --------------------------------------------------------
    # Random simultaneous hidden-unit loss.
    # --------------------------------------------------------

    for layer in (
        1,
        2,
    ):

        for k in GROUP_ABLATION_SIZES:

            metrics = []

            for _ in range(
                GROUP_ABLATION_REPEATS
            ):

                indices = (
                    torch.randperm(
                        H,
                        device=states.device,
                        generator=generator,
                    )[
                        :k
                    ]
                )

                changed = ablated_outputs(
                    actor=actor,
                    states=states,
                    action_mask=action_mask,
                    layer=layer,
                    indices=indices,
                )

                metrics.append(
                    compare(
                        reference,
                        changed,
                    )
                )


            rows.append(
                {
                    "condition":
                        (
                            f"layer{layer}_"
                            f"random_k{k}"
                        ),

                    "ablation_type":
                        "random_group",

                    "layer":
                        layer,

                    "num_ablated_neurons":
                        k,

                    "neuron_index":
                        "",

                    "repeat_count":
                        GROUP_ABLATION_REPEATS,

                    **mean_dict(
                        metrics
                    ),
                }
            )


    return rows


# ============================================================
# EXPERIMENT 3
# PARAMETER-SPACE NOISE
# ============================================================

def selected_parameters(
    actor,
    group,
):

    mapping = {
        "hidden1":
            (
                actor.network[0],
            ),

        "hidden2":
            (
                actor.network[2],
            ),

        "output":
            (
                actor.network[4],
            ),

        "all":
            (
                actor.network[0],
                actor.network[2],
                actor.network[4],
            ),
    }


    return [
        parameter

        for module
        in mapping[
            group
        ]

        for parameter
        in module.parameters()
    ]


@torch.no_grad()
def add_parameter_noise(
    actor,
    group,
    epsilon,
    generator,
):

    for parameter in selected_parameters(
        actor,
        group,
    ):

        rms = (
            parameter
            .detach()
            .float()
            .square()
            .mean()
            .sqrt()
        )

        if float(
            rms.item()
        ) == 0.0:
            continue


        noise = torch.randn(
            parameter.shape,
            dtype=parameter.dtype,
            device=parameter.device,
            generator=generator,
        )

        parameter.add_(
            epsilon
            *
            rms.to(
                parameter.dtype
            )
            *
            noise
        )


def eval_parameter_noise(
    actor,
    states,
    action_mask,
    reference,
    generator,
):

    rows = []


    for group in (
        "hidden1",
        "hidden2",
        "output",
        "all",
    ):

        for epsilon in (
            PARAMETER_NOISE_EPSILONS
        ):

            metrics = []

            for _ in range(
                PARAMETER_NOISE_REPEATS
            ):

                changed_actor = (
                    copy.deepcopy(
                        actor
                    )
                )

                add_parameter_noise(
                    changed_actor,
                    group,
                    epsilon,
                    generator,
                )

                changed_actor.eval()

                changed = (
                    base.policy_outputs(
                        actor=changed_actor,
                        states=states,
                        action_mask=action_mask,
                    )
                )

                metrics.append(
                    compare(
                        reference,
                        changed,
                    )
                )


            rows.append(
                {
                    "condition":
                        (
                            f"{group}_"
                            f"eps_{epsilon:.4g}"
                        ),

                    "parameter_group":
                        group,

                    "epsilon":
                        epsilon,

                    "repeat_count":
                        PARAMETER_NOISE_REPEATS,

                    **mean_dict(
                        metrics
                    ),
                }
            )


    return rows


# ============================================================
# EXPERIMENT 4
# PRUNING + LOW-RANK COMPRESSION
# ============================================================

@torch.no_grad()
def prune_weights(
    actor,
    fraction,
    mode,
    generator,
):

    for index in (
        0,
        2,
        4,
    ):

        flat = (
            actor.network[
                index
            ]
            .weight
            .view(
                -1
            )
        )

        k = int(
            round(
                fraction
                * flat.numel()
            )
        )

        if k <= 0:
            continue


        if mode == "magnitude":

            indices = (
                flat
                .abs()
                .argsort()[
                    :k
                ]
            )

        elif mode == "random":

            indices = (
                torch.randperm(
                    flat.numel(),
                    device=flat.device,
                    generator=generator,
                )[
                    :k
                ]
            )

        else:

            raise ValueError(
                mode
            )


        flat[
            indices
        ] = 0.0


@torch.no_grad()
def low_rank_hidden(
    actor,
    rank,
):

    # Hidden affine maps only.
    #
    # output layer stays intact.

    for index in (
        0,
        2,
    ):

        weight = (
            actor.network[
                index
            ]
            .weight
        )

        u, s, vh = (
            torch.linalg.svd(
                weight
                .detach()
                .float(),
                full_matrices=False,
            )
        )

        r = min(
            rank,
            int(
                s.numel()
            ),
        )

        approximation = (
            (
                u[
                    :,
                    :r,
                ]
                *
                s[
                    :r
                ]
                .unsqueeze(
                    0
                )
            )
            @
            vh[
                :r,
                :,
            ]
        )

        weight.copy_(
            approximation.to(
                weight.dtype
            )
        )


def eval_compression(
    actor,
    states,
    action_mask,
    reference,
    generator,
):

    rows = []


    for fraction in (
        PRUNE_FRACTIONS
    ):

        # ----------------------------------------------------
        # Magnitude pruning.
        # ----------------------------------------------------

        changed_actor = (
            copy.deepcopy(
                actor
            )
        )

        prune_weights(
            changed_actor,
            fraction,
            "magnitude",
            generator,
        )

        changed = (
            base.policy_outputs(
                actor=changed_actor,
                states=states,
                action_mask=action_mask,
            )
        )

        rows.append(
            {
                "condition":
                    (
                        "magnitude_prune_"
                        f"{fraction:.2f}"
                    ),

                "compression_type":
                    "magnitude_pruning",

                "prune_fraction":
                    fraction,

                "rank":
                    "",

                "repeat_count":
                    1,

                **compare(
                    reference,
                    changed,
                ),
            }
        )


        # ----------------------------------------------------
        # Random-pruning control.
        # ----------------------------------------------------

        metrics = []

        for _ in range(
            RANDOM_PRUNE_REPEATS
        ):

            changed_actor = (
                copy.deepcopy(
                    actor
                )
            )

            prune_weights(
                changed_actor,
                fraction,
                "random",
                generator,
            )

            changed = (
                base.policy_outputs(
                    actor=changed_actor,
                    states=states,
                    action_mask=action_mask,
                )
            )

            metrics.append(
                compare(
                    reference,
                    changed,
                )
            )


        rows.append(
            {
                "condition":
                    (
                        "random_prune_"
                        f"{fraction:.2f}"
                    ),

                "compression_type":
                    "random_pruning",

                "prune_fraction":
                    fraction,

                "rank":
                    "",

                "repeat_count":
                    RANDOM_PRUNE_REPEATS,

                **mean_dict(
                    metrics
                ),
            }
        )


    # --------------------------------------------------------
    # SVD rank restriction.
    # --------------------------------------------------------

    for rank in LOW_RANK_VALUES:

        changed_actor = (
            copy.deepcopy(
                actor
            )
        )

        low_rank_hidden(
            changed_actor,
            rank,
        )

        changed = (
            base.policy_outputs(
                actor=changed_actor,
                states=states,
                action_mask=action_mask,
            )
        )

        rows.append(
            {
                "condition":
                    (
                        "hidden_svd_rank_"
                        f"{rank}"
                    ),

                "compression_type":
                    "hidden_low_rank_svd",

                "prune_fraction":
                    "",

                "rank":
                    rank,

                "repeat_count":
                    1,

                **compare(
                    reference,
                    changed,
                ),
            }
        )


    return rows


# ============================================================
# EXPERIMENT 5
# POST-TRAINING QUANTIZATION
# ============================================================

def fake_quant_symmetric(
    x,
    bits,
):

    qmax = (
        2 ** (
            bits - 1
        )
        - 1
    )

    max_abs = (
        x
        .detach()
        .abs()
        .max()
    )

    if float(
        max_abs.item()
    ) == 0.0:
        return x.clone()


    scale = (
        max_abs
        / float(
            qmax
        )
    )


    return (
        torch.round(
            x
            / scale
        )
        .clamp(
            -qmax,
            qmax,
        )
        *
        scale
    )


def fake_quant_unsigned(
    x,
    bits,
):

    qmax = (
        2 ** bits
        - 1
    )

    maximum = (
        x
        .detach()
        .max()
    )

    if float(
        maximum.item()
    ) <= 0.0:
        return x.clone()


    scale = (
        maximum
        / float(
            qmax
        )
    )


    return (
        torch.round(
            x
            / scale
        )
        .clamp(
            0,
            qmax,
        )
        *
        scale
    )


@torch.no_grad()
def quantize_weights(
    actor,
    bits,
):

    # Biases deliberately remain FP32.
    #
    # This isolates weight quantization.

    for index in (
        0,
        2,
        4,
    ):

        weight = (
            actor.network[
                index
            ]
            .weight
        )

        weight.copy_(
            fake_quant_symmetric(
                weight,
                bits,
            )
        )


@torch.no_grad()
def quantized_activation_outputs(
    actor,
    states,
    action_mask,
    bits,
):

    h1 = (
        actor.network[1](
            actor.network[0](
                states
            )
        )
    )

    h1 = (
        fake_quant_unsigned(
            h1,
            bits,
        )
    )


    h2 = (
        actor.network[3](
            actor.network[2](
                h1
            )
        )
    )

    h2 = (
        fake_quant_unsigned(
            h2,
            bits,
        )
    )


    logits = (
        actor.network[4](
            h2
        )
        .reshape(
            states.shape[0],
            M,
            A,
        )
    )


    return outputs_from_logits(
        logits=logits,
        action_mask=action_mask,
    )


def eval_quantization(
    actor,
    states,
    action_mask,
    reference,
):

    rows = []


    # --------------------------------------------------------
    # FP16 arithmetic.
    # --------------------------------------------------------

    actor16 = (
        copy.deepcopy(
            actor
        )
        .half()
    )

    with torch.no_grad():

        logits16 = (
            actor16(
                states.half()
            )
            .float()
        )


    rows.append(
        {
            "condition":
                "fp16_arithmetic",

            "quantization_type":
                "fp16_arithmetic",

            "bits":
                16,

            **compare(
                reference,

                outputs_from_logits(
                    logits=logits16,
                    action_mask=action_mask,
                ),
            ),
        }
    )


    # --------------------------------------------------------
    # Integer-like fake quantization.
    # --------------------------------------------------------

    for bits in QUANT_BITS:

        changed_actor = (
            copy.deepcopy(
                actor
            )
        )

        quantize_weights(
            changed_actor,
            bits,
        )


        weight_only = (
            base.policy_outputs(
                actor=changed_actor,
                states=states,
                action_mask=action_mask,
            )
        )


        rows.append(
            {
                "condition":
                    (
                        "weight_only_"
                        f"{bits}bit"
                    ),

                "quantization_type":
                    "weight_only",

                "bits":
                    bits,

                **compare(
                    reference,
                    weight_only,
                ),
            }
        )


        weight_activation = (
            quantized_activation_outputs(
                changed_actor,
                states,
                action_mask,
                bits,
            )
        )


        rows.append(
            {
                "condition":
                    (
                        "weight_and_hidden_"
                        "activation_"
                        f"{bits}bit"
                    ),

                "quantization_type":
                    (
                        "weight_and_hidden_"
                        "activation"
                    ),

                "bits":
                    bits,

                **compare(
                    reference,
                    weight_activation,
                ),
            }
        )


    return rows


# ============================================================
# EXPERIMENT 6
# NORMALIZATION-CONTRACT MISMATCH
# ============================================================

def eval_normalization(
    actor,
    states,
    action_mask,
    reference,
):

    rows = []


    original = (
        states.reshape(
            states.shape[0],
            K,
            F,
        )
    )


    groups = {
        **FEATURE_GROUPS,

        "all_continuous":
            None,
    }


    for group, bounds in (
        groups.items()
    ):

        for scale in (
            NORMALIZATION_SCALES
        ):

            changed_segments = (
                original.clone()
            )


            if bounds is None:

                for (
                    start,
                    end,
                ) in (
                    FEATURE_GROUPS
                    .values()
                ):

                    changed_segments[
                        :,
                        :,
                        start:end,
                    ] *= scale

            else:

                start, end = bounds

                changed_segments[
                    :,
                    :,
                    start:end,
                ] *= scale


            changed = (
                base.policy_outputs(
                    actor=actor,

                    states=(
                        changed_segments
                        .reshape(
                            states.shape[0],
                            S,
                        )
                    ),

                    action_mask=action_mask,
                )
            )


            rows.append(
                {
                    "condition":
                        (
                            f"{group}_"
                            f"scale_{scale:.2f}"
                        ),

                    "feature_group":
                        group,

                    "encoding_scale":
                        scale,

                    **compare(
                        reference,
                        changed,
                    ),
                }
            )


    return rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()


    (
        family,
        replicate,
        seed,
    ) = task_spec(
        args.task_id
    )


    if not torch.cuda.is_available():

        raise RuntimeError(
            "RL Set 3 requires "
            "the allocated GPU"
        )


    device = torch.device(
        "cuda:0"
    )

    torch.cuda.set_device(
        device
    )


    run_id = os.getenv(
        "RL_EVAL3_RUN_ID",
        "rl_eval3_manual",
    )


    output_root = Path(
        os.getenv(
            "RL_EVAL3_OUTPUT_ROOT",

            str(
                Path.home()
                / "oran_experiments"
                / "rl_eval3"
            ),
        )
    ).expanduser()


    num_states = int(
        os.getenv(
            "RL_EVAL3_NUM_STATES",
            "2048",
        )
    )


    checkpoint_path = (
        resolve_checkpoint()
    )


    generator = (
        torch.Generator(
            device=device
        )
    )

    generator.manual_seed(
        seed
    )


    actor = (
        OneLDSPPOActor(
            OneLDSPPOActorConfig()
        )
        .to(
            device
        )
    )


    checkpoint = (
        load_ppo_model_checkpoint(
            path=checkpoint_path,
            actor=actor,
            map_location=device,
        )
    )


    actor.eval()


    checkpoint_tti = (
        checkpoint.get(
            "tti_index"
        )
    )


    if (
        checkpoint_tti
        is not None
        and int(
            checkpoint_tti
        )
        < 400
    ):

        raise RuntimeError(
            "Refusing early checkpoint "
            f"tti={checkpoint_tti}"
        )


    torch.cuda.reset_peak_memory_stats(
        device
    )

    torch.cuda.synchronize(
        device
    )


    start = (
        time.perf_counter()
    )


    states = (
        base.build_states(
            num_states=num_states,
            device=device,
            generator=generator,
        )
    )


    action_mask = (
        base.build_action_mask(
            num_states=num_states,
            device=device,
        )
    )


    reference = (
        base.policy_outputs(
            actor=actor,
            states=states,
            action_mask=action_mask,
        )
    )


    if family == (
        "representation_capacity"
    ):

        rows = eval_representation(
            actor,
            states,
        )


    elif family == (
        "neuron_ablation"
    ):

        rows = eval_neuron_ablation(
            actor,
            states,
            action_mask,
            reference,
            generator,
        )


    elif family == (
        "parameter_noise"
    ):

        rows = eval_parameter_noise(
            actor,
            states,
            action_mask,
            reference,
            generator,
        )


    elif family == (
        "compression"
    ):

        rows = eval_compression(
            actor,
            states,
            action_mask,
            reference,
            generator,
        )


    elif family == (
        "quantization"
    ):

        rows = eval_quantization(
            actor,
            states,
            action_mask,
            reference,
        )


    elif family == (
        "normalization_contract"
    ):

        rows = eval_normalization(
            actor,
            states,
            action_mask,
            reference,
        )


    else:

        raise AssertionError(
            family
        )


    torch.cuda.synchronize(
        device
    )


    elapsed = (
        time.perf_counter()
        - start
    )


    peak_gpu_mib = (
        torch.cuda
        .max_memory_allocated(
            device
        )
        /
        (
            1024.0
            ** 2
        )
    )


    common = {
        "run_id":
            run_id,

        "task_id":
            args.task_id,

        "evaluation":
            family,

        "replicate":
            replicate,

        "seed":
            seed,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_tti":
            checkpoint_tti,

        "ppo_updates":
            checkpoint.get(
                "num_ppo_updates"
            ),

        "actor_norm":
            l2_norm(
                actor
            ),

        "device":
            str(
                device
            ),

        "num_states":
            num_states,

        "elapsed_s":
            elapsed,

        "peak_gpu_mib":
            peak_gpu_mib,
    }


    rows = [
        {
            **common,
            **row,
        }

        for row
        in rows
    ]


    output_path = (
        output_root
        / run_id
        /
        (
            f"task_{args.task_id:02d}_"
            f"{family}_"
            f"seed_{seed}.csv"
        )
    )


    base.write_csv(
        output_path,
        rows,
    )


    print(
        "=" * 78
    )

    print(
        "FROZEN PPO ACTOR "
        "RL EVALUATION — SET 3"
    )

    print(
        "=" * 78
    )

    print(
        f"Run ID:          {run_id}"
    )

    print(
        f"Task:            {args.task_id}"
    )

    print(
        f"Evaluation:      {family}"
    )

    print(
        f"Replicate:       {replicate}"
    )

    print(
        f"Seed:            {seed}"
    )

    print(
        f"Checkpoint TTI:  {checkpoint_tti}"
    )

    print(
        "PPO updates:     "
        f"{checkpoint.get('num_ppo_updates')}"
    )

    print(
        "Actor norm:      "
        f"{l2_norm(actor):.9f}"
    )

    print(
        f"States:          {num_states}"
    )

    print(
        f"Rows written:    {len(rows)}"
    )

    print(
        f"Elapsed:         {elapsed:.3f} s"
    )

    print(
        "Peak GPU:        "
        f"{peak_gpu_mib:.1f} MiB"
    )

    print(
        f"Output:          {output_path}"
    )

    print(
        "=" * 78
    )

    print(
        "RL_POLICY_EVAL3_TASK_PASS"
    )


if __name__ == "__main__":
    main()
