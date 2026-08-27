from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PPORewardConfig:
    """
    Configuration for the v3 1LDS-PPO reward.

    PAPER-SPECIFIED:
        k = 0.2

    PAPER-UNSPECIFIED:
        The numerical value / estimation procedure
        for G_max is not publicly provided.
    """

    geometric_mean_normalizer_bps: float
    pairing_reward_scale: float = 0.2

    def __post_init__(self) -> None:
        if (
            self.geometric_mean_normalizer_bps
            <= 0.0
        ):
            raise ValueError(
                "geometric_mean_normalizer_bps "
                "must be positive."
            )

        if self.pairing_reward_scale < 0.0:
            raise ValueError(
                "pairing_reward_scale cannot "
                "be negative."
            )


@dataclass(frozen=True)
class PPORewardData:
    """
    Paper-defined PPO reward quantities.

    geometric_mean_throughput_bps:
        Scalar G.

    normalized_geometric_mean:
        Scalar P = G / G_max.

    greedy_indicator:
        Shape [user_layer, RBG].

        +1:
            no better PF allocation was found.

        -1:
            a better PF allocation was found.

    reward_by_layer_rbg:
        Shape [user_layer, RBG].

        This module deliberately preserves the
        complete paper-defined reward tensor.
    """

    geometric_mean_throughput_bps: torch.Tensor

    normalized_geometric_mean: torch.Tensor

    greedy_indicator: torch.Tensor

    reward_by_layer_rbg: torch.Tensor


def compute_geometric_mean_throughput(
    *,
    throughput_bps: torch.Tensor,
    valid_ue_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute geometric-mean throughput G over real UEs.

    Args:
        throughput_bps:
            Shape [UE].

        valid_ue_mask:
            Shape [UE].

            True for real UEs and False for padded
            serving-layout positions.

    Returns:
        Scalar tensor containing G in bit/s.
    """

    if throughput_bps.ndim != 1:
        raise ValueError(
            "throughput_bps must have shape [UE]."
        )

    if tuple(
        valid_ue_mask.shape
    ) != tuple(
        throughput_bps.shape
    ):
        raise ValueError(
            "valid_ue_mask must have the same "
            "shape as throughput_bps."
        )

    if valid_ue_mask.dtype != torch.bool:
        raise ValueError(
            "valid_ue_mask must have dtype bool."
        )

    if (
        valid_ue_mask.device
        != throughput_bps.device
    ):
        raise ValueError(
            "valid_ue_mask and throughput_bps "
            "must be on the same device."
        )

    if not torch.is_floating_point(
        throughput_bps
    ):
        raise ValueError(
            "throughput_bps must use a "
            "floating-point dtype."
        )

    if not torch.isfinite(
        throughput_bps
    ).all():
        raise ValueError(
            "throughput_bps must contain only "
            "finite values."
        )

    if torch.any(
        throughput_bps < 0.0
    ):
        raise ValueError(
            "Throughput cannot be negative."
        )


    valid_throughput = (
        throughput_bps[
            valid_ue_mask
        ]
    )

    if valid_throughput.numel() == 0:
        raise ValueError(
            "At least one valid UE is required."
        )

    # Exact mathematical geometric mean:
    #
    # if any real UE has zero throughput,
    # the geometric mean is zero.
    #
    # IMPORTANT:
    # The paper explicitly replaces zero-valued
    # throughputs by one for FINAL EVALUATION
    # comparisons, but does not explicitly state
    # that the same replacement is used inside
    # the PPO training reward.
    #
    # Therefore we do not silently introduce that
    # evaluation-only rule here.
    if torch.any(
        valid_throughput == 0.0
    ):
        return torch.zeros(
            (),
            dtype=throughput_bps.dtype,
            device=throughput_bps.device,
        )

    # Use log-space rather than:
    #
    #     product(...) ** (1 / N)
    #
    # because throughput values in bit/s can be
    # large and their direct product can overflow.
    log_throughput = torch.log(
        valid_throughput
    )

    return torch.exp(
        log_throughput.mean()
    )


def build_greedy_indicator(
    *,
    better_allocation_exists: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert greedy-search results into the paper's
    v indicator.

    Args:
        better_allocation_exists:
            Shape [user_layer, RBG].

            True:
                greedy PF search found a better
                allocation.

            False:
                no better allocation was found.

    Returns:
        Shape [user_layer, RBG].

        -1 for True.
        +1 for False.
    """

    if better_allocation_exists.ndim != 2:
        raise ValueError(
            "better_allocation_exists must have "
            "shape [user_layer, RBG]."
        )

    if better_allocation_exists.dtype != torch.bool:
        raise ValueError(
            "better_allocation_exists must have "
            "dtype bool."
        )

    negative_one = torch.tensor(
        -1.0,
        dtype=dtype,
        device=(
            better_allocation_exists.device
        ),
    )

    positive_one = torch.tensor(
        1.0,
        dtype=dtype,
        device=(
            better_allocation_exists.device
        ),
    )

    return torch.where(
        better_allocation_exists,
        negative_one,
        positive_one,
    )


def compute_ppo_reward(
    *,
    throughput_bps: torch.Tensor,
    valid_ue_mask: torch.Tensor,
    better_allocation_exists: torch.Tensor,
    config: PPORewardConfig,
) -> PPORewardData:
    """
    Compute the v3 paper-defined 1LDS-PPO reward.

    Python user-layer index 0 corresponds to
    paper layer l = 1.

    PAPER:

        r[m, l] =
            P * v[m, l]     for l = 1
            k * v[m, l]     otherwise

        P = G / G_max

    The public paper writes v_m rather than
    v_{m,l}, but reward is computed after each
    user-layer iteration. We therefore preserve
    one indicator per user-layer/RBG decision.
    """

    if better_allocation_exists.shape[0] < 1:
        raise ValueError(
            "At least one user layer is required."
        )

    if better_allocation_exists.shape[1] < 1:
        raise ValueError(
            "At least one RBG is required."
        )

    if (
        better_allocation_exists.device
        != throughput_bps.device
    ):
        raise ValueError(
            "Greedy indicators and throughput "
            "must be on the same device."
        )

    geometric_mean = (
        compute_geometric_mean_throughput(
            throughput_bps=throughput_bps,
            valid_ue_mask=valid_ue_mask,
        )
    )

    normalized_geometric_mean = (
        geometric_mean
        / config.geometric_mean_normalizer_bps
    )

    greedy_indicator = (
        build_greedy_indicator(
            better_allocation_exists=(
                better_allocation_exists
            ),
            dtype=throughput_bps.dtype,
        )
    )


    reward_by_layer_rbg = (
        config.pairing_reward_scale
        * greedy_indicator
    )

    # PAPER layer l = 1 corresponds to
    # Python index 0.
    reward_by_layer_rbg[
        0,
        :,
    ] = (
        normalized_geometric_mean
        * greedy_indicator[
            0,
            :,
        ]
    )

    return PPORewardData(
        geometric_mean_throughput_bps=(
            geometric_mean
        ),
        normalized_geometric_mean=(
            normalized_geometric_mean
        ),
        greedy_indicator=(
            greedy_indicator
        ),
        reward_by_layer_rbg=(
            reward_by_layer_rbg
        ),
    )


