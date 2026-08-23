from dataclasses import dataclass

import torch

from sionna.phy.mimo import (
    rzf_precoding_matrix,
)

@dataclass
class MRCPostSINRData:
    """
    Post-MRC quantities for rank-1 streams.

    For K scheduled UEs:

        precoded_channel:
            [..., K, num_rx_ant, K]

        mrc_combiner:
            [..., K, num_rx_ant]

        desired_power:
            [..., K]

        intra_cell_interference_power:
            [..., K]

        inter_cell_interference_power:
            [..., K]

        noise_power:
            [..., K]

        sinr_linear:
            [..., K]
    """

    precoded_channel: torch.Tensor

    mrc_combiner: torch.Tensor

    desired_power: torch.Tensor

    intra_cell_interference_power: torch.Tensor

    inter_cell_interference_power: torch.Tensor

    noise_power: torch.Tensor

    sinr_linear: torch.Tensor

def build_rank1_effective_channel(
    h: torch.Tensor,
    rx_combiner: torch.Tensor,
) -> torch.Tensor:
    """
    Form one effective transmit-side channel per scheduled UE.

    Args:
        h:
            [..., K, num_rx_ant, num_tx_ant]

        rx_combiner:
            [..., K, num_rx_ant]

    Returns:
        effective_channel:
            [..., K, num_tx_ant]

    Mathematical operation:

        h_eff,k = f_k^H H_k

    The combiner is normalized internally.

    Important:
        This function does NOT decide how f_k is obtained.
        That belongs to the later CSI/PMI/RI module.
    """

    if h.ndim < 3:
        raise ValueError(
            "h must end with "
            "[UE, RX_ant, TX_ant]."
        )

    if rx_combiner.shape != h.shape[:-1]:
        raise ValueError(
            "rx_combiner must have shape matching "
            "h without the TX-antenna dimension."
        )

    combiner_norm = torch.linalg.vector_norm(
        rx_combiner,
        dim=-1,
        keepdim=True,
    )

    if torch.any(combiner_norm <= 0):
        raise ValueError(
            "rx_combiner contains a zero-norm vector."
        )

    normalized_combiner = (
        rx_combiner
        / combiner_norm
    )

    effective_channel = torch.einsum(
        "...kr,...krm->...km",
        normalized_combiner.conj(),
        h,
    )

    return effective_channel


def compute_rzf_matrix(
    effective_channel: torch.Tensor,
    alpha: float,
    precision: str = "single",
) -> torch.Tensor:
    """
    Compute the RZF transmit-precoding matrix.

    Args:
        effective_channel:
            [..., K, num_tx_ant]

        alpha:
            RZF regularization parameter.

    Returns:
        precoding_matrix:
            [..., num_tx_ant, K]

    Important:
        The Bell Labs paper specifies RZF but does not publicly
        provide the exact regularization rule/value.

        Therefore alpha must be provided explicitly.
    """

    if effective_channel.ndim < 2:
        raise ValueError(
            "effective_channel must end with "
            "[stream, TX_ant]."
        )

    num_streams = (
        effective_channel.shape[-2]
    )

    num_tx_ant = (
        effective_channel.shape[-1]
    )

    if num_streams > num_tx_ant:
        raise ValueError(
            "Number of streams cannot exceed "
            "the number of TX antennas."
        )

    if alpha < 0.0:
        raise ValueError(
            "RZF alpha cannot be negative."
        )

    if not torch.isfinite(
        effective_channel.real
    ).all():
        raise ValueError(
            "effective_channel contains "
            "non-finite real values."
        )

    if not torch.isfinite(
        effective_channel.imag
    ).all():
        raise ValueError(
            "effective_channel contains "
            "non-finite imaginary values."
        )
    
    precoding_matrix = (
        rzf_precoding_matrix(
            effective_channel,
            alpha=alpha,
            precision=precision,
        )
    )

    return precoding_matrix


alpha = 0.1

def _broadcast_per_stream_value(
    value: float | torch.Tensor,
    reference: torch.Tensor,
    num_streams: int,
) -> torch.Tensor:
    """
    Convert a scalar or tensor into one value per stream.

    reference has shape:

        [..., M, K]

    Output:

        [..., K]
    """

    target_shape = (
        reference.shape[:-2]
        + (num_streams,)
    )

    value_tensor = torch.as_tensor(
        value,
        dtype=reference.real.dtype,
        device=reference.device,
    )

    return torch.broadcast_to(
        value_tensor,
        target_shape,
    )


def compute_rank1_mrc_post_sinr(
    h: torch.Tensor,
    precoding_matrix: torch.Tensor,
    stream_power_w: float | torch.Tensor,
    noise_power_w: float | torch.Tensor,
    inter_cell_covariance: torch.Tensor | None = None,
) -> MRCPostSINRData:
    """
    Compute post-MRC SINR for one rank-1 stream per scheduled UE.

    Args:
        h:
            [..., K, num_rx_ant, num_tx_ant]

        precoding_matrix:
            [..., num_tx_ant, K]

        stream_power_w:
            Scalar or [..., K].

        noise_power_w:
            AWGN power per RX branch.
            Scalar or broadcastable to [..., K].

        inter_cell_covariance:
            Optional interference covariance:

                [..., K, num_rx_ant, num_rx_ant]

    This computes the actual receive-side MRC vector from each
    UE's desired precoded spatial channel.
    """

    if h.ndim < 3:
        raise ValueError(
            "h must end with "
            "[UE, RX_ant, TX_ant]."
        )

    num_users = h.shape[-3]
    num_rx_ant = h.shape[-2]
    num_tx_ant = h.shape[-1]

    if precoding_matrix.shape[-2] != num_tx_ant:
        raise ValueError(
            "Precoder TX-antenna dimension does not "
            "match the channel."
        )

    if precoding_matrix.shape[-1] != num_users:
        raise ValueError(
            "Current rank-1 implementation requires "
            "one stream per scheduled UE."
        )

    expanded_precoder = (
        precoding_matrix.unsqueeze(-3)
    )

    precoded_channel = torch.matmul(
        h,
        expanded_precoder,
    )

    desired_spatial_channel = (
        torch.diagonal(
            precoded_channel,
            dim1=-3,
            dim2=-1,
        )
        .movedim(
            -1,
            -2,
        )
    )

    desired_norm = (
        torch.linalg.vector_norm(
            desired_spatial_channel,
            dim=-1,
            keepdim=True,
        )
    )

    if torch.any(desired_norm <= 0):
        raise ValueError(
            "A desired precoded channel has zero norm."
        )

    mrc_combiner = (
        desired_spatial_channel
        / desired_norm
    )

    combined_channel = torch.einsum(
        "...kr,...krj->...kj",
        mrc_combiner.conj(),
        precoded_channel,
    )

    stream_power = (
        _broadcast_per_stream_value(
            value=stream_power_w,
            reference=precoding_matrix,
            num_streams=num_users,
        )
    )

    combined_power = (
        torch.abs(
            combined_channel
        ) ** 2
    )

    combined_power = (
        combined_power
        * stream_power.unsqueeze(-2)
    )

    desired_power = torch.diagonal(
        combined_power,
        dim1=-2,
        dim2=-1,
    )

    total_same_cell_power = (
        combined_power.sum(
            dim=-1
        )
    )

    intra_cell_interference_power = (
        total_same_cell_power
        - desired_power
    )

    noise_power = torch.as_tensor(
        noise_power_w,
        dtype=h.real.dtype,
        device=h.device,
    )

    noise_shape = h.shape[:-2]

    noise_power = torch.broadcast_to(
        noise_power,
        noise_shape,
    )

    if torch.any(noise_power < 0):
        raise ValueError(
            "noise_power_w cannot be negative."
        )
    
    if inter_cell_covariance is None:
        inter_cell_interference_power = (
            torch.zeros_like(
                desired_power
            )
        )

    else:
        expected_covariance_shape = (
            h.shape[:-1]
            + (num_rx_ant,)
        )

        if (
            inter_cell_covariance.shape
            != expected_covariance_shape
        ):
            raise ValueError(
                "Unexpected inter-cell covariance shape."
            )

        inter_cell_interference_power = (
            torch.einsum(
                "...kr,...krs,...ks->...k",
                mrc_combiner.conj(),
                inter_cell_covariance,
                mrc_combiner,
            )
            .real
        )

    denominator = (
        intra_cell_interference_power
        + inter_cell_interference_power
        + noise_power
    )

    if torch.any(denominator <= 0):
        raise ValueError(
            "SINR denominator must be positive."
        )

    sinr_linear = (
        desired_power
        / denominator
    )

    return MRCPostSINRData(
        precoded_channel=precoded_channel,
        mrc_combiner=mrc_combiner,
        desired_power=desired_power,
        intra_cell_interference_power=(
            intra_cell_interference_power
        ),
        inter_cell_interference_power=(
            inter_cell_interference_power
        ),
        noise_power=noise_power,
        sinr_linear=sinr_linear,
    )