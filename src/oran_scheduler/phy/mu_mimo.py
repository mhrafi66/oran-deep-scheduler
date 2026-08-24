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


@dataclass
class MUMIMOLayerSINRData:
    """
    Post-MRC SINR for an arbitrary set of spatial layers.

    effective_channel:
        [..., layer, TX_ant]

        Effective CSI rows used to compute RZF.

    precoding_matrix:
        [..., TX_ant, layer]

    combined_channel:
        [..., layer, layer]

        Entry [i, j] is the effective channel from
        transmit beam j into receive combiner i.

    stream_power_w:
        [..., layer]

    desired_power:
        [..., layer]

    intra_cell_interference_power:
        [..., layer]

    inter_cell_interference_power:
        [..., layer]

    noise_power:
        [..., layer]

    sinr_linear:
        [..., layer]
    """

    effective_channel: torch.Tensor
    precoding_matrix: torch.Tensor
    combined_channel: torch.Tensor

    stream_power_w: torch.Tensor

    desired_power: torch.Tensor
    intra_cell_interference_power: torch.Tensor
    inter_cell_interference_power: torch.Tensor
    noise_power: torch.Tensor

    sinr_linear: torch.Tensor

def compute_mu_mimo_layer_sinr(
    layer_physical_channel: torch.Tensor,
    layer_rx_combiner: torch.Tensor,
    total_tx_power_w: float | torch.Tensor,
    noise_power_w: float | torch.Tensor,
    rzf_alpha: float,
    inter_cell_covariance: torch.Tensor | None = None,
    precision: str = "single",
) -> MUMIMOLayerSINRData:
    """
    Evaluate RZF + MRC SINR for an arbitrary set of spatial layers.

    Args:
        layer_physical_channel:
            [..., layer, RX_ant, TX_ant]

            Each layer carries the physical channel of the UE
            intended to receive that layer.

            If one UE has rank 2, its physical channel therefore
            appears twice in this tensor.

        layer_rx_combiner:
            [..., layer, RX_ant]

            Receive-combining vector associated with every layer.

        total_tx_power_w:
            Total serving-BS transmit power available on this
            subcarrier.

            Current reproduction assumption:
            power is divided equally among active spatial layers.

        noise_power_w:
            AWGN power per physical RX branch.

        rzf_alpha:
            Explicit RZF regularization parameter.

            The Bell Labs paper specifies RZF but does not publish
            its numerical regularization rule, so this remains an
            explicit reproduction parameter.

        inter_cell_covariance:
            Optional tensor

                [..., layer, RX_ant, RX_ant]

            describing interference from other cells as observed
            by the UE associated with every layer.

    Returns:
        Per-layer post-MRC SINR quantities.
    """

    if layer_physical_channel.ndim < 3:
        raise ValueError(
            "layer_physical_channel must end with "
            "[layer, RX_ant, TX_ant]."
        )

    if not torch.is_complex(
        layer_physical_channel
    ):
        raise ValueError(
            "layer_physical_channel must be complex-valued."
        )

    if (
        layer_rx_combiner.shape
        != layer_physical_channel.shape[:-1]
    ):
        raise ValueError(
            "layer_rx_combiner must match "
            "layer_physical_channel without "
            "the TX-antenna dimension."
        )

    if not torch.is_complex(
        layer_rx_combiner
    ):
        raise ValueError(
            "layer_rx_combiner must be complex-valued."
        )

    num_layers = (
        layer_physical_channel.shape[-3]
    )

    num_rx_ant = (
        layer_physical_channel.shape[-2]
    )

    num_tx_ant = (
        layer_physical_channel.shape[-1]
    )

    if num_layers <= 0:
        raise ValueError(
            "At least one spatial layer is required."
        )

    if num_layers > num_tx_ant:
        raise ValueError(
            "Number of layers cannot exceed "
            "the number of TX antenna ports."
        )

    if rzf_alpha < 0.0:
        raise ValueError(
            "rzf_alpha cannot be negative."
        )

    combiner_norm = torch.linalg.vector_norm(
        layer_rx_combiner,
        dim=-1,
        keepdim=True,
    )

    if torch.any(combiner_norm <= 0):
        raise ValueError(
            "A receive combiner has zero norm."
        )

    normalized_combiner = (
        layer_rx_combiner
        / combiner_norm
    )

    effective_channel = torch.einsum(
        "...lr,...lrt->...lt",
        normalized_combiner.conj(),
        layer_physical_channel,
    )

    precoding_matrix = compute_rzf_matrix(
        effective_channel=effective_channel,
        alpha=rzf_alpha,
        precision=precision,
    )

    precoded_spatial_channel = torch.matmul(
        layer_physical_channel,
        precoding_matrix.unsqueeze(-3),
    )

    combined_channel = torch.einsum(
        "...lr,...lrj->...lj",
        normalized_combiner.conj(),
        precoded_spatial_channel,
    )

    total_tx_power = torch.as_tensor(
        total_tx_power_w,
        dtype=layer_physical_channel.real.dtype,
        device=layer_physical_channel.device,
    )

    if torch.any(total_tx_power <= 0):
        raise ValueError(
            "total_tx_power_w must be positive."
        )

    stream_power_shape = (
        layer_physical_channel.shape[:-3]
        + (num_layers,)
    )

    per_layer_power = (
        total_tx_power
        / float(num_layers)
    )

    stream_power_w = torch.broadcast_to(
        per_layer_power,
        stream_power_shape,
    )

    combined_power = (
        torch.abs(
            combined_channel
        ) ** 2
    )

    combined_power = (
        combined_power
        * stream_power_w.unsqueeze(-2)
    )

    desired_power = torch.diagonal(
        combined_power,
        dim1=-2,
        dim2=-1,
    )

    total_serving_power = (
        combined_power.sum(
            dim=-1
        )
    )

    intra_cell_interference_power = (
        total_serving_power
        - desired_power
    )

    noise_power = torch.as_tensor(
        noise_power_w,
        dtype=layer_physical_channel.real.dtype,
        device=layer_physical_channel.device,
    )

    if torch.any(noise_power < 0):
        raise ValueError(
            "noise_power_w cannot be negative."
        )

    noise_power = torch.broadcast_to(
        noise_power,
        stream_power_shape,
    )

    if inter_cell_covariance is None:
        inter_cell_interference_power = (
            torch.zeros_like(
                desired_power
            )
        )

    else:
        expected_covariance_shape = (
            layer_physical_channel.shape[:-1]
            + (num_rx_ant,)
        )

        if (
            inter_cell_covariance.shape
            != expected_covariance_shape
        ):
            raise ValueError(
                "inter_cell_covariance must have shape "
                "[..., layer, RX_ant, RX_ant]."
            )
        inter_cell_interference_power = (
            torch.einsum(
                "...lr,...lrs,...ls->...l",
                normalized_combiner.conj(),
                inter_cell_covariance,
                normalized_combiner,
            )
            .real
        )

        inter_cell_interference_power = (
            torch.clamp(
                inter_cell_interference_power,
                min=0.0,
            )
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

    return MUMIMOLayerSINRData(
        effective_channel=effective_channel,
        precoding_matrix=precoding_matrix,
        combined_channel=combined_channel,
        stream_power_w=stream_power_w,
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


# alpha = 0.1

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