from dataclasses import dataclass

import torch

from sionna.phy.nr.utils import (
    calculate_tb_size,
    decode_mcs_index,
)

@dataclass(frozen=True)
class RateConfig:
    """
    Configuration for the current per-RBG SISO rate abstraction.

    Paper-specified:
        - downlink
        - BLER target = 10%
        - modulation support through 256-QAM

    Current open-reproduction choices:
        - PDSCH MCS table 2
        - one stream
        - 14 OFDM symbols per scheduling slot
        - all 12 x 14 REs of one RBG are treated as allocated
        - no DMRS/control overhead is removed yet

    At 30 kHz SCS with normal CP, a slot is 0.5 ms.
    """

    mcs_category: int = 1
    mcs_table_index: int = 2

    subcarriers_per_rbg: int = 12
    num_slot_ofdm_symbols: int = 14
    num_streams_per_ue: int = 1

    slot_duration_s: float = 0.5e-3

    bler_target: float = 0.10

    device: str = "cuda:0"

@dataclass
class RateData:
    """
    Per-UE/per-RBG rate quantities.

    Every tensor has shape:

        [batch, UE, RBG]
    """

    modulation_order: torch.Tensor

    target_coderate: torch.Tensor

    tb_size_bits: torch.Tensor

    nominal_rate_bps: torch.Tensor

    expected_goodput_bps: torch.Tensor

    target_compliant_rate_bps: torch.Tensor

    meets_bler_target: torch.Tensor

def validate_rate_input(
    mcs_index: torch.Tensor,
    tbler: torch.Tensor,
    config: RateConfig,
) -> None:
    """
    Validate per-RBG MCS and TBLER tensors.
    """

    if mcs_index.ndim != 3:
        raise ValueError(
            "mcs_index must have shape "
            "[batch, UE, RBG]."
        )

    if tbler.ndim != 3:
        raise ValueError(
            "tbler must have shape "
            "[batch, UE, RBG]."
        )

    if mcs_index.shape != tbler.shape:
        raise ValueError(
            "mcs_index and tbler must have "
            "the same shape."
        )

    if mcs_index.device != tbler.device:
        raise ValueError(
            "mcs_index and tbler must be "
            "on the same device."
        )

    if not torch.isfinite(tbler).all():
        raise ValueError(
            "tbler contains non-finite values."
        )

    if torch.any(tbler < 0.0):
        raise ValueError(
            "tbler cannot be below zero."
        )

    if torch.any(tbler > 1.0):
        raise ValueError(
            "tbler cannot exceed one."
        )

    if config.num_slot_ofdm_symbols <= 0:
        raise ValueError(
            "num_slot_ofdm_symbols must be positive."
        )

    if config.subcarriers_per_rbg <= 0:
        raise ValueError(
            "subcarriers_per_rbg must be positive."
        )

    if config.num_streams_per_ue <= 0:
        raise ValueError(
            "num_streams_per_ue must be positive."
        )

    if config.slot_duration_s <= 0.0:
        raise ValueError(
            "slot_duration_s must be positive."
        )

def compute_rbg_rates(
    mcs_index: torch.Tensor,
    tbler: torch.Tensor,
    config: RateConfig,
) -> RateData:
    """
    Convert per-RBG MCS and TBLER into deterministic rate estimates.

    Inputs:

        mcs_index:
            [B, UE, RBG]

        tbler:
            [B, UE, RBG]
    """

    validate_rate_input(
        mcs_index=mcs_index,
        tbler=tbler,
        config=config,
    )

    is_pusch = (
        config.mcs_category == 0
    )

    (
        modulation_order,
        target_coderate,
    ) = decode_mcs_index(
        mcs_index=mcs_index,
        table_index=config.mcs_table_index,
        is_pusch=is_pusch,
        device=config.device,
    )

    num_allocated_re = (
        config.num_slot_ofdm_symbols
        * config.subcarriers_per_rbg
        * config.num_streams_per_ue
    )

    num_coded_bits = (
        modulation_order
        * num_allocated_re
    )

    (
        tb_size_bits,
        _,
        _,
        _,
        _,
    ) = calculate_tb_size(
        modulation_order=modulation_order,
        target_coderate=target_coderate,
        num_coded_bits=num_coded_bits,
        return_cw_length=False,
        device=config.device,
    )

    tb_size_float = tb_size_bits.to(
        dtype=torch.float32
    )

    nominal_rate_bps = (
        tb_size_float
        / config.slot_duration_s
    )

    expected_goodput_bps = (
        nominal_rate_bps
        * (1.0 - tbler)
    )

    meets_bler_target = (
        tbler <= config.bler_target
    )

    target_compliant_rate_bps = torch.where(
        meets_bler_target,
        nominal_rate_bps,
        torch.zeros_like(
            nominal_rate_bps
        ),
    )

    return RateData(
        modulation_order=modulation_order,
        target_coderate=target_coderate,
        tb_size_bits=tb_size_bits,
        nominal_rate_bps=nominal_rate_bps,
        expected_goodput_bps=(
            expected_goodput_bps
        ),
        target_compliant_rate_bps=(
            target_compliant_rate_bps
        ),
        meets_bler_target=(
            meets_bler_target
        ),
    )