from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class OneLDSStateConfig:
    """
    Configuration for the 1LDS state representation.

    Paper-specified dimensions:
        - maximum candidates = 10
        - RBGs = 18
        - maximum UE rank = 2

    Paper-specified normalization forms:
        past throughput:
            R_hat = R / R_max

        rank:
            h_hat = h / 2

        allocated RBG count:
            d_hat = d / N_RBG

        buffer:
            b_hat = b / b_max

        wideband CQI:
            min-max normalization

        sub-band CQI:
            g_hat = g / g_bar

    The numerical values of R_max, b_max, and g_bar
    are exposed as reproduction configuration parameters.
    """

    throughput_normalization_bps: float

    buffer_normalization: float

    subband_cqi_normalization: float

    num_candidates: int = 10
    num_rbgs: int = 18
    max_rank: int = 2

    wideband_cqi_min: float = 0.0
    wideband_cqi_max: float = 15.0


    @property
    def ue_feature_size(self) -> int:
        """
        Features belonging to one candidate UE.

            5 UE-wide
            +
            N_RBG sub-band CQIs
            +
            N_RBG cross-correlations
        """

        return (
            5
            + 2 * self.num_rbgs
        )

    @property
    def state_size(self) -> int:
        return (
            self.num_candidates
            * self.ue_feature_size
        )

    @property
    def actor_output_size(self) -> int:
        """
        1LDS actor output:

            N_RBG * (num_candidates + no-allocation)
        """

        return (
            self.num_rbgs
            * (self.num_candidates + 1)
        )


@dataclass
class OneLDSRawFeatures:
    """
    Raw scheduler features before normalization.

    UE-wide tensors:

        past_average_throughput:
            [..., candidate]

        rank:
            [..., candidate]

        allocated_rbg_count:
            [..., candidate]

        dl_buffer:
            [..., candidate]

        wideband_cqi:
            [..., candidate]

        candidate_valid_mask:
            [..., candidate]

    RBG-specific tensors:

        subband_cqi:
            [..., candidate, RBG]

        max_precoder_cross_correlation:
            [..., candidate, RBG]

    Leading dimensions can represent batch, cell, etc.
    """

    past_average_throughput: torch.Tensor

    rank: torch.Tensor

    allocated_rbg_count: torch.Tensor

    dl_buffer: torch.Tensor

    wideband_cqi: torch.Tensor

    subband_cqi: torch.Tensor

    max_precoder_cross_correlation: torch.Tensor

    candidate_valid_mask: torch.Tensor

@dataclass
class OneLDSStateData:
    """
    Constructed 1LDS state.

    ue_feature_segments:
        [..., candidate, 5 + 2*N_RBG]

    state:
        [..., candidate * (5 + 2*N_RBG)]

    Invalid/padded candidate segments are completely zeroed.
    """

    ue_feature_segments: torch.Tensor

    state: torch.Tensor

def validate_1lds_config(
    config: OneLDSStateConfig,
) -> None:

    if config.num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive."
        )

    if config.num_rbgs <= 0:
        raise ValueError(
            "num_rbgs must be positive."
        )

    if config.max_rank <= 0:
        raise ValueError(
            "max_rank must be positive."
        )

    if (
        config.throughput_normalization_bps
        <= 0.0
    ):
        raise ValueError(
            "throughput_normalization_bps "
            "must be positive."
        )

    if config.buffer_normalization <= 0.0:
        raise ValueError(
            "buffer_normalization must be positive."
        )

    if (
        config.subband_cqi_normalization
        <= 0.0
    ):
        raise ValueError(
            "subband_cqi_normalization "
            "must be positive."
        )

    if (
        config.wideband_cqi_max
        <= config.wideband_cqi_min
    ):
        raise ValueError(
            "wideband_cqi_max must be greater "
            "than wideband_cqi_min."
        )


def validate_1lds_raw_features(
    features: OneLDSRawFeatures,
    config: OneLDSStateConfig,
) -> None:
    """
    Validate shapes and physically meaningful values.
    """

    validate_1lds_config(
        config
    )

    ue_shape = (
        features
        .past_average_throughput
        .shape
    )

    if len(ue_shape) < 1:
        raise ValueError(
            "UE features must have at least "
            "one dimension."
        )

    if ue_shape[-1] != (
        config.num_candidates
    ):
        raise ValueError(
            "Candidate dimension does not match "
            "config.num_candidates."
        )


    ue_tensors = (
        features.rank,
        features.allocated_rbg_count,
        features.dl_buffer,
        features.wideband_cqi,
        features.candidate_valid_mask,
    )

    for tensor in ue_tensors:

        if tuple(
            tensor.shape
        ) != tuple(
            ue_shape
        ):
            raise ValueError(
                "All UE-wide features must have "
                "identical shapes."
            )

    expected_rbg_shape = (
        tuple(ue_shape)
        + (
            config.num_rbgs,
        )
    )

    if tuple(
        features.subband_cqi.shape
    ) != expected_rbg_shape:
        raise ValueError(
            "subband_cqi must have shape "
            "[..., candidate, RBG]."
        )

    if tuple(
        features
        .max_precoder_cross_correlation
        .shape
    ) != expected_rbg_shape:
        raise ValueError(
            "max_precoder_cross_correlation must "
            "have shape [..., candidate, RBG]."
        )


    reference_device = (
        features
        .past_average_throughput
        .device
    )

    all_tensors = (
        features.rank,
        features.allocated_rbg_count,
        features.dl_buffer,
        features.wideband_cqi,
        features.subband_cqi,
        features.max_precoder_cross_correlation,
        features.candidate_valid_mask,
    )

    for tensor in all_tensors:

        if tensor.device != reference_device:
            raise ValueError(
                "All 1LDS feature tensors must "
                "be on the same device."
            )


    valid_mask = (
        features
        .candidate_valid_mask
        .to(
            dtype=torch.bool
        )
    )

    valid_past_throughput = (
        features
        .past_average_throughput[
            valid_mask
        ]
    )

    valid_rank = (
        features.rank[
            valid_mask
        ]
    )

    valid_allocated_count = (
        features
        .allocated_rbg_count[
            valid_mask
        ]
    )

    valid_buffer = (
        features.dl_buffer[
            valid_mask
        ]
    )

    valid_wideband_cqi = (
        features.wideband_cqi[
            valid_mask
        ]
    )


    if torch.any(
        valid_past_throughput < 0
    ):
        raise ValueError(
            "Past throughput cannot be negative."
        )

    if torch.any(
        valid_rank < 1
    ) or torch.any(
        valid_rank > config.max_rank
    ):
        raise ValueError(
            "Valid UE ranks are outside the "
            "supported range."
        )

    if torch.any(
        valid_allocated_count < 0
    ) or torch.any(
        valid_allocated_count
        > config.num_rbgs
    ):
        raise ValueError(
            "Allocated RBG count is invalid."
        )

    if torch.any(
        valid_buffer < 0
    ):
        raise ValueError(
            "DL buffer cannot be negative."
        )

    if torch.any(
        valid_wideband_cqi
        < config.wideband_cqi_min
    ) or torch.any(
        valid_wideband_cqi
        > config.wideband_cqi_max
    ):
        raise ValueError(
            "Wideband CQI is outside the "
            "configured range."
        )

    valid_rbg_mask = (
        valid_mask
        .unsqueeze(-1)
        .expand_as(
            features.subband_cqi
        )
    )

    valid_subband_cqi = (
        features.subband_cqi[
            valid_rbg_mask
        ]
    )

    valid_cross_correlation = (
        features
        .max_precoder_cross_correlation[
            valid_rbg_mask
        ]
    )

    if torch.any(
        valid_subband_cqi < 0
    ):
        raise ValueError(
            "Sub-band CQI cannot be negative."
        )

    if torch.any(
        valid_cross_correlation < 0
    ):
        raise ValueError(
            "Precoder cross-correlation "
            "cannot be negative."
        )

    floating_tensors = (
        features.past_average_throughput,
        features.dl_buffer,
        features.wideband_cqi,
        features.subband_cqi,
        features.max_precoder_cross_correlation,
    )

    for tensor in floating_tensors:

        if not torch.isfinite(
            tensor
        ).all():
            raise ValueError(
                "1LDS features contain "
                "non-finite values."
            )
        

def build_1lds_state(
    features: OneLDSRawFeatures,
    config: OneLDSStateConfig,
) -> OneLDSStateData:
    """
    Normalize and assemble the paper-style 1LDS state.

    Per-candidate feature ordering used by this
    reproduction is:

        0: normalized past throughput
        1: normalized rank
        2: normalized allocated RBG count
        3: normalized DL buffer
        4: normalized wideband CQI

        5 ... 5+N_RBG-1:
            normalized sub-band CQIs

        remaining N_RBG:
            max precoder cross-correlations

    The exact memory ordering of the two RBG vectors does
    not affect a model trained from scratch as long as the
    ordering is kept consistent.
    """

    validate_1lds_raw_features(
        features=features,
        config=config,
    )


    device = (
        features
        .past_average_throughput
        .device
    )

    dtype = torch.float32


    valid_mask = (
        features
        .candidate_valid_mask
        .to(
            device=device,
            dtype=torch.bool,
        )
    )

    normalized_past_throughput = (
        features
        .past_average_throughput
        .to(dtype=dtype)
        / config.throughput_normalization_bps
    )

    normalized_rank = (
        features.rank.to(
            dtype=dtype
        )
        / float(
            config.max_rank
        )
    )

    normalized_allocated_count = (
        features
        .allocated_rbg_count
        .to(dtype=dtype)
        / float(
            config.num_rbgs
        )
    )

    normalized_buffer = (
        features.dl_buffer.to(
            dtype=dtype
        )
        / config.buffer_normalization
    )

    wideband_range = (
        config.wideband_cqi_max
        - config.wideband_cqi_min
    )

    normalized_wideband_cqi = (
        (
            features.wideband_cqi.to(
                dtype=dtype
            )
            - config.wideband_cqi_min
        )
        / wideband_range
    )


    normalized_subband_cqi = (
        features.subband_cqi.to(
            dtype=dtype
        )
        / config.subband_cqi_normalization
    )

    cross_correlation = (
        features
        .max_precoder_cross_correlation
        .to(dtype=dtype)
    )

    ue_wide_features = torch.stack(
        (
            normalized_past_throughput,
            normalized_rank,
            normalized_allocated_count,
            normalized_buffer,
            normalized_wideband_cqi,
        ),
        dim=-1,
    )

    ue_feature_segments = torch.cat(
        (
            ue_wide_features,
            normalized_subband_cqi,
            cross_correlation,
        ),
        dim=-1,
    )

    if ue_feature_segments.shape[-1] != (
        config.ue_feature_size
    ):
        raise RuntimeError(
            "Internal 1LDS feature-size error."
        )

    ue_feature_segments = torch.where(
        valid_mask.unsqueeze(-1),
        ue_feature_segments,
        torch.zeros_like(
            ue_feature_segments
        ),
    )

    state = ue_feature_segments.flatten(
        start_dim=-2,
    )

    if state.shape[-1] != (
        config.state_size
    ):
        raise RuntimeError(
            "Internal 1LDS state-size error."
        )

    return OneLDSStateData(
        ue_feature_segments=(
            ue_feature_segments
        ),
        state=state,
    )




