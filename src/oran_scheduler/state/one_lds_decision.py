from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
    NO_ALLOCATION,
    build_next_user_slot_action_mask,
)
from oran_scheduler.state.one_lds import (
    OneLDSRawFeatures,
    OneLDSStateConfig,
    OneLDSStateData,
    build_1lds_state,
)
from oran_scheduler.state.spatial_features import (
    SpatialAllocationFeatures,
    build_spatial_allocation_features,
)


@dataclass
class OneLDSDecisionInputs:
    """
    Candidate information that is treated as fixed while
    iterating through the 1LDS MU-MIMO user slots of one TTI.

    Shapes:

        past_average_throughput:
            [candidate]

        rank:
            [candidate]

        dl_buffer:
            [candidate]

        wideband_cqi:
            [candidate]

        subband_cqi:
            [candidate, RBG]

        candidate_precoder_directions:
            [candidate, RBG, mode, TX_ant]

        candidate_valid_mask:
            [candidate]
    """

    past_average_throughput: torch.Tensor

    rank: torch.Tensor

    dl_buffer: torch.Tensor

    wideband_cqi: torch.Tensor

    subband_cqi: torch.Tensor

    candidate_precoder_directions: torch.Tensor

    candidate_valid_mask: torch.Tensor


@dataclass
class OneLDSDecisionData:
    """
    Complete scheduler-side information for one 1LDS
    user-slot decision.

    spatial_features:
        Allocation-dependent raw state features.

    raw_features:
        Complete unnormalized 1LDS state features.

    state_data:
        Normalized/flattened neural-network state.

    action_mask:
        [RBG, candidate + no-allocation]
    """

    spatial_features: SpatialAllocationFeatures

    raw_features: OneLDSRawFeatures

    state_data: OneLDSStateData

    action_mask: torch.Tensor


def build_1lds_decision_data(
    allocation: CellAllocation,
    user_slot_index: int,
    inputs: OneLDSDecisionInputs,
    config: OneLDSStateConfig,
) -> OneLDSDecisionData:
    """
    Build the state and action mask for one 1LDS
    MU-MIMO user-slot decision.

    The allocation represents all decisions completed
    before user_slot_index.

    The function:

        allocation
            ->
        allocation-dependent spatial features
            ->
        complete raw 1LDS features
            ->
        normalized 1LDS state
            ->
        valid action mask for this user slot
    """

    if allocation.num_rbgs != config.num_rbgs:
        raise ValueError(
            "Allocation RBG count does not match "
            "the 1LDS state configuration."
        )

    if not (
        0
        <= user_slot_index
        < allocation.num_user_slots
    ):
        raise ValueError(
            "user_slot_index is invalid."
        )

    if tuple(
        inputs.rank.shape
    ) != (
        config.num_candidates,
    ):
        raise ValueError(
            "rank must have shape [candidate]."
        )

    target_slot = (
        allocation
        .candidate_by_user_slot[
            user_slot_index,
            :,
        ]
    )

    if torch.any(
        target_slot != NO_ALLOCATION
    ):
        raise ValueError(
            "The requested user slot has already "
            "been populated."
        )

    spatial_features = (
        build_spatial_allocation_features(
            allocation=allocation,
            candidate_rank=inputs.rank,
            candidate_precoder_directions=(
                inputs
                .candidate_precoder_directions
            ),
            candidate_valid_mask=(
                inputs
                .candidate_valid_mask
            ),
        )
    )

    raw_features = OneLDSRawFeatures(
        past_average_throughput=(
            inputs
            .past_average_throughput
        ),
        rank=inputs.rank,
        allocated_rbg_count=(
            spatial_features
            .allocated_rbg_count
        ),
        dl_buffer=inputs.dl_buffer,
        wideband_cqi=(
            inputs.wideband_cqi
        ),
        subband_cqi=(
            inputs.subband_cqi
        ),
        max_precoder_cross_correlation=(
            spatial_features
            .max_precoder_cross_correlation
        ),
        candidate_valid_mask=(
            inputs
            .candidate_valid_mask
        ),
    )

    state_data = build_1lds_state(
        features=raw_features,
        config=config,
    )

    action_mask = (
        build_next_user_slot_action_mask(
            allocation=allocation,
            num_candidates=(
                config.num_candidates
            ),
            user_slot_index=(
                user_slot_index
            ),
            candidate_valid_mask=(
                inputs
                .candidate_valid_mask
            ),
        )
    )

    expected_action_mask_shape = (
        config.num_rbgs,
        config.num_candidates + 1,
    )

    if tuple(
        action_mask.shape
    ) != expected_action_mask_shape:
        raise RuntimeError(
            "Internal 1LDS action-mask "
            "shape error."
        )

    return OneLDSDecisionData(
        spatial_features=(
            spatial_features
        ),
        raw_features=raw_features,
        state_data=state_data,
        action_mask=action_mask,
    )

