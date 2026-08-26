import math

import torch

from oran_scheduler.schedulers.allocation import (
    apply_user_slot_actions,
    build_empty_cell_allocation,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)
from oran_scheduler.state.one_lds_decision import (
    OneLDSDecisionInputs,
    build_1lds_decision_data,
)


def build_test_inputs(
    device: str,
) -> OneLDSDecisionInputs:
    directions = torch.zeros(
        (
            2,
            2,
            1,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    # Candidate 0:
    #
    # beam = [1, 0]
    directions[
        0,
        :,
        0,
        0,
    ] = 1.0

    # Candidate 1:
    #
    # beam = [1/sqrt(2), 1/sqrt(2)]
    value = (
        1.0
        / math.sqrt(2.0)
    )

    directions[
        1,
        :,
        0,
        0,
    ] = value

    directions[
        1,
        :,
        0,
        1,
    ] = value

    return OneLDSDecisionInputs(
        past_average_throughput=(
            torch.tensor(
                [2.0, 4.0],
                device=device,
            )
        ),
        rank=torch.tensor(
            [1, 1],
            dtype=torch.long,
            device=device,
        ),
        dl_buffer=torch.tensor(
            [50.0, 75.0],
            device=device,
        ),
        wideband_cqi=torch.tensor(
            [10.0, 12.0],
            device=device,
        ),
        subband_cqi=torch.tensor(
            [
                [9.0, 10.0],
                [11.0, 12.0],
            ],
            device=device,
        ),
        candidate_precoder_directions=(
            directions
        ),
        candidate_valid_mask=torch.tensor(
            [True, True],
            dtype=torch.bool,
            device=device,
        ),
    )



def test_initial_1lds_decision_has_zero_spatial_features():
    device = "cuda:0"

    config = OneLDSStateConfig(
        throughput_normalization_bps=4.0,
        buffer_normalization=100.0,
        subband_cqi_normalization=15.0,
        num_candidates=2,
        num_rbgs=2,
        max_rank=2,
    )

    allocation = build_empty_cell_allocation(
        num_user_slots=3,
        num_rbgs=2,
        device=device,
    )

    inputs = build_test_inputs(
        device=device,
    )

    decision = build_1lds_decision_data(
        allocation=allocation,
        user_slot_index=0,
        inputs=inputs,
        config=config,
    )

    torch.testing.assert_close(
        decision
        .spatial_features
        .allocated_rbg_count,
        torch.zeros(
            2,
            dtype=torch.long,
            device=device,
        ),
    )

    torch.testing.assert_close(
        decision
        .spatial_features
        .max_precoder_cross_correlation,
        torch.zeros(
            (
                2,
                2,
            ),
            device=device,
        ),
    )

    assert decision.state_data.state.shape == (
        18,
    )

    assert decision.action_mask.shape == (
        2,
        3,
    )

    assert bool(
        decision.action_mask.all().item()
    )

def test_1lds_state_changes_after_user_slot_action():
    device = "cuda:0"

    config = OneLDSStateConfig(
        throughput_normalization_bps=4.0,
        buffer_normalization=100.0,
        subband_cqi_normalization=15.0,
        num_candidates=2,
        num_rbgs=2,
        max_rank=2,
    )

    allocation = build_empty_cell_allocation(
        num_user_slots=3,
        num_rbgs=2,
        device=device,
    )

    inputs = build_test_inputs(
        device=device,
    )

    decision_0 = build_1lds_decision_data(
        allocation=allocation,
        user_slot_index=0,
        inputs=inputs,
        config=config,
    )

    first_actions = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.long,
        device=device,
    )

    allocation_after_slot_0 = (
        apply_user_slot_actions(
            allocation=allocation,
            user_slot_index=0,
            actions=first_actions,
            num_candidates=2,
            candidate_valid_mask=(
                inputs
                .candidate_valid_mask
            ),
        )
    )

    decision_1 = build_1lds_decision_data(
        allocation=(
            allocation_after_slot_0
        ),
        user_slot_index=1,
        inputs=inputs,
        config=config,
    )

    expected_count = torch.tensor(
        [
            1,
            1,
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        decision_1
        .spatial_features
        .allocated_rbg_count,
        expected_count,
    )

    torch.testing.assert_close(
        decision_1
        .state_data
        .ue_feature_segments[
            :,
            2,
        ],
        torch.tensor(
            [
                0.5,
                0.5,
            ],
            device=device,
        ),
    )

    expected_correlation = (
        1.0
        / math.sqrt(2.0)
    )

    correlations = (
        decision_1
        .spatial_features
        .max_precoder_cross_correlation
    )

    # Candidate 0 is already scheduled on RBG 0.
    # Self-comparison is excluded.
    torch.testing.assert_close(
        correlations[
            0,
            0,
        ],
        torch.tensor(
            0.0,
            device=device,
        ),
    )

    # On RBG 0, candidate 1 is compared against
    # scheduled candidate 0.
    torch.testing.assert_close(
        correlations[
            1,
            0,
        ],
        torch.tensor(
            expected_correlation,
            device=device,
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    # On RBG 1, candidate 0 is compared against
    # scheduled candidate 1.
    torch.testing.assert_close(
        correlations[
            0,
            1,
        ],
        torch.tensor(
            expected_correlation,
            device=device,
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    # Candidate 1 is already scheduled on RBG 1,
    # so self-comparison is excluded.
    torch.testing.assert_close(
        correlations[
            1,
            1,
        ],
        torch.tensor(
            0.0,
            device=device,
        ),
    )


    # RBG 0 already contains candidate 0.
    assert not bool(
        decision_1
        .action_mask[
            0,
            0,
        ]
        .item()
    )

    assert bool(
        decision_1
        .action_mask[
            0,
            1,
        ]
        .item()
    )

    # RBG 1 already contains candidate 1.
    assert bool(
        decision_1
        .action_mask[
            1,
            0,
        ]
        .item()
    )

    assert not bool(
        decision_1
        .action_mask[
            1,
            1,
        ]
        .item()
    )

    # Final action index 2 is NO ALLOCATION.
    assert bool(
        decision_1
        .action_mask[
            0,
            2,
        ]
        .item()
    )

    assert bool(
        decision_1
        .action_mask[
            1,
            2,
        ]
        .item()
    )

    assert not torch.equal(
        decision_0.state_data.state,
        decision_1.state_data.state,
    )


