import torch

from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
)
from oran_scheduler.schedulers.one_lds_loop import (
    run_1lds_user_slot_loop,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)
from oran_scheduler.state.one_lds_decision import (
    OneLDSDecisionInputs,
)


def build_loop_test_inputs() -> OneLDSDecisionInputs:
    directions = torch.zeros(
        (
            2,
            2,
            1,
            2,
        ),
        dtype=torch.complex64,
    )

    directions[
        0,
        :,
        0,
        0,
    ] = 1.0

    directions[
        1,
        :,
        0,
        1,
    ] = 1.0

    return OneLDSDecisionInputs(
        past_average_throughput=torch.tensor(
            [1.0, 1.0]
        ),
        rank=torch.tensor(
            [1, 1],
            dtype=torch.long,
        ),
        dl_buffer=torch.tensor(
            [1.0, 1.0]
        ),
        wideband_cqi=torch.tensor(
            [10.0, 10.0]
        ),
        subband_cqi=torch.tensor(
            [
                [10.0, 10.0],
                [10.0, 10.0],
            ]
        ),
        candidate_precoder_directions=(
            directions
        ),
        candidate_valid_mask=torch.tensor(
            [True, True],
            dtype=torch.bool,
        ),
    )

def test_run_complete_1lds_user_slot_loop():
    config = OneLDSStateConfig(
        throughput_normalization_bps=1.0,
        buffer_normalization=1.0,
        subband_cqi_normalization=15.0,
        num_candidates=2,
        num_rbgs=2,
        max_rank=2,
    )

    inputs = build_loop_test_inputs()

    planned_actions = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [2, 2],
        ],
        dtype=torch.long,
    )

    def controlled_policy(
        user_slot_index,
        decision,
    ):
        return planned_actions[
            user_slot_index
        ]

    result = run_1lds_user_slot_loop(
        num_user_slots=3,
        inputs=inputs,
        state_config=config,
        action_policy=controlled_policy,
        device="cpu",
    )


    expected_allocation = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [
                NO_ALLOCATION,
                NO_ALLOCATION,
            ],
        ],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        result
        .allocation
        .candidate_by_user_slot,
        expected_allocation,
    )

    torch.testing.assert_close(
        result.actions,
        planned_actions,
    )

    assert len(
        result.decisions
    ) == 3


    decision_0 = result.decisions[0]
    decision_1 = result.decisions[1]
    decision_2 = result.decisions[2]

    torch.testing.assert_close(
        decision_0
        .spatial_features
        .allocated_rbg_count,
        torch.tensor(
            [0, 0],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        decision_1
        .spatial_features
        .allocated_rbg_count,
        torch.tensor(
            [1, 1],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        decision_2
        .spatial_features
        .allocated_rbg_count,
        torch.tensor(
            [2, 2],
            dtype=torch.long,
        ),
    )


    # Before slot 0, both UEs and NO ALLOCATION
    # are valid on both RBGs.
    assert bool(
        decision_0
        .action_mask
        .all()
        .item()
    )

    # Before slot 1:
    #
    # RBG 0 already contains UE 0.
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

    # RBG 1 already contains UE 1.
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

    # Before slot 2 both real candidates have
    # already been used on both RBGs.
    #
    # Therefore only action index 2
    # (NO ALLOCATION) remains valid.
    assert not bool(
        decision_2
        .action_mask[
            :,
            :2,
        ]
        .any()
        .item()
    )

    assert bool(
        decision_2
        .action_mask[
            :,
            2,
        ]
        .all()
        .item()
    )


