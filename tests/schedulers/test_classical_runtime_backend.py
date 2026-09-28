from types import SimpleNamespace

import torch

import oran_scheduler.schedulers.classical_runtime_backend as backend

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    PFGreedyRBGScore,
)


class FakePhysicalScorer:

    def __init__(
        self,
        inputs,
    ) -> None:

        del inputs

        self.num_score_requests = 0

        self.num_unique_phy_evaluations = 0


    def __call__(
        self,
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:

        self.num_score_requests += 1

        self.num_unique_phy_evaluations += 1

        candidates = tuple(
            int(x)
            for x
            in selected_candidates.tolist()
        )

        candidate_set = frozenset(
            candidates
        )

        #
        # Single-user rates:
        #
        # RBG 0:
        #   c0 = 10
        #   c1 = 8
        #   c2 = 6
        #
        # RBG 1:
        #   c0 = 7
        #   c1 = 10
        #   c2 = 6
        #
        if len(
            candidates
        ) == 1:

            candidate = candidates[
                0
            ]

            singles = {
                0: {
                    0: 10.0,
                    1: 8.0,
                    2: 6.0,
                },

                1: {
                    0: 7.0,
                    1: 10.0,
                    2: 6.0,
                },
            }

            rate = singles[
                rbg_index
            ][
                candidate
            ]

            selected_rates = [
                rate,
            ]

        elif (
            candidate_set
            == frozenset(
                {
                    0,
                    1,
                }
            )
        ):

            rate = 12.0

            selected_rates = [
                6.0
                for _
                in candidates
            ]

        elif (
            candidate_set
            == frozenset(
                {
                    0,
                    2,
                }
            )
        ):

            rate = 9.0

            selected_rates = [
                4.5
                for _
                in candidates
            ]

        elif (
            candidate_set
            == frozenset(
                {
                    1,
                    2,
                }
            )
        ):

            rate = 11.0

            selected_rates = [
                5.5
                for _
                in candidates
            ]

        else:

            rate = 12.0

            selected_rates = [
                rate
                / len(
                    candidates
                )
                for _
                in candidates
            ]

        return PFGreedyRBGScore(
            total_rate_bps=torch.tensor(
                rate,
                dtype=torch.float32,
            ),

            selected_candidate_rate_bps=(
                torch.tensor(
                    selected_rates,
                    dtype=torch.float32,
                )
            ),
        )


def _prepared():

    return SimpleNamespace(
        candidate_valid_mask=(
            torch.tensor(
                [
                    True,
                    True,
                    True,
                ],
                dtype=torch.bool,
            )
        ),

        candidate_global_ue_indices=(
            torch.tensor(
                [
                    100,
                    101,
                    102,
                ],
                dtype=torch.long,
            )
        ),

        decision_inputs=(
            SimpleNamespace(
                past_average_throughput=(
                    torch.ones(
                        3,
                        dtype=torch.float32,
                    )
                ),

                subband_cqi=(
                    torch.zeros(
                        (
                            3,
                            2,
                        ),
                        dtype=torch.float32,
                    )
                ),
            )
        ),
    )


def _physical_inputs():

    return SimpleNamespace(
        candidate_global_ue_indices=(
            torch.tensor(
                [
                    100,
                    101,
                    102,
                ],
                dtype=torch.long,
            )
        )
    )


def test_allocation_to_ppo_actions() -> None:

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.tensor(
                [
                    [
                        0,
                        -1,
                    ],
                    [
                        2,
                        1,
                    ],
                ],
                dtype=torch.long,
            )
        )
    )

    actions = (
        backend
        .allocation_to_ppo_actions(
            allocation=allocation,
            num_candidates=3,
        )
    )

    expected = torch.tensor(
        [
            [
                0,
                3,
            ],
            [
                2,
                1,
            ],
        ],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        actions,
        expected,
    )


def test_physical_pf_fds_selects_best_single_user() -> None:

    rate = torch.tensor(
        [
            [
                10.0,
                7.0,
            ],
            [
                8.0,
                10.0,
            ],
            [
                6.0,
                6.0,
            ],
        ],
        dtype=torch.float32,
    )

    allocation = (
        backend
        .build_physical_pf_initial_allocation(
            single_user_rate_bps=rate,

            candidate_past_average_throughput=(
                torch.ones(
                    3
                )
            ),

            candidate_valid_mask=(
                torch.ones(
                    3,
                    dtype=torch.bool,
                )
            ),

            num_user_slots=2,
        )
    )

    expected = torch.tensor(
        [
            [
                0,
                1,
            ],
            [
                -1,
                -1,
            ],
        ],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        allocation
        .candidate_by_user_slot,
        expected,
    )


def test_baseline_runtime_backend(
    monkeypatch,
) -> None:

    monkeypatch.setattr(
        backend,
        "CachedPPOPhysicalRBGScorer",
        FakePhysicalScorer,
    )

    sentinel_outcome = object()

    captured = {}

    def fake_final_evaluator(
        **kwargs,
    ):

        captured.update(
            kwargs
        )

        return sentinel_outcome

    monkeypatch.setattr(
        backend,
        "evaluate_ppo_physical_tti",
        fake_final_evaluator,
    )

    result = (
        backend
        .run_classical_runtime_scheduler(
            mode="baseline",
            prepared=_prepared(),
            physical_inputs=(
                _physical_inputs()
            ),
            num_user_slots=2,
        )
    )

    assert result.mode == "baseline"

    assert (
        result.physical_outcome
        is sentinel_outcome
    )

    #
    # Baseline adds candidate 1 next to candidate 0
    # on RBG 0 and candidate 0 next to candidate 1
    # on RBG 1 because both improve throughput.
    #
    expected = torch.tensor(
        [
            [
                0,
                1,
            ],
            [
                1,
                0,
            ],
        ],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        result
        .allocation
        .candidate_by_user_slot,
        expected,
    )

    assert (
        captured[
            "run_counterfactual_greedy"
        ]
        is False
    )


def test_pf_greedy_runtime_backend(
    monkeypatch,
) -> None:

    monkeypatch.setattr(
        backend,
        "CachedPPOPhysicalRBGScorer",
        FakePhysicalScorer,
    )

    sentinel_outcome = object()

    monkeypatch.setattr(
        backend,
        "evaluate_ppo_physical_tti",
        lambda **kwargs: sentinel_outcome,
    )

    result = (
        backend
        .run_classical_runtime_scheduler(
            mode="pf_greedy",
            prepared=_prepared(),
            physical_inputs=(
                _physical_inputs()
            ),
            num_user_slots=2,
        )
    )

    assert result.mode == "pf_greedy"

    assert (
        result.physical_outcome
        is sentinel_outcome
    )

    assert tuple(
        result
        .allocation
        .candidate_by_user_slot
        .shape
    ) == (
        2,
        2,
    )

    assert (
        result.scheduler_num_phy_requests
        > 0
    )
