from types import SimpleNamespace

import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.rl.ppo_physical_score import (
    CachedPPOPhysicalRBGScorer,
    PPOPhysicalScoreInputs,
)


def build_inputs() -> PPOPhysicalScoreInputs:
    return PPOPhysicalScoreInputs(
        candidate_global_ue_indices=(
            torch.tensor(
                [
                    4,
                    8,
                    12,
                ],
                dtype=torch.long,
            )
        ),
        h_freq=torch.zeros(
            (
                1,
                1,
                1,
                1,
                1,
                1,
                12,
            ),
            dtype=torch.complex64,
        ),
        serving_cell_index=0,
        recommended_rank=torch.ones(
            (
                1,
                1,
            ),
            dtype=torch.long,
        ),
        rx_combiners=torch.zeros(
            (
                1,
                1,
                1,
                1,
                1,
            ),
            dtype=torch.complex64,
        ),
        csi_subcarrier_index=0,
        subcarriers_per_rbg=12,
        tx_power_per_subcarrier_w=1.0,
        noise_power_per_subcarrier_w=1.0e-3,
        link_adaptation_config=(
            LinkAdaptationConfig(
                num_rbgs=1,
                device="cpu",
            )
        ),
        rate_config=RateConfig(
            device="cpu",
        ),
    )


def test_physical_score_is_cached(
    monkeypatch,
):
    calls = []

    def fake_evaluate_rbg_candidate_set(
        **kwargs,
    ):
        calls.append(
            kwargs
        )

        selected = (
            kwargs[
                "selected_candidate_indices"
            ]
        )

        rates = torch.tensor(
            [
                8.0,
                4.0,
            ],
            dtype=torch.float32,
        )

        assert selected.tolist() == [
            0,
            2,
        ]

        return SimpleNamespace(
            total_target_compliant_rate_bps=(
                rates.sum()
            ),
            target_compliant_rate_bps=rates,
        )

    monkeypatch.setattr(
        (
            "oran_scheduler.rl."
            "ppo_physical_score."
            "evaluate_rbg_candidate_set"
        ),
        fake_evaluate_rbg_candidate_set,
    )

    scorer = CachedPPOPhysicalRBGScorer(
        build_inputs()
    )

    selected = torch.tensor(
        [
            0,
            2,
        ],
        dtype=torch.long,
    )

    first_score = scorer(
        selected,
        0,
    )

    second_score = scorer(
        selected,
        0,
    )


    assert len(calls) == 1

    assert scorer.num_score_requests == 2

    assert (
        scorer.num_unique_phy_evaluations
        == 1
    )

    torch.testing.assert_close(
        first_score.total_rate_bps,
        torch.tensor(
            12.0
        ),
    )

    torch.testing.assert_close(
        first_score
        .selected_candidate_rate_bps,
        torch.tensor(
            [
                8.0,
                4.0,
            ]
        ),
    )

    torch.testing.assert_close(
        second_score.total_rate_bps,
        first_score.total_rate_bps,
    )


def test_different_rbgs_use_different_cache_entries(
    monkeypatch,
):
    call_count = 0

    def fake_evaluate_rbg_candidate_set(
        **kwargs,
    ):
        nonlocal call_count

        call_count += 1

        rate = torch.tensor(
            float(
                kwargs[
                    "rbg_index"
                ]
                + 1
            ),
            dtype=torch.float32,
        )

        return SimpleNamespace(
            total_target_compliant_rate_bps=(
                rate
            ),
            target_compliant_rate_bps=(
                rate.reshape(
                    1
                )
            ),
        )

    monkeypatch.setattr(
        (
            "oran_scheduler.rl."
            "ppo_physical_score."
            "evaluate_rbg_candidate_set"
        ),
        fake_evaluate_rbg_candidate_set,
    )

    scorer = CachedPPOPhysicalRBGScorer(
        build_inputs()
    )

    selected = torch.tensor(
        [
            0,
        ],
        dtype=torch.long,
    )

    scorer(
        selected,
        0,
    )

    scorer(
        selected,
        1,
    )

    assert call_count == 2

    assert (
        scorer.num_unique_phy_evaluations
        == 2
    )


def test_candidate_order_is_preserved_in_cache(
    monkeypatch,
):
    call_count = 0

    def fake_evaluate_rbg_candidate_set(
        **kwargs,
    ):
        nonlocal call_count

        call_count += 1

        selected = (
            kwargs[
                "selected_candidate_indices"
            ]
        )

        rates = (
            selected
            .to(
                dtype=torch.float32
            )
            + 1.0
        )

        return SimpleNamespace(
            total_target_compliant_rate_bps=(
                rates.sum()
            ),
            target_compliant_rate_bps=rates,
        )

    monkeypatch.setattr(
        (
            "oran_scheduler.rl."
            "ppo_physical_score."
            "evaluate_rbg_candidate_set"
        ),
        fake_evaluate_rbg_candidate_set,
    )

    scorer = CachedPPOPhysicalRBGScorer(
        build_inputs()
    )

    scorer(
        torch.tensor(
            [
                0,
                2,
            ],
            dtype=torch.long,
        ),
        0,
    )

    scorer(
        torch.tensor(
            [
                2,
                0,
            ],
            dtype=torch.long,
        ),
        0,
    )

    assert call_count == 2

    assert (
        scorer.num_unique_phy_evaluations
        == 2
    )


