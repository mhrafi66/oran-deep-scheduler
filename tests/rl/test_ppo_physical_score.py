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


def test_physical_scorer_forwards_local_ue_mapping(
    monkeypatch,
):
    calls = []

    def fake_evaluate_rbg_candidate_set(
        **kwargs,
    ):
        calls.append(
            kwargs
        )

        return SimpleNamespace(
            total_target_compliant_rate_bps=(
                torch.tensor(
                    9.0,
                    dtype=torch.float32,
                )
            ),
            target_compliant_rate_bps=(
                torch.tensor(
                    [
                        9.0,
                    ],
                    dtype=torch.float32,
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

    base = build_inputs()

    #
    # This test uses three explicit local PHY UE
    # mappings:
    #
    #     candidate 0 -> PHY UE 0
    #     candidate 1 -> PHY UE 1
    #     candidate 2 -> PHY UE 2
    #
    # The generic build_inputs() fixture contains only
    # one dummy PHY UE because older scorer tests
    # monkeypatch the actual PHY evaluator.
    #
    # Candidate-PHY precomputation now legitimately
    # requires the synthetic channel tensors to expose
    # those three physical UE rows.
    #
    h_freq = base.h_freq.repeat(
        1,
        3,
        1,
        1,
        1,
        1,
        1,
    )

    recommended_rank = (
        base.recommended_rank.repeat(
            1,
            3,
        )
    )

    rx_combiners = (
        base.rx_combiners.repeat(
            1,
            3,
            1,
            1,
            1,
        )
    )

    inputs = PPOPhysicalScoreInputs(
        candidate_global_ue_indices=(
            torch.tensor(
                [
                    103,
                    151,
                    317,
                ],
                dtype=torch.long,
            )
        ),

        candidate_physical_ue_indices=(
            torch.tensor(
                [
                    0,
                    1,
                    2,
                ],
                dtype=torch.long,
            )
        ),

        h_freq=h_freq,

        serving_cell_index=(
            base.serving_cell_index
        ),

        recommended_rank=(
            recommended_rank
        ),

        rx_combiners=(
            rx_combiners
        ),

        csi_subcarrier_index=(
            base.csi_subcarrier_index
        ),

        subcarriers_per_rbg=(
            base.subcarriers_per_rbg
        ),

        tx_power_per_subcarrier_w=(
            base.tx_power_per_subcarrier_w
        ),

        noise_power_per_subcarrier_w=(
            base.noise_power_per_subcarrier_w
        ),

        link_adaptation_config=(
            base.link_adaptation_config
        ),

        rate_config=(
            base.rate_config
        ),

        batch_index=(
            base.batch_index
        ),
    )

    scorer = (
        CachedPPOPhysicalRBGScorer(
            inputs
        )
    )

    scorer(
        torch.tensor(
            [
                1,
            ],
            dtype=torch.long,
        ),
        0,
    )

    assert len(
        calls
    ) == 1

    torch.testing.assert_close(
        calls[0][
            "candidate_global_ue_indices"
        ],
        torch.tensor(
            [
                103,
                151,
                317,
            ],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        calls[0][
            "candidate_physical_ue_indices"
        ],
        torch.tensor(
            [
                0,
                1,
                2,
            ],
            dtype=torch.long,
        ),
    )


def test_chunked_physical_scorer_precomputes_candidate_phy():
    num_candidates = 3
    num_rx = 2
    num_bs = 2
    num_tx = 4
    num_subcarriers = 12

    h_freq = torch.randn(
        (
            1,
            num_candidates,
            num_rx,
            num_bs,
            num_tx,
            1,
            num_subcarriers,
        ),
        dtype=torch.complex64,
    )

    inputs = PPOPhysicalScoreInputs(
        candidate_global_ue_indices=(
            torch.tensor(
                [
                    101,
                    205,
                    317,
                ],
                dtype=torch.long,
            )
        ),

        candidate_physical_ue_indices=(
            torch.tensor(
                [
                    0,
                    1,
                    2,
                ],
                dtype=torch.long,
            )
        ),

        h_freq=h_freq,

        serving_cell_index=0,

        recommended_rank=torch.ones(
            (
                1,
                num_candidates,
            ),
            dtype=torch.long,
        ),

        rx_combiners=torch.ones(
            (
                1,
                num_candidates,
                1,
                2,
                num_rx,
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

        rate_config=(
            RateConfig(
                device="cpu",
            )
        ),
    )

    scorer = CachedPPOPhysicalRBGScorer(
        inputs
    )

    serving = (
        scorer
        .precomputed_candidate_serving_channel
    )

    covariance = (
        scorer
        .precomputed_candidate_inter_cell_covariance
    )

    assert serving is not None
    assert covariance is not None

    assert tuple(
        serving.shape
    ) == (
        num_candidates,
        1,
        num_subcarriers,
        num_rx,
        num_tx,
    )

    assert tuple(
        covariance.shape
    ) == (
        num_candidates,
        1,
        num_subcarriers,
        num_rx,
        num_rx,
    )

    assert torch.isfinite(
        serving.real
    ).all()

    assert torch.isfinite(
        covariance.real
    ).all()