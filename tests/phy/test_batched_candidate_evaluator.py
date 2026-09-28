import torch

from oran_scheduler.phy.batched_candidate_evaluator import (
    evaluate_candidate_hypothesis_batch_same_rank_pattern,
)
from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.mu_mimo_rbg import (
    compute_isotropic_inter_cell_covariance,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.phy.schedule_evaluator import (
    evaluate_rbg_candidate_set,
)


def test_batched_candidate_phy_matches_scalar_rank1():
    device = "cuda:0"

    num_candidates = 3
    num_rx = 2
    num_bs = 2
    num_tx = 4
    num_subcarriers = 12

    torch.manual_seed(1234)

    real = torch.randn(
        (
            1,
            num_candidates,
            num_rx,
            num_bs,
            num_tx,
            1,
            num_subcarriers,
        ),
        device=device,
    )

    imag = torch.randn_like(
        real
    )

    h_freq = torch.complex(
        real,
        imag,
    )

    recommended_rank = torch.ones(
        (
            1,
            num_candidates,
        ),
        dtype=torch.long,
        device=device,
    )

    rx_combiners = torch.zeros(
        (
            1,
            num_candidates,
            1,
            2,
            num_rx,
        ),
        dtype=torch.complex64,
        device=device,
    )

    rx_combiners[
        :,
        :,
        :,
        0,
        0,
    ] = 1.0

    rx_combiners[
        :,
        :,
        :,
        1,
        1,
    ] = 1.0

    candidate_global_ue_indices = torch.arange(
        num_candidates,
        dtype=torch.long,
        device=device,
    )

    candidate_physical_ue_indices = torch.arange(
        num_candidates,
        dtype=torch.long,
        device=device,
    )

    tx_power = torch.tensor(
        1.0,
        dtype=torch.float32,
        device=device,
    )

    noise_power = torch.tensor(
        1.0e-3,
        dtype=torch.float32,
        device=device,
    )

    link_config = LinkAdaptationConfig(
        num_rbgs=1,
        device=device,
    )

    rate_config = RateConfig(
        device=device,
    )

    #
    # Build exactly the same candidate-wide PHY cache
    # used by the optimized scorer.
    #
    candidate_all_bs_channel = (
        h_freq[
            0,
            :,
            :,
            :,
            :,
            :,
            :,
        ]
        .permute(
            0,
            4,
            5,
            2,
            1,
            3,
        )
        .contiguous()
    )

    candidate_inter_cell_covariance = (
        compute_isotropic_inter_cell_covariance(
            all_bs_channel=(
                candidate_all_bs_channel
            ),
            serving_cell_index=0,
            tx_power_per_subcarrier_w=(
                tx_power
            ),
        )
    )

    candidate_serving_channel = (
        candidate_all_bs_channel[
            :,
            :,
            :,
            0,
            :,
            :,
        ]
    )

    candidate_ranks = (
        recommended_rank[
            0
        ]
    )

    candidate_rx_combiners = (
        rx_combiners[
            0
        ]
    )

    #
    # Two independent hypotheses, now evaluated
    # in one GPU batch.
    #
    batched = (
        evaluate_candidate_hypothesis_batch_same_rank_pattern(
            selected_candidate_indices=(
                torch.tensor(
                    [
                        [0],
                        [1],
                    ],
                    dtype=torch.long,
                    device=device,
                )
            ),
            rbg_indices=torch.tensor(
                [
                    0,
                    0,
                ],
                dtype=torch.long,
                device=device,
            ),
            candidate_ranks=(
                candidate_ranks
            ),
            candidate_rx_combiners=(
                candidate_rx_combiners
            ),
            candidate_serving_channel=(
                candidate_serving_channel
            ),
            candidate_inter_cell_covariance=(
                candidate_inter_cell_covariance
            ),
            csi_subcarrier_index=0,
            subcarriers_per_rbg=12,
            tx_power_per_subcarrier_w=(
                tx_power
            ),
            noise_power_per_subcarrier_w=(
                noise_power
            ),
            link_adaptation_config=(
                link_config
            ),
            rate_config=rate_config,
        )
    )

    scalar_0 = evaluate_rbg_candidate_set(
        selected_candidate_indices=(
            torch.tensor(
                [0],
                dtype=torch.long,
                device=device,
            )
        ),
        candidate_global_ue_indices=(
            candidate_global_ue_indices
        ),
        candidate_physical_ue_indices=(
            candidate_physical_ue_indices
        ),
        h_freq=h_freq,
        serving_cell_index=0,
        recommended_rank=(
            recommended_rank
        ),
        rx_combiners=(
            rx_combiners
        ),
        rbg_index=0,
        csi_subcarrier_index=0,
        subcarriers_per_rbg=12,
        tx_power_per_subcarrier_w=(
            tx_power
        ),
        noise_power_per_subcarrier_w=(
            noise_power
        ),
        link_adaptation_config=(
            link_config
        ),
        rate_config=rate_config,
    )

    scalar_1 = evaluate_rbg_candidate_set(
        selected_candidate_indices=(
            torch.tensor(
                [1],
                dtype=torch.long,
                device=device,
            )
        ),
        candidate_global_ue_indices=(
            candidate_global_ue_indices
        ),
        candidate_physical_ue_indices=(
            candidate_physical_ue_indices
        ),
        h_freq=h_freq,
        serving_cell_index=0,
        recommended_rank=(
            recommended_rank
        ),
        rx_combiners=(
            rx_combiners
        ),
        rbg_index=0,
        csi_subcarrier_index=0,
        subcarriers_per_rbg=12,
        tx_power_per_subcarrier_w=(
            tx_power
        ),
        noise_power_per_subcarrier_w=(
            noise_power
        ),
        link_adaptation_config=(
            link_config
        ),
        rate_config=rate_config,
    )

    torch.testing.assert_close(
        batched
        .target_compliant_rate_bps[
            0
        ],
        scalar_0.target_compliant_rate_bps,
    )

    torch.testing.assert_close(
        batched
        .target_compliant_rate_bps[
            1
        ],
        scalar_1.target_compliant_rate_bps,
    )

    torch.testing.assert_close(
        batched
        .total_target_compliant_rate_bps[
            0
        ],
        scalar_0
        .total_target_compliant_rate_bps,
    )

    torch.testing.assert_close(
        batched
        .total_target_compliant_rate_bps[
            1
        ],
        scalar_1
        .total_target_compliant_rate_bps,
    )

    torch.testing.assert_close(
        batched.rzf_alpha[
            0
        ],
        torch.as_tensor(
            scalar_0.rzf_alpha,
            dtype=batched.rzf_alpha.dtype,
            device=device,
        ),
    )

    torch.testing.assert_close(
        batched.rzf_alpha[
            1
        ],
        torch.as_tensor(
            scalar_1.rzf_alpha,
            dtype=batched.rzf_alpha.dtype,
            device=device,
        ),
    )