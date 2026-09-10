from dataclasses import dataclass

import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.phy.schedule_evaluator import (
    evaluate_rbg_candidate_set,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    PFGreedyRBGScore,
)


@dataclass(frozen=True)
class PPOPhysicalScoreInputs:
    """
    Physical inputs required to score hypothetical
    PPO RBG allocations.

    selected candidate sets are expressed in PF-TDS
    candidate-slot indices, not global UE IDs.

    The underlying physical evaluator performs the
    common reproduction PHY:

        candidate set
            ->
        rank expansion
            ->
        RZF
            ->
        post-RZF MRC
            ->
        SINR
            ->
        MCS / TBLER
            ->
        target-compliant rate
    """

    candidate_global_ue_indices: torch.Tensor

    h_freq: torch.Tensor

    serving_cell_index: int

    recommended_rank: torch.Tensor

    rx_combiners: torch.Tensor

    csi_subcarrier_index: int

    subcarriers_per_rbg: int

    tx_power_per_subcarrier_w: (
        float | torch.Tensor
    )

    noise_power_per_subcarrier_w: (
        float | torch.Tensor
    )

    link_adaptation_config: (
        LinkAdaptationConfig
    )

    rate_config: RateConfig

    #
    # OPEN-REPRODUCTION ENGINEERING MECHANISM.
    #
    # None:
    #     global UE ID == h_freq UE-axis position
    #
    # Chunked PHY:
    #     global identity and local PHY storage
    #     position may differ.
    #
    candidate_physical_ue_indices: (
        torch.Tensor | None
    ) = None

    batch_index: int = 0


class CachedPPOPhysicalRBGScorer:
    """
    Adapt the common RBG PHY evaluator to the
    PFGreedyRBGScore callback expected by the PPO
    greedy-search code.

    Physical evaluations are cached because the same
    hypothetical candidate set may be requested again
    while replaying later scheduler layers.
    """

    def __init__(
        self,
        inputs: PPOPhysicalScoreInputs,
    ) -> None:
        if (
            inputs
            .candidate_global_ue_indices
            .ndim
            != 1
        ):
            raise ValueError(
                "candidate_global_ue_indices must "
                "have shape [candidate]."
            )

        physical_indices = (
            inputs
            .candidate_physical_ue_indices
        )

        if physical_indices is not None:
            if physical_indices.ndim != 1:
                raise ValueError(
                    "candidate_physical_ue_indices "
                    "must have shape [candidate]."
                )

            if tuple(
                physical_indices.shape
            ) != tuple(
                inputs
                .candidate_global_ue_indices
                .shape
            ):
                raise ValueError(
                    "Global and physical candidate "
                    "UE mappings must have the same "
                    "shape."
                )

            if (
                physical_indices.device
                != inputs.h_freq.device
            ):
                raise ValueError(
                    "candidate_physical_ue_indices "
                    "and h_freq must be on the same "
                    "device."
                )

            if torch.is_floating_point(
                physical_indices
            ):
                raise ValueError(
                    "candidate_physical_ue_indices "
                    "must use an integer dtype."
                )

            if (
                physical_indices.dtype
                == torch.bool
            ):
                raise ValueError(
                    "candidate_physical_ue_indices "
                    "cannot use torch.bool."
                )

        if inputs.subcarriers_per_rbg <= 0:
            raise ValueError(
                "subcarriers_per_rbg must be positive."
            )

        self.inputs = inputs

        self._score_cache: dict[
            tuple[
                int,
                tuple[int, ...],
            ],
            PFGreedyRBGScore,
        ] = {}

        self._num_score_requests = 0


    @property
    def num_score_requests(
        self,
    ) -> int:
        """
        Total number of times the greedy-search code
        requested a score.
        """

        return self._num_score_requests

    @property
    def num_unique_phy_evaluations(
        self,
    ) -> int:
        """
        Number of unique RBG/candidate-set combinations
        that actually required physical evaluation.
        """

        return len(
            self._score_cache
        )


    @staticmethod
    def _candidate_tuple(
        selected_candidates: torch.Tensor,
    ) -> tuple[int, ...]:
        if selected_candidates.ndim != 1:
            raise ValueError(
                "selected_candidates must have "
                "shape [scheduled_UE]."
            )

        return tuple(
            int(value)
            for value
            in (
                selected_candidates
                .detach()
                .cpu()
                .tolist()
            )
        )


    def __call__(
        self,
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:
        self._num_score_requests += 1

        candidate_tuple = (
            self._candidate_tuple(
                selected_candidates
            )
        )

        cache_key = (
            rbg_index,
            candidate_tuple,
        )

        cached_score = (
            self._score_cache.get(
                cache_key
            )
        )

        if cached_score is not None:
            return cached_score


        evaluation = (
            evaluate_rbg_candidate_set(
                selected_candidate_indices=(
                    selected_candidates
                ),
                candidate_global_ue_indices=(
                    self.inputs
                    .candidate_global_ue_indices
                ),

                candidate_physical_ue_indices=(
                    self.inputs
                    .candidate_physical_ue_indices
                ),
                h_freq=(
                    self.inputs
                    .h_freq
                ),
                serving_cell_index=(
                    self.inputs
                    .serving_cell_index
                ),
                recommended_rank=(
                    self.inputs
                    .recommended_rank
                ),
                rx_combiners=(
                    self.inputs
                    .rx_combiners
                ),
                rbg_index=rbg_index,
                csi_subcarrier_index=(
                    self.inputs
                    .csi_subcarrier_index
                ),
                subcarriers_per_rbg=(
                    self.inputs
                    .subcarriers_per_rbg
                ),
                tx_power_per_subcarrier_w=(
                    self.inputs
                    .tx_power_per_subcarrier_w
                ),
                noise_power_per_subcarrier_w=(
                    self.inputs
                    .noise_power_per_subcarrier_w
                ),
                link_adaptation_config=(
                    self.inputs
                    .link_adaptation_config
                ),
                rate_config=(
                    self.inputs
                    .rate_config
                ),
                batch_index=(
                    self.inputs
                    .batch_index
                ),
            )
        )


        score = PFGreedyRBGScore(
            total_rate_bps=(
                evaluation
                .total_target_compliant_rate_bps
            ),
            selected_candidate_rate_bps=(
                evaluation
                .target_compliant_rate_bps
            ),
        )

        self._score_cache[
            cache_key
        ] = score

        return score


