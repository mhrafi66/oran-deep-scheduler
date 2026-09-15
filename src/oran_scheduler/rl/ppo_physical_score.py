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
from oran_scheduler.phy.mu_mimo_rbg import (
    compute_isotropic_inter_cell_covariance,
)
from oran_scheduler.phy.batched_candidate_evaluator import (
    evaluate_candidate_hypothesis_batch_same_rank_pattern,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    PFGreedyRBGScore,
)

from oran_scheduler.utils.perf_timing import (
    perf_region,
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
        #
        # PERFORMANCE OPTIMIZATION.
        #
        # The current chunked PPO path stores at most
        # K=10 candidate UE rows. Inter-cell covariance
        # depends on:
        #
        #     channel realization
        #     UE
        #     RBG/subcarrier
        #
        # but NOT on which same-cell candidate set is
        # being hypothetically co-scheduled.
        #
        # Therefore compute it ONCE for every PF-TDS
        # candidate and reuse it across all PF-expert
        # and PPO-greedy counterfactual evaluations.
        #
        self._precomputed_candidate_serving_channel: (
            torch.Tensor | None
        ) = None

        self._precomputed_candidate_inter_cell_covariance: (
            torch.Tensor | None
        ) = None

        self._precomputed_candidate_ranks: (
            torch.Tensor | None
        ) = None

        self._precomputed_candidate_rx_combiners: (
            torch.Tensor | None
        ) = None

        self._candidate_ranks_cpu: (
            torch.Tensor | None
        ) = None

        self._precompute_candidate_phy()


    def _precompute_candidate_phy(
        self,
    ) -> None:
        """
        Precompute candidate-only serving channels and
        inter-cell covariance.

        OPEN-REPRODUCTION ENGINEERING OPTIMIZATION.

        No PHY model is changed.

        Input h_freq:

            [B, PHY_UE, RX, BS, TX, symbol, subcarrier]

        Cached serving channel:

            [candidate, symbol, subcarrier, RX, TX]

        Cached inter-cell covariance:

            [candidate, symbol, subcarrier, RX, RX]

        The optimization is enabled for the compact
        chunked-PHY path, where candidate physical UE
        indices are explicitly available.
        """

        physical_indices = (
            self.inputs
            .candidate_physical_ue_indices
        )

        #
        # Preserve compatibility with older/non-chunked
        # callers. They continue through the original
        # evaluator path.
        #
        if physical_indices is None:
            return

        valid_candidate_mask = (
            physical_indices >= 0
        )

        safe_physical_indices = torch.where(
            valid_candidate_mask,
            physical_indices,
            torch.zeros_like(
                physical_indices
            ),
        )

        num_phy_ues = int(
            self.inputs.h_freq.shape[1]
        )

        if torch.any(
            safe_physical_indices
            >= num_phy_ues
        ):
            raise ValueError(
                "Candidate physical UE index exceeds "
                "the stored PHY UE dimension."
            )

        batch_index = (
            self.inputs.batch_index
        )

        with torch.no_grad():

            #
            # candidate_h_freq:
            #
            # [candidate, RX, BS, TX, symbol, SC]
            #
            candidate_h_freq = (
                self.inputs
                .h_freq[
                    batch_index,
                    safe_physical_indices,
                    :,
                    :,
                    :,
                    :,
                    :,
                ]
            )

            #
            # Padded candidate rows use physical index 0
            # only as a safe gather position.
            # Blank them immediately.
            #
            candidate_valid_broadcast = (
                valid_candidate_mask.reshape(
                    -1,
                    1,
                    1,
                    1,
                    1,
                    1,
                )
            )

            candidate_h_freq = torch.where(
                candidate_valid_broadcast,
                candidate_h_freq,
                torch.zeros_like(
                    candidate_h_freq
                ),
            )

            #
            # Candidate-wide RI and receive combiners.
            #
            # These are also invariant across the many
            # hypothetical PF-expert/reward schedules.
            #
            candidate_ranks = (
                self.inputs
                .recommended_rank[
                    batch_index,
                    safe_physical_indices,
                ]
            )

            candidate_ranks = torch.where(
                valid_candidate_mask,
                candidate_ranks,
                torch.ones_like(
                    candidate_ranks
                ),
            )

            candidate_rx_combiners = (
                self.inputs
                .rx_combiners[
                    batch_index,
                    safe_physical_indices,
                    :,
                    :,
                    :,
                ]
            )

            candidate_rx_combiners = torch.where(
                valid_candidate_mask[
                    :,
                    None,
                    None,
                    None,
                ],
                candidate_rx_combiners,
                torch.zeros_like(
                    candidate_rx_combiners
                ),
            )

            #
            # Convert to the common MU-MIMO layout:
            #
            # [candidate, symbol, SC, BS, RX, TX]
            #
            candidate_all_bs_channel = (
                candidate_h_freq
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

            #
            # One large GPU operation replaces thousands
            # of repeated per-hypothesis covariance
            # calculations.
            #
            inter_cell_covariance = (
                compute_isotropic_inter_cell_covariance(
                    all_bs_channel=(
                        candidate_all_bs_channel
                    ),
                    serving_cell_index=(
                        self.inputs
                        .serving_cell_index
                    ),
                    tx_power_per_subcarrier_w=(
                        self.inputs
                        .tx_power_per_subcarrier_w
                    ),
                )
            )

            serving_channel = (
                candidate_all_bs_channel[
                    :,
                    :,
                    :,
                    self.inputs.serving_cell_index,
                    :,
                    :,
                ]
                .contiguous()
            )

            self._precomputed_candidate_serving_channel = (
                serving_channel
            )

            self._precomputed_candidate_inter_cell_covariance = (
                inter_cell_covariance
            )
            self._precomputed_candidate_ranks = (
                candidate_ranks
            )

            self._precomputed_candidate_rx_combiners = (
                candidate_rx_combiners
            )

            #
            # One CPU copy per scorer/cell/TTI.
            #
            # This replaces thousands of candidate-rank
            # GPU -> CPU synchronizations while grouping
            # hypotheses by rank pattern.
            #
            self._candidate_ranks_cpu = (
                candidate_ranks
                .detach()
                .cpu()
            )

    @property
    def precomputed_candidate_serving_channel(
        self,
    ) -> torch.Tensor | None:
        return (
            self
            ._precomputed_candidate_serving_channel
        )


    @property
    def precomputed_candidate_inter_cell_covariance(
        self,
    ) -> torch.Tensor | None:
        return (
            self
            ._precomputed_candidate_inter_cell_covariance
        )

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

    def score_many(
        self,
        requests: list[
            tuple[
                tuple[int, ...],
                int,
            ]
        ],
    ) -> list[PFGreedyRBGScore]:
        """
        Score many RBG/candidate-set hypotheses.

        Each request is:

            (
                candidate_tuple,
                rbg_index,
            )

        Example:

            [
                ((0,), 0),
                ((1,), 0),
                ((2,), 0),
                ((0,), 1),
                ((1,), 1),
                ...
            ]

        PERFORMANCE PATH:

        Hypotheses are grouped by physical rank pattern
        and each group is evaluated as one GPU batch.

        Generic/non-chunked callers retain the original
        scalar path.
        """

        if not requests:
            return []

        batch_ready = (
            self
            ._precomputed_candidate_serving_channel
            is not None
            and self
            ._precomputed_candidate_inter_cell_covariance
            is not None
            and self
            ._precomputed_candidate_ranks
            is not None
            and self
            ._precomputed_candidate_rx_combiners
            is not None
            and self
            ._candidate_ranks_cpu
            is not None
        )

        #
        # Backward-compatible fallback.
        #
        if not batch_ready:
            output = []

            device = (
                self.inputs
                .h_freq
                .device
            )

            for (
                candidate_tuple,
                rbg_index,
            ) in requests:

                selected_candidates = torch.tensor(
                    candidate_tuple,
                    dtype=torch.long,
                    device=device,
                )

                output.append(
                    self(
                        selected_candidates,
                        rbg_index,
                    )
                )

            return output

        assert (
            self._precomputed_candidate_ranks
            is not None
        )

        assert (
            self._precomputed_candidate_rx_combiners
            is not None
        )

        assert (
            self._precomputed_candidate_serving_channel
            is not None
        )

        assert (
            self
            ._precomputed_candidate_inter_cell_covariance
            is not None
        )

        assert (
            self._candidate_ranks_cpu
            is not None
        )

        num_candidates = int(
            self.inputs
            .candidate_global_ue_indices
            .numel()
        )

        num_subcarriers = int(
            self.inputs
            .h_freq
            .shape[-1]
        )

        if (
            num_subcarriers
            % self.inputs.subcarriers_per_rbg
            != 0
        ):
            raise ValueError(
                "PHY subcarrier count is not divisible "
                "by subcarriers_per_rbg."
            )

        num_rbgs = (
            num_subcarriers
            // self.inputs.subcarriers_per_rbg
        )

        #
        # This count intentionally remains the number
        # of logical score requests, matching the
        # previous scalar implementation.
        #
        self._num_score_requests += len(
            requests
        )

        request_keys: list[
            tuple[
                int,
                tuple[int, ...],
            ]
        ] = []

        #
        # dict preserves insertion order.
        #
        pending_keys: dict[
            tuple[
                int,
                tuple[int, ...],
            ],
            None,
        ] = {}

        for (
            candidate_tuple,
            rbg_index,
        ) in requests:

            rbg_index = int(
                rbg_index
            )

            candidate_tuple = tuple(
                int(value)
                for value
                in candidate_tuple
            )

            if not (
                0
                <= rbg_index
                < num_rbgs
            ):
                raise ValueError(
                    "RBG index is invalid."
                )

            for candidate_index in (
                candidate_tuple
            ):
                if not (
                    0
                    <= candidate_index
                    < num_candidates
                ):
                    raise ValueError(
                        "Candidate index is invalid."
                    )

            cache_key = (
                rbg_index,
                candidate_tuple,
            )

            request_keys.append(
                cache_key
            )

            if (
                cache_key
                not in self._score_cache
            ):
                pending_keys[
                    cache_key
                ] = None

        #
        # Empty scheduled set:
        #
        # no physical transmission, therefore zero
        # throughput. Cache it without invoking RZF.
        #
        zero_dtype = (
            self.inputs
            .h_freq
            .real
            .dtype
        )

        device = (
            self.inputs
            .h_freq
            .device
        )

        grouped_keys: dict[
            tuple[int, ...],
            list[
                tuple[
                    int,
                    tuple[int, ...],
                ]
            ],
        ] = {}

        for cache_key in pending_keys:

            (
                rbg_index,
                candidate_tuple,
            ) = cache_key

            if len(
                candidate_tuple
            ) == 0:
                self._score_cache[
                    cache_key
                ] = PFGreedyRBGScore(
                    total_rate_bps=torch.zeros(
                        (),
                        dtype=zero_dtype,
                        device=device,
                    ),
                    selected_candidate_rate_bps=(
                        torch.zeros(
                            (
                                0,
                            ),
                            dtype=zero_dtype,
                            device=device,
                        )
                    ),
                )

                continue

            rank_pattern = tuple(
                int(
                    self
                    ._candidate_ranks_cpu[
                        candidate_index
                    ]
                    .item()
                )
                for candidate_index
                in candidate_tuple
            )

            grouped_keys.setdefault(
                rank_pattern,
                [],
            ).append(
                cache_key
            )

        #
        # Usually only a handful of groups exist:
        #
        #   (1,)
        #   (2,)
        #   (1,1)
        #   (1,2)
        #   (2,1)
        #   ...
        #
        # Each group may contain tens/hundreds of
        # RBG/action hypotheses and becomes ONE GPU
        # batched physical evaluation.
        #
        with torch.no_grad():

            for (
                rank_pattern,
                group_keys,
            ) in grouped_keys.items():

                del rank_pattern

                selected_candidate_indices = (
                    torch.tensor(
                        [
                            list(
                                candidate_tuple
                            )
                            for (
                                _,
                                candidate_tuple,
                            )
                            in group_keys
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                )

                rbg_indices = torch.tensor(
                    [
                        rbg_index
                        for (
                            rbg_index,
                            _,
                        )
                        in group_keys
                    ],
                    dtype=torch.long,
                    device=device,
                )

                with perf_region(
                    "scorer.batched_phy",
                    device=device,
                ):
                    batch_result = (
                        evaluate_candidate_hypothesis_batch_same_rank_pattern(
                            selected_candidate_indices=(
                                selected_candidate_indices
                            ),
                            rbg_indices=(
                                rbg_indices
                            ),
                            candidate_ranks=(
                                self
                                ._precomputed_candidate_ranks
                            ),
                            candidate_rx_combiners=(
                                self
                                ._precomputed_candidate_rx_combiners
                            ),
                            candidate_serving_channel=(
                                self
                                ._precomputed_candidate_serving_channel
                            ),
                            candidate_inter_cell_covariance=(
                                self
                                ._precomputed_candidate_inter_cell_covariance
                            ),
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
                        )
                    )

                for (
                    row_index,
                    cache_key,
                ) in enumerate(
                    group_keys
                ):

                    self._score_cache[
                        cache_key
                    ] = PFGreedyRBGScore(
                        total_rate_bps=(
                            batch_result
                            .total_target_compliant_rate_bps[
                                row_index
                            ]
                        ),
                        selected_candidate_rate_bps=(
                            batch_result
                            .target_compliant_rate_bps[
                                row_index,
                                :,
                            ]
                        ),
                    )

        return [
            self._score_cache[
                cache_key
            ]
            for cache_key
            in request_keys
        ]


    def score_many_pf(
        self,
        requests: list[
            tuple[
                tuple[int, ...],
                int,
            ]
        ],
        *,
        past_average_throughput: torch.Tensor,
        denominator_epsilon: float,
    ) -> torch.Tensor:
        """
        Batched physical scoring followed by batched
        PF objective calculation.

        Returns:

            [request]

        This deliberately calculates PF sums in only
        a few tensor batches instead of launching one
        tiny division/sum kernel per hypothesis.
        """

        if denominator_epsilon <= 0.0:
            raise ValueError(
                "denominator_epsilon must be positive."
            )

        num_candidates = int(
            self.inputs
            .candidate_global_ue_indices
            .numel()
        )

        if tuple(
            past_average_throughput.shape
        ) != (
            num_candidates,
        ):
            raise ValueError(
                "past_average_throughput must have "
                "shape [candidate]."
            )

        if (
            past_average_throughput.device
            != self.inputs.h_freq.device
        ):
            raise ValueError(
                "past_average_throughput is on the "
                "wrong device."
            )

        # scores = self.score_many(
        #     requests
        # )
        with perf_region(
            "scorer.score_many_total",
            device=(
                past_average_throughput
                .device
            ),
        ):
            scores = self.score_many(
                requests
            )

        num_requests = len(
            requests
        )

        pf_sum = torch.zeros(
            num_requests,
            dtype=(
                past_average_throughput
                .dtype
            ),
            device=(
                past_average_throughput
                .device
            ),
        )

        #
        # Candidate-set length is at most four in
        # paper training, so there are only a few
        # groups here.
        #
        by_length: dict[
            int,
            list[int],
        ] = {}

        for request_index, (
            candidate_tuple,
            _,
        ) in enumerate(
            requests
        ):
            by_length.setdefault(
                len(
                    candidate_tuple
                ),
                [],
            ).append(
                request_index
            )

        for (
            selected_count,
            request_positions,
        ) in by_length.items():

            if selected_count == 0:
                continue

            position_tensor = torch.tensor(
                request_positions,
                dtype=torch.long,
                device=(
                    past_average_throughput
                    .device
                ),
            )

            selected_candidate_indices = (
                torch.tensor(
                    [
                        list(
                            requests[
                                position
                            ][0]
                        )
                        for position
                        in request_positions
                    ],
                    dtype=torch.long,
                    device=(
                        past_average_throughput
                        .device
                    ),
                )
            )

            selected_rates = torch.stack(
                [
                    scores[
                        position
                    ]
                    .selected_candidate_rate_bps
                    for position
                    in request_positions
                ],
                dim=0,
            )

            denominator = (
                past_average_throughput[
                    selected_candidate_indices
                ]
            )

            denominator = torch.clamp(
                denominator,
                min=denominator_epsilon,
            )

            grouped_pf_sum = (
                (
                    selected_rates
                    / denominator
                )
                .sum(
                    dim=-1
                )
            )

            pf_sum[
                position_tensor
            ] = grouped_pf_sum

        return pf_sum
    
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
                precomputed_candidate_serving_channel=(
                    self
                    ._precomputed_candidate_serving_channel
                ),

                precomputed_candidate_inter_cell_covariance=(
                    self
                    ._precomputed_candidate_inter_cell_covariance
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


