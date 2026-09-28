from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    PFTimeDomainResult,
    run_pf_tds,
)
from oran_scheduler.schedulers.throughput_history import (
    CellThroughputHistoryUpdate,
    update_cell_throughput_history,
)
from oran_scheduler.state.one_lds_decision import (
    OneLDSDecisionInputs,
)


@dataclass(frozen=True)
class OneLDSCellTTIObservation:
    """
    Scheduler-facing serving-UE information for one
    cell at one TTI.

    All quantities are in the padded serving-UE
    layout BEFORE PF TDS.

    Shapes:

        serving_global_ue_indices:
            [serving_ue]

        serving_ue_valid_mask:
            [serving_ue]

        td_instantaneous_rate_bps:
            [serving_ue]

        rank:
            [serving_ue]

        dl_buffer:
            [serving_ue]

        wideband_cqi:
            [serving_ue]

        subband_cqi:
            [serving_ue, RBG]

        precoder_directions:
            [serving_ue, RBG, mode, TX_ant]
    """

    serving_global_ue_indices: torch.Tensor

    serving_ue_valid_mask: torch.Tensor

    td_instantaneous_rate_bps: torch.Tensor

    rank: torch.Tensor

    dl_buffer: torch.Tensor

    wideband_cqi: torch.Tensor

    subband_cqi: torch.Tensor

    precoder_directions: torch.Tensor


@dataclass(frozen=True)
class PreparedOneLDSCellTTI:
    """
    Complete candidate-level scheduler input for one
    cell and one TTI after PF TDS.

    candidate_serving_indices:
        [candidate]

        Indices into the padded serving-UE layout.

    candidate_global_ue_indices:
        [candidate]

        Global UE IDs. Invalid candidate slots are -1.

    candidate_valid_mask:
        [candidate]

    decision_inputs:
        Candidate-level information consumed by the
        existing 1LDS user-slot loop.
    """

    tti_index: int

    tds_result: PFTimeDomainResult

    candidate_serving_indices: torch.Tensor

    candidate_global_ue_indices: torch.Tensor

    candidate_valid_mask: torch.Tensor

    decision_inputs: OneLDSDecisionInputs


class OneLDSCellTTIStateManager:
    """
    Maintain one cell's scheduler state across TTIs.

    PAPER-SPECIFIED:
        PF history is updated from realized
        throughput using exponential smoothing.

    OPEN-REPRODUCTION PARAMETER:
        throughput_forgetting_factor.

    TEMPORARY SCAFFOLDING:
        The padded serving-UE layout must remain
        unchanged across TTIs.

        Radio reports, rank, buffer, rates, and
        precoder directions may still change every
        TTI.
    """

    def __init__(
        self,
        *,
        initial_average_throughput_bps: torch.Tensor,
        serving_global_ue_indices: torch.Tensor,
        serving_ue_valid_mask: torch.Tensor,
        tds_config: PFTimeDomainConfig,
        throughput_forgetting_factor: float,
    ) -> None:
        if (
            initial_average_throughput_bps.ndim
            != 1
        ):
            raise ValueError(
                "initial_average_throughput_bps "
                "must have shape [serving_ue]."
            )

        expected_shape = (
            initial_average_throughput_bps.shape
        )

        if tuple(
            serving_global_ue_indices.shape
        ) != tuple(
            expected_shape
        ):
            raise ValueError(
                "serving_global_ue_indices must "
                "match the throughput-history shape."
            )

        if tuple(
            serving_ue_valid_mask.shape
        ) != tuple(
            expected_shape
        ):
            raise ValueError(
                "serving_ue_valid_mask must match "
                "the throughput-history shape."
            )

        if (
            serving_ue_valid_mask.dtype
            != torch.bool
        ):
            raise ValueError(
                "serving_ue_valid_mask must use "
                "torch.bool."
            )

        device = (
            initial_average_throughput_bps.device
        )

        if (
            serving_global_ue_indices.device
            != device
            or serving_ue_valid_mask.device
            != device
        ):
            raise ValueError(
                "Initial serving-layout tensors "
                "must be on the same device."
            )

        if torch.any(
            initial_average_throughput_bps
            < 0.0
        ):
            raise ValueError(
                "Initial throughput history cannot "
                "be negative."
            )

        if not torch.isfinite(
            initial_average_throughput_bps
        ).all():
            raise ValueError(
                "Initial throughput history contains "
                "non-finite values."
            )

        self.tds_config = tds_config

        self.throughput_forgetting_factor = (
            throughput_forgetting_factor
        )

        self._serving_global_ue_indices = (
            serving_global_ue_indices
            .detach()
            .clone()
        )

        self._serving_ue_valid_mask = (
            serving_ue_valid_mask
            .detach()
            .clone()
        )

        self._average_throughput_bps = (
            initial_average_throughput_bps
            .detach()
            .clone()
        )

        self._active_tti: (
            PreparedOneLDSCellTTI | None
        ) = None

        self._last_completed_tti_index: (
            int | None
        ) = None


    def reset_average_throughput_bps(
        self,
        value_bps: float | torch.Tensor,
    ) -> None:
        """
        Reset persistent PF throughput history.

        Intended for controlled restart / state-loss
        experiments.

        The serving-UE layout and temporal TTI index are
        preserved.

        A reset is legal only between TTIs.
        """

        if self._active_tti is not None:
            raise RuntimeError(
                "Cannot reset PF history while a TTI "
                "is active."
            )

        if isinstance(
            value_bps,
            torch.Tensor,
        ):
            if tuple(
                value_bps.shape
            ) != tuple(
                self._average_throughput_bps.shape
            ):
                raise ValueError(
                    "PF reset tensor must have shape "
                    "[serving_ue]."
                )

            if (
                value_bps.device
                != self._average_throughput_bps.device
            ):
                raise ValueError(
                    "PF reset tensor is on the wrong "
                    "device."
                )

            reset_value = (
                value_bps
                .to(
                    dtype=(
                        self
                        ._average_throughput_bps
                        .dtype
                    )
                )
                .detach()
                .clone()
            )

        else:
            reset_scalar = float(
                value_bps
            )

            if (
                not torch.isfinite(
                    torch.tensor(
                        reset_scalar
                    )
                )
                or reset_scalar < 0.0
            ):
                raise ValueError(
                    "PF reset value must be finite "
                    "and non-negative."
                )

            reset_value = torch.full_like(
                self._average_throughput_bps,
                fill_value=reset_scalar,
            )

        if not torch.isfinite(
            reset_value
        ).all():
            raise ValueError(
                "PF reset history contains "
                "non-finite values."
            )

        if torch.any(
            reset_value < 0.0
        ):
            raise ValueError(
                "PF reset history cannot be negative."
            )

        #
        # Invalid padded serving slots should not carry
        # artificial PF history.
        #
        self._average_throughput_bps = torch.where(
            self._serving_ue_valid_mask,
            reset_value,
            torch.zeros_like(
                reset_value
            ),
        )


    @property
    def current_average_throughput_bps(
        self,
    ) -> torch.Tensor:
        return (
            self._average_throughput_bps
            .detach()
            .clone()
        )


    @property
    def has_active_tti(
        self,
    ) -> bool:
        return (
            self._active_tti
            is not None
        )


    def _validate_observation(
        self,
        observation: OneLDSCellTTIObservation,
    ) -> None:
        num_serving_ues = (
            self._average_throughput_bps
            .shape[0]
        )

        expected_ue_shape = (
            num_serving_ues,
        )

        scalar_ue_tensors = {
            "serving_global_ue_indices": (
                observation
                .serving_global_ue_indices
            ),
            "serving_ue_valid_mask": (
                observation
                .serving_ue_valid_mask
            ),
            "td_instantaneous_rate_bps": (
                observation
                .td_instantaneous_rate_bps
            ),
            "rank": observation.rank,
            "dl_buffer": observation.dl_buffer,
            "wideband_cqi": (
                observation.wideband_cqi
            ),
        }

        for name, tensor in (
            scalar_ue_tensors.items()
        ):
            if tuple(
                tensor.shape
            ) != expected_ue_shape:
                raise ValueError(
                    f"{name} must have shape "
                    "[serving_ue]."
                )

        if observation.subband_cqi.ndim != 2:
            raise ValueError(
                "subband_cqi must have shape "
                "[serving_ue, RBG]."
            )

        if (
            observation
            .subband_cqi
            .shape[0]
            != num_serving_ues
        ):
            raise ValueError(
                "subband_cqi serving-UE dimension "
                "is incorrect."
            )

        if (
            observation
            .precoder_directions
            .ndim
            != 4
        ):
            raise ValueError(
                "precoder_directions must have "
                "shape "
                "[serving_ue, RBG, mode, TX_ant]."
            )

        if (
            observation
            .precoder_directions
            .shape[0]
            != num_serving_ues
        ):
            raise ValueError(
                "precoder_directions serving-UE "
                "dimension is incorrect."
            )

        if (
            observation
            .precoder_directions
            .shape[1]
            != observation
            .subband_cqi
            .shape[1]
        ):
            raise ValueError(
                "CQI and precoder data disagree on "
                "the number of RBGs."
            )

        device = (
            self._average_throughput_bps
            .device
        )

        for name, tensor in (
            scalar_ue_tensors.items()
        ):
            if tensor.device != device:
                raise ValueError(
                    f"{name} is on the wrong device."
                )

        if (
            observation.subband_cqi.device
            != device
            or observation
            .precoder_directions
            .device
            != device
        ):
            raise ValueError(
                "Radio observation must be on the "
                "same device as scheduler history."
            )

        #
        # TEMPORARY SCAFFOLDING:
        # serving layout is stable for this
        # single-cell multi-TTI component.
        #
        if not torch.equal(
            observation
            .serving_global_ue_indices,
            self._serving_global_ue_indices,
        ):
            raise ValueError(
                "Serving-UE identity/order changed. "
                "This single-cell state manager does "
                "not yet support reassociation."
            )

        if not torch.equal(
            observation
            .serving_ue_valid_mask,
            self._serving_ue_valid_mask,
        ):
            raise ValueError(
                "Serving-UE valid layout changed. "
                "Reassociation support is not yet "
                "implemented."
            )

        if (
            observation
            .serving_ue_valid_mask
            .dtype
            != torch.bool
        ):
            raise ValueError(
                "serving_ue_valid_mask must use "
                "torch.bool."
            )

        if torch.any(
            observation
            .td_instantaneous_rate_bps
            < 0.0
        ):
            raise ValueError(
                "Instantaneous rate cannot be "
                "negative."
            )


    @staticmethod
    def _gather_candidate_values(
        *,
        values: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather a serving-UE tensor into candidate
        order.

        Input:
            [serving_ue, ...]

        Output:
            [candidate, ...]
        """

        gathered = values[
            candidate_indices
        ]

        trailing_dimensions = (
            gathered.ndim - 1
        )

        mask_shape = (
            candidate_valid_mask.shape
            + (
                1,
            ) * trailing_dimensions
        )

        expanded_mask = (
            candidate_valid_mask.reshape(
                mask_shape
            )
        )

        return torch.where(
            expanded_mask,
            gathered,
            torch.zeros_like(
                gathered
            ),
        )

    def prepare_tti(
        self,
        *,
        tti_index: int,
        observation: OneLDSCellTTIObservation,
        tds_eligible_mask: (
            torch.Tensor | None
        ) = None,
    ) -> PreparedOneLDSCellTTI:
        """
        Build the PF-TDS candidate set and candidate
        1LDS inputs for one TTI.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if self._active_tti is not None:
            raise RuntimeError(
                "The previous TTI must be completed "
                "before preparing another TTI."
            )

        if (
            self._last_completed_tti_index
            is not None
        ):
            expected_index = (
                self._last_completed_tti_index
                + 1
            )

            if tti_index != expected_index:
                raise ValueError(
                    "TTIs must be prepared in "
                    "contiguous temporal order."
                )

        self._validate_observation(
            observation
        )

        if tds_eligible_mask is None:
            #
            # Backward-compatible behavior:
            # every real associated UE may enter
            # PF-TDS.
            #
            effective_tds_mask = (
                observation
                .serving_ue_valid_mask
            )

        else:
            if tuple(
                tds_eligible_mask.shape
            ) != tuple(
                observation
                .serving_ue_valid_mask
                .shape
            ):
                raise ValueError(
                    "tds_eligible_mask must have "
                    "shape [serving_ue]."
                )

            if (
                tds_eligible_mask.dtype
                != torch.bool
            ):
                raise ValueError(
                    "tds_eligible_mask must use "
                    "torch.bool."
                )

            if (
                tds_eligible_mask.device
                != observation
                .serving_ue_valid_mask
                .device
            ):
                raise ValueError(
                    "tds_eligible_mask is on the "
                    "wrong device."
                )

            #
            # Scheduling eligibility can only
            # REMOVE associated UEs.
            #
            # It must never turn padding or an
            # otherwise invalid serving slot into a
            # real UE.
            #
            if torch.any(
                tds_eligible_mask
                & ~(
                    observation
                    .serving_ue_valid_mask
                )
            ):
                raise ValueError(
                    "TDS eligibility cannot enable "
                    "an invalid serving-UE slot."
                )

            effective_tds_mask = (
                tds_eligible_mask
            )

        #
        # run_pf_tds currently uses:
        #
        # [batch, cell, serving_ue]
        #
        # This manager represents one cell, so add
        # singleton batch and cell dimensions.
        #
        tds_result = run_pf_tds(
            instantaneous_rate=(
                observation
                .td_instantaneous_rate_bps[
                    None,
                    None,
                    :,
                ]
            ),
            past_average_throughput=(
                self
                ._average_throughput_bps[
                    None,
                    None,
                    :,
                ]
            ),
            config=self.tds_config,
            valid_ue_mask=(
                effective_tds_mask[
                    None,
                    None,
                    :,
                ]
            ),
        )

        candidate_indices = (
            tds_result
            .candidate_indices[
                0,
                0,
                :,
            ]
        )

        candidate_valid_mask = (
            tds_result
            .candidate_valid_mask[
                0,
                0,
                :,
            ]
        )

        candidate_global_ue_indices = (
            observation
            .serving_global_ue_indices[
                candidate_indices
            ]
        )

        candidate_global_ue_indices = torch.where(
            candidate_valid_mask,
            candidate_global_ue_indices,
            torch.full_like(
                candidate_global_ue_indices,
                fill_value=-1,
            ),
        )

        candidate_history = (
            self._gather_candidate_values(
                values=(
                    self
                    ._average_throughput_bps
                ),
                candidate_indices=(
                    candidate_indices
                ),
                candidate_valid_mask=(
                    candidate_valid_mask
                ),
            )
        )

        candidate_rank = (
            self._gather_candidate_values(
                values=observation.rank,
                candidate_indices=(
                    candidate_indices
                ),
                candidate_valid_mask=(
                    candidate_valid_mask
                ),
            )
        )

        candidate_buffer = (
            self._gather_candidate_values(
                values=observation.dl_buffer,
                candidate_indices=(
                    candidate_indices
                ),
                candidate_valid_mask=(
                    candidate_valid_mask
                ),
            )
        )

        candidate_wideband_cqi = (
            self._gather_candidate_values(
                values=(
                    observation.wideband_cqi
                ),
                candidate_indices=(
                    candidate_indices
                ),
                candidate_valid_mask=(
                    candidate_valid_mask
                ),
            )
        )

        candidate_subband_cqi = (
            self._gather_candidate_values(
                values=(
                    observation.subband_cqi
                ),
                candidate_indices=(
                    candidate_indices
                ),
                candidate_valid_mask=(
                    candidate_valid_mask
                ),
            )
        )

        candidate_precoder_directions = (
            self._gather_candidate_values(
                values=(
                    observation
                    .precoder_directions
                ),
                candidate_indices=(
                    candidate_indices
                ),
                candidate_valid_mask=(
                    candidate_valid_mask
                ),
            )
        )

        decision_inputs = OneLDSDecisionInputs(
            past_average_throughput=(
                candidate_history
            ),
            rank=candidate_rank,
            dl_buffer=candidate_buffer,
            wideband_cqi=(
                candidate_wideband_cqi
            ),
            subband_cqi=(
                candidate_subband_cqi
            ),
            candidate_precoder_directions=(
                candidate_precoder_directions
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
        )

        prepared = PreparedOneLDSCellTTI(
            tti_index=tti_index,
            tds_result=tds_result,
            candidate_serving_indices=(
                candidate_indices
                .detach()
                .clone()
            ),
            candidate_global_ue_indices=(
                candidate_global_ue_indices
                .detach()
                .clone()
            ),
            candidate_valid_mask=(
                candidate_valid_mask
                .detach()
                .clone()
            ),
            decision_inputs=decision_inputs,
        )

        self._active_tti = prepared

        return prepared


    def complete_tti(
        self,
        *,
        candidate_delivered_rate_bps: torch.Tensor,
    ) -> CellThroughputHistoryUpdate:
        """
        Advance PF history after the physical outcome
        of the active TTI is known.

        candidate_delivered_rate_bps:
            [candidate]

        This should be the final realized throughput
        across all RBGs for each TDS candidate.
        """

        if self._active_tti is None:
            raise RuntimeError(
                "No active TTI exists."
            )

        prepared = self._active_tti

        expected_shape = (
            prepared
            .candidate_serving_indices
            .shape
        )

        if tuple(
            candidate_delivered_rate_bps.shape
        ) != tuple(
            expected_shape
        ):
            raise ValueError(
                "candidate_delivered_rate_bps must "
                "have shape [candidate]."
            )

        if (
            candidate_delivered_rate_bps.device
            != self
            ._average_throughput_bps
            .device
        ):
            raise ValueError(
                "Delivered rate is on the wrong "
                "device."
            )

        history_update = (
            update_cell_throughput_history(
                previous_average_throughput_bps=(
                    self
                    ._average_throughput_bps
                ),
                serving_ue_valid_mask=(
                    self
                    ._serving_ue_valid_mask
                ),
                candidate_indices=(
                    prepared
                    .candidate_serving_indices
                ),
                candidate_valid_mask=(
                    prepared
                    .candidate_valid_mask
                ),
                candidate_delivered_rate_bps=(
                    candidate_delivered_rate_bps
                ),
                forgetting_factor=(
                    self
                    .throughput_forgetting_factor
                ),
            )
        )

        self._average_throughput_bps = (
            history_update
            .updated_average_throughput_bps
            .detach()
            .clone()
        )

        self._last_completed_tti_index = (
            prepared.tti_index
        )

        self._active_tti = None

        return history_update


