from __future__ import annotations

from dataclasses import dataclass

import torch

from oran_scheduler.rl.ppo_traffic_cell_step import (
    PreparedTrafficAwarePPOCellTTI,
)
from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.schedulers.one_lds_loop import (
    OneLDSScheduleResult,
)
from oran_scheduler.simulator.control_loop import (
    ControlLoopTimingConfig,
    ControlLoopTimingModel,
)
from oran_scheduler.simulator.identity_action_delay import (
    GlobalUEAllocation,
    IdentitySafeActionDelay,
    NO_GLOBAL_UE,
    remap_global_allocation_to_candidates,
)


def schedule_to_global_ue_allocation(
    *,
    schedule: OneLDSScheduleResult,
    preparation: PreparedTrafficAwarePPOCellTTI,
) -> GlobalUEAllocation:
    """
    Convert candidate-slot allocation to persistent
    global UE identities.
    """

    candidate_by_slot = (
        schedule
        .allocation
        .candidate_by_user_slot
    )

    candidate_ids = (
        preparation
        .prepared
        .candidate_global_ue_indices
    )

    candidate_valid = (
        preparation
        .prepared
        .candidate_valid_mask
    )

    output = torch.full_like(
        candidate_by_slot,
        fill_value=NO_GLOBAL_UE,
    )

    scheduled = (
        candidate_by_slot
        >= 0
    )

    if torch.any(
        scheduled
    ):
        candidate_indices = (
            candidate_by_slot[
                scheduled
            ]
        )

        if torch.any(
            candidate_indices
            >= candidate_ids.numel()
        ):
            raise ValueError(
                "Schedule contains candidate index "
                "outside current candidate set."
            )

        if not torch.all(
            candidate_valid[
                candidate_indices
            ]
        ):
            raise ValueError(
                "Schedule selected padded candidate."
            )

        output[
            scheduled
        ] = candidate_ids[
            candidate_indices
        ]

    return GlobalUEAllocation(
        global_ue_by_user_slot=(
            output
        )
    )


def global_ue_allocation_to_schedule(
    *,
    allocation: GlobalUEAllocation,
    policy_schedule: OneLDSScheduleResult,
    preparation: PreparedTrafficAwarePPOCellTTI,
) -> OneLDSScheduleResult:
    """
    Map persistent UE identities into the CURRENT
    candidate set.

    IMPORTANT LIMITATION:
        Current paper PHY stores only current PF-TDS
        candidates.

        Therefore a delayed UE that is no longer in
        the CURRENT candidate shortlist becomes
        no-allocation.

    This is scientifically labeled:

        candidate-gated identity-safe delay

    It is NOT yet unrestricted stale-action execution.
    """

    candidate_ids = (
        preparation
        .prepared
        .candidate_global_ue_indices
    )

    valid = (
        preparation
        .prepared
        .candidate_valid_mask
    )

    num_candidates = int(
        candidate_ids.numel()
    )

    actions = (
        remap_global_allocation_to_candidates(
            allocation=allocation,

            candidate_global_ue_indices=(
                candidate_ids
            ),

            candidate_valid_mask=(
                valid
            ),

            no_allocation_action=(
                num_candidates
            ),
        )
    )

    candidate_by_slot = (
        torch.where(
            actions
            == num_candidates,

            torch.full_like(
                actions,
                -1,
            ),

            actions,
        )
    )

    return OneLDSScheduleResult(
        allocation=CellAllocation(
            candidate_by_user_slot=(
                candidate_by_slot
            )
        ),

        decisions=(
            policy_schedule.decisions
        ),

        actions=actions,
    )


@dataclass(frozen=True)
class ExecutionControlStats:
    num_ttis: int
    num_deadline_misses: int
    num_requested_assignments: int
    num_dropped_assignments: int


class CandidateGatedDelayedPPOExecutionController:
    """
    Fixed D-TTI action delay using GLOBAL UE IDs.

    Startup:
        no allocation until FIFO matures.

    A delayed UE missing from the current top-K
    candidate set is dropped rather than silently
    reinterpreted as another UE.
    """

    def __init__(
        self,
        *,
        delay_ttis: int,
        num_user_slots: int,
        num_rbgs: int,
        device: str | torch.device,
    ) -> None:

        self.delay = (
            IdentitySafeActionDelay(
                delay_ttis=delay_ttis,

                num_user_slots=(
                    num_user_slots
                ),

                num_rbgs=num_rbgs,

                device=device,
            )
        )

        self._num_ttis = 0

        self._num_requested = 0

        self._num_dropped = 0


    @property
    def stats(
        self,
    ) -> ExecutionControlStats:

        return ExecutionControlStats(
            num_ttis=self._num_ttis,

            num_deadline_misses=0,

            num_requested_assignments=(
                self._num_requested
            ),

            num_dropped_assignments=(
                self._num_dropped
            ),
        )


    def __call__(
        self,
        tti_index: int,
        policy_schedule: OneLDSScheduleResult,
        preparation: PreparedTrafficAwarePPOCellTTI,
    ) -> OneLDSScheduleResult:

        del tti_index

        current_global = (
            schedule_to_global_ue_allocation(
                schedule=policy_schedule,
                preparation=preparation,
            )
        )

        delayed_global = (
            self.delay.push(
                current_global
            )
        )

        requested = int(
            torch.count_nonzero(
                delayed_global
                .global_ue_by_user_slot
                >= 0
            ).item()
        )

        executed = (
            global_ue_allocation_to_schedule(
                allocation=delayed_global,

                policy_schedule=(
                    policy_schedule
                ),

                preparation=preparation,
            )
        )

        executed_count = int(
            torch.count_nonzero(
                executed
                .allocation
                .candidate_by_user_slot
                >= 0
            ).item()
        )

        self._num_ttis += 1

        self._num_requested += (
            requested
        )

        self._num_dropped += max(
            0,
            requested - executed_count,
        )

        return executed


class CandidateGatedControlLoopPPOExecutionController:
    """
    Control-deadline failure model.

    Normal:
        execute current PPO schedule.

    Deadline miss:
        empty
        OR
        repeat previous successful schedule using
        persistent UE identity and current-candidate
        remapping.
    """

    def __init__(
        self,
        *,
        config: ControlLoopTimingConfig,
        num_user_slots: int,
        num_rbgs: int,
        device: str | torch.device,
    ) -> None:

        self.model = (
            ControlLoopTimingModel(
                config
            )
        )

        self.num_user_slots = (
            num_user_slots
        )

        self.num_rbgs = num_rbgs

        self.device = torch.device(
            device
        )

        self.previous_successful_global: (
            GlobalUEAllocation | None
        ) = None

        self._num_ttis = 0

        self._num_misses = 0

        self._num_requested = 0

        self._num_dropped = 0


    @property
    def stats(
        self,
    ) -> ExecutionControlStats:

        return ExecutionControlStats(
            num_ttis=self._num_ttis,

            num_deadline_misses=(
                self._num_misses
            ),

            num_requested_assignments=(
                self._num_requested
            ),

            num_dropped_assignments=(
                self._num_dropped
            ),
        )


    def _empty(
        self,
        *,
        policy_schedule: OneLDSScheduleResult,
        preparation: PreparedTrafficAwarePPOCellTTI,
    ) -> OneLDSScheduleResult:

        empty = GlobalUEAllocation.empty(
            num_user_slots=(
                self.num_user_slots
            ),

            num_rbgs=(
                self.num_rbgs
            ),

            device=self.device,
        )

        return global_ue_allocation_to_schedule(
            allocation=empty,

            policy_schedule=(
                policy_schedule
            ),

            preparation=preparation,
        )


    def __call__(
        self,
        tti_index: int,
        policy_schedule: OneLDSScheduleResult,
        preparation: PreparedTrafficAwarePPOCellTTI,
    ) -> OneLDSScheduleResult:

        decision = (
            self.model.decision(
                tti_index
            )
        )

        current_global = (
            schedule_to_global_ue_allocation(
                schedule=policy_schedule,

                preparation=preparation,
            )
        )

        self._num_ttis += 1

        if not decision.deadline_missed:

            self.previous_successful_global = (
                current_global
            )

            return policy_schedule

        self._num_misses += 1

        if (
            decision.miss_policy
            == "empty"
        ):
            return self._empty(
                policy_schedule=(
                    policy_schedule
                ),

                preparation=preparation,
            )

        if (
            self.previous_successful_global
            is None
        ):
            return self._empty(
                policy_schedule=(
                    policy_schedule
                ),

                preparation=preparation,
            )

        requested = int(
            torch.count_nonzero(
                self
                .previous_successful_global
                .global_ue_by_user_slot
                >= 0
            ).item()
        )

        executed = (
            global_ue_allocation_to_schedule(
                allocation=(
                    self
                    .previous_successful_global
                ),

                policy_schedule=(
                    policy_schedule
                ),

                preparation=preparation,
            )
        )

        executed_count = int(
            torch.count_nonzero(
                executed
                .allocation
                .candidate_by_user_slot
                >= 0
            ).item()
        )

        self._num_requested += (
            requested
        )

        self._num_dropped += max(
            0,
            requested - executed_count,
        )

        return executed
