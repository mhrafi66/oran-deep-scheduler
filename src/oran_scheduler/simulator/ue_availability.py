from __future__ import annotations

from dataclasses import dataclass, replace
import json

import torch

from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)


@dataclass(frozen=True)
class UEAvailabilityPhase:
    """
    Scheduler-side UE availability phase.

    IMPORTANT:
        This is NOT yet handover.

        Global UE identity, serving-cell ownership,
        PF history, and traffic queue remain attached
        to the same cell.

        An unavailable UE is simply excluded from
        scheduling eligibility.

    Traffic may continue to arrive while unavailable,
    which intentionally models a temporarily
    unreachable/disconnected UE whose queue grows.
    """

    start_tti: int

    name: str

    unavailable_fraction: float = 0.0

    unavailable_global_ue_indices: tuple[
        int,
        ...
    ] = ()


    def __post_init__(
        self,
    ) -> None:

        if self.start_tti < 0:
            raise ValueError(
                "start_tti must be non-negative."
            )

        if not self.name:
            raise ValueError(
                "name cannot be empty."
            )

        if not (
            0.0
            <= self.unavailable_fraction
            <= 1.0
        ):
            raise ValueError(
                "unavailable_fraction must lie "
                "in [0, 1]."
            )

        if any(
            value < 0
            for value
            in self
            .unavailable_global_ue_indices
        ):
            raise ValueError(
                "Global UE IDs cannot be negative."
            )


class UEAvailabilityScenario:
    def __init__(
        self,
        *,
        phases: tuple[
            UEAvailabilityPhase,
            ...
        ],
        seed: int = 99173,
    ) -> None:

        if not phases:
            raise ValueError(
                "At least one phase is required."
            )

        ordered = tuple(
            sorted(
                phases,
                key=lambda phase: (
                    phase.start_tti
                ),
            )
        )

        if ordered[0].start_tti != 0:
            raise ValueError(
                "First phase must begin at TTI 0."
            )

        starts = [
            phase.start_tti
            for phase in ordered
        ]

        if (
            len(set(starts))
            != len(starts)
        ):
            raise ValueError(
                "Availability phase starts must "
                "be unique."
            )

        self.phases = ordered

        self.seed = int(
            seed
        )


    def phase_for_tti(
        self,
        tti_index: int,
    ) -> UEAvailabilityPhase:

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        selected = self.phases[0]

        for phase in self.phases:

            if (
                phase.start_tti
                > tti_index
            ):
                break

            selected = phase

        return selected


    def unavailable_mask(
        self,
        *,
        global_ue_indices: torch.Tensor,
        existing_valid_mask: torch.Tensor,
        tti_index: int,
    ) -> torch.Tensor:

        phase = self.phase_for_tti(
            tti_index
        )

        if (
            global_ue_indices.shape
            != existing_valid_mask.shape
        ):
            raise ValueError(
                "UE ID and validity shapes differ."
            )

        output = torch.zeros_like(
            existing_valid_mask,
            dtype=torch.bool,
        )

        explicit = set(
            phase
            .unavailable_global_ue_indices
        )

        ids_cpu = (
            global_ue_indices
            .detach()
            .cpu()
        )

        valid_cpu = (
            existing_valid_mask
            .detach()
            .cpu()
            .bool()
        )

        selected = []

        for index in range(
            int(ids_cpu.numel())
        ):

            if not bool(
                valid_cpu[index].item()
            ):
                selected.append(
                    False
                )
                continue

            ue_id = int(
                ids_cpu[index].item()
            )

            unavailable = (
                ue_id in explicit
            )

            if (
                not unavailable
                and phase
                .unavailable_fraction
                > 0.0
            ):
                #
                # Deterministic hash by global UE
                # identity and experiment seed.
                #
                hashed = (
                    (
                        ue_id
                        * 1_103_515_245
                        + self.seed
                    )
                    % 1_000_003
                )

                fraction = (
                    hashed
                    / 1_000_003.0
                )

                unavailable = (
                    fraction
                    < phase
                    .unavailable_fraction
                )

            selected.append(
                unavailable
            )

        return torch.tensor(
            selected,
            dtype=torch.bool,
            device=(
                global_ue_indices
                .device
            ),
        )


def parse_ue_availability_scenario_json(
    raw: str,
) -> UEAvailabilityScenario:

    data = json.loads(
        raw
    )

    if isinstance(
        data,
        dict,
    ):
        seed = int(
            data.get(
                "seed",
                99173,
            )
        )

        phase_data = data[
            "phases"
        ]

    elif isinstance(
        data,
        list,
    ):
        seed = 99173

        phase_data = data

    else:
        raise ValueError(
            "UE availability JSON must be "
            "a list or object."
        )

    phases = []

    for item in phase_data:

        phases.append(
            UEAvailabilityPhase(
                start_tti=int(
                    item[
                        "start_tti"
                    ]
                ),

                name=str(
                    item[
                        "name"
                    ]
                ),

                unavailable_fraction=float(
                    item.get(
                        "unavailable_fraction",
                        0.0,
                    )
                ),

                unavailable_global_ue_indices=tuple(
                    int(value)
                    for value
                    in item.get(
                        "unavailable_global_ue_indices",
                        [],
                    )
                ),
            )
        )

    return UEAvailabilityScenario(
        phases=tuple(
            phases
        ),
        seed=seed,
    )


class UEAvailabilityInputProvider:
    """
    Temporarily remove UEs from PF-TDS scheduling
    eligibility WITHOUT changing association.

    Important separation:

        observation.serving_ue_valid_mask
            =
        persistent association / padded UE layout

        tds_eligibility_override_mask
            =
        temporary per-TTI schedulability

    Therefore an unavailable UE remains attached to
    the same cell, retains its traffic queue and PF
    history, but cannot enter PF-TDS while the
    availability gate is false.

    Traffic arrivals continue while unavailable, so
    finite-buffer backlog may accumulate.

    This is NOT handover or reassociation.
    """

    def __init__(
        self,
        *,
        base_provider,
        scenario: UEAvailabilityScenario,
    ) -> None:

        self.base_provider = (
            base_provider
        )

        self.scenario = scenario


    def __call__(
        self,
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:

        base = self.base_provider(
            tti_index,
            stream_index,
        )

        observation = (
            base.observation
        )

        unavailable = (
            self
            .scenario
            .unavailable_mask(
                global_ue_indices=(
                    observation
                    .serving_global_ue_indices
                ),

                existing_valid_mask=(
                    observation
                    .serving_ue_valid_mask
                ),

                tti_index=(
                    tti_index
                ),
            )
        )

        availability_gate = (
            observation
            .serving_ue_valid_mask
            & (~unavailable)
        )

        #
        # CRITICAL SEMANTIC SEPARATION
        # ----------------------------
        #
        # Do NOT modify:
        #
        #     observation.serving_ue_valid_mask
        #
        # That mask describes the persistent serving
        # layout/association and is intentionally
        # required to remain constant by the
        # OneLDSCellTTIStateManager.
        #
        # Temporary availability is instead carried
        # as a separate TDS eligibility gate.
        #
        # Also preserve packet_arrivals unchanged:
        # traffic may continue entering the queue
        # while a UE is temporarily unavailable.
        #
        return replace(
            base,
            tds_eligibility_override_mask=(
                availability_gate
            ),
        )
