from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from oran_scheduler.rl.ppo_traffic_cell_step import (
    PreparedTrafficAwarePPOCellTTI,
    prepare_traffic_aware_ppo_cell_tti,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.schedulers.classical_runtime_backend import (
    ClassicalSchedulerMode,
)
from oran_scheduler.schedulers.classical_traffic_eval import (
    TrafficAwareClassicalCellTTIResult,
    run_prepared_classical_traffic_tti,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIStateManager,
)
from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)
from oran_scheduler.simulator.traffic import (
    TrafficBufferManager,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)


ClassicalInputProvider = Callable[
    [
        int,
        int,
    ],
    PPOTrainingTTIInputs,
]


ClassicalPreparationObserver = Callable[
    [
        int,
        int,
        PreparedTrafficAwarePPOCellTTI,
    ],
    None,
]


ClassicalResultObserver = Callable[
    [
        int,
        int,
        TrafficAwareClassicalCellTTIResult,
    ],
    None,
]


@dataclass(frozen=True)
class ClassicalMultiCellEvalResult:
    """
    Deliberately mirrors the PPO runner diagnostics
    needed by run_robustness_eval.py.
    """

    start_tti_index: int

    end_tti_index_exclusive: int

    num_ttis: int

    num_cells: int

    num_warmup_ttis: int = 0

    num_collection_ttis: int = 0

    first_collected_tti_index: None = None

    num_ppo_updates: int = 0

    num_expert_guidance_updates: int = 0

    num_expert_demonstrations_added: int = 0

    final_transition_buffer_size: int = 0

    unresolved_boundary_by_cell: tuple[
        bool,
        ...
    ] = tuple()


def run_multicell_classical_evaluation(
    *,
    start_tti_index: int,
    num_ttis: int,
    mode: ClassicalSchedulerMode,
    input_provider: ClassicalInputProvider,
    traffic_managers: tuple[
        TrafficBufferManager,
        ...,
    ],
    state_managers: tuple[
        OneLDSCellTTIStateManager,
        ...,
    ],
    tds_eligibility_config: (
        TDSBufferEligibilityConfig
    ),
    state_config: OneLDSStateConfig,
    num_user_slots: int,
    preparation_observer: (
        ClassicalPreparationObserver | None
    ) = None,
    cell_result_observer: (
        ClassicalResultObserver | None
    ) = None,
    device: str | torch.device = "cuda:0",
) -> ClassicalMultiCellEvalResult:
    """
    Synchronized classical evaluation.

    For each TTI:

        Phase A:
            prepare ALL cells first

        Phase B:
            schedule/evaluate ALL cells

    This keeps the same multicell synchronization
    principle used by the PPO runner.
    """

    if start_tti_index < 0:
        raise ValueError(
            "start_tti_index must be non-negative."
        )

    if num_ttis <= 0:
        raise ValueError(
            "num_ttis must be positive."
        )

    if num_user_slots <= 0:
        raise ValueError(
            "num_user_slots must be positive."
        )

    num_cells = len(
        state_managers
    )

    if num_cells <= 0:
        raise ValueError(
            "At least one cell is required."
        )

    if len(
        traffic_managers
    ) != num_cells:
        raise ValueError(
            "traffic/state manager counts differ."
        )

    for tti_index in range(
        start_tti_index,
        start_tti_index
        + num_ttis,
    ):

        preparations: list[
            tuple[
                PreparedTrafficAwarePPOCellTTI,
                PPOTrainingTTIInputs,
            ]
        ] = []

        #
        # Phase A:
        # prepare all cells.
        #
        for cell_index in range(
            num_cells
        ):

            inputs = input_provider(
                tti_index,
                cell_index,
            )

            preparation = (
                prepare_traffic_aware_ppo_cell_tti(
                    tti_index=tti_index,

                    observation=(
                        inputs
                        .observation
                    ),

                    traffic_manager=(
                        traffic_managers[
                            cell_index
                        ]
                    ),

                    tds_eligibility_config=(
                        tds_eligibility_config
                    ),

                    state_manager=(
                        state_managers[
                            cell_index
                        ]
                    ),

                    state_config=(
                        state_config
                    ),

                    physical_inputs_builder=(
                        inputs
                        .physical_inputs_builder
                    ),

                    num_user_slots=(
                        num_user_slots
                    ),

                    packet_arrivals=(
                        inputs
                        .packet_arrivals
                    ),

                    tds_eligibility_override_mask=(
                        inputs
                        .tds_eligibility_override_mask
                    ),

                    device=device,
                )
            )

            if preparation_observer is not None:
                preparation_observer(
                    tti_index,
                    cell_index,
                    preparation,
                )

            preparations.append(
                (
                    preparation,
                    inputs,
                )
            )

        #
        # Phase B:
        # schedule + PHY + traffic.
        #
        for cell_index, (
            preparation,
            _,
        ) in enumerate(
            preparations
        ):

            result = (
                run_prepared_classical_traffic_tti(
                    mode=mode,

                    preparation=(
                        preparation
                    ),

                    traffic_manager=(
                        traffic_managers[
                            cell_index
                        ]
                    ),

                    state_manager=(
                        state_managers[
                            cell_index
                        ]
                    ),

                    num_user_slots=(
                        num_user_slots
                    ),
                )
            )

            if cell_result_observer is not None:
                cell_result_observer(
                    tti_index,
                    cell_index,
                    result,
                )

    return ClassicalMultiCellEvalResult(
        start_tti_index=(
            start_tti_index
        ),

        end_tti_index_exclusive=(
            start_tti_index
            + num_ttis
        ),

        num_ttis=num_ttis,

        num_cells=num_cells,

        unresolved_boundary_by_cell=tuple(
            False
            for _
            in range(
                num_cells
            )
        ),
    )
