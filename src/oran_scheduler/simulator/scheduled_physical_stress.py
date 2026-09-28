from __future__ import annotations

from dataclasses import replace
import json

from oran_scheduler.rl.physical_input_stress import (
    PhysicalExecutionStressConfig,
    stress_physical_inputs,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.simulator.network_events import (
    PhysicalNetworkPhase,
    PhysicalNetworkScenario,
)


def parse_physical_network_scenario_json(
    raw: str,
) -> PhysicalNetworkScenario:
    """
    JSON format:

    [
      {
        "start_tti": 0,
        "name": "normal"
      },
      {
        "start_tti": 20,
        "name": "interference",
        "non_serving_interference_power_scale": 4.0
      },
      {
        "start_tti": 40,
        "name": "bs_failure",
        "failed_bs_index": 0
      },
      {
        "start_tti": 60,
        "name": "recovery"
      }
    ]
    """

    data = json.loads(
        raw
    )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Physical network scenario JSON "
            "must contain a list."
        )

    phases = []

    for item in data:

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Every network phase must be "
                "a JSON object."
            )

        phases.append(
            PhysicalNetworkPhase(
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

                non_serving_interference_power_scale=float(
                    item.get(
                        "non_serving_interference_power_scale",
                        1.0,
                    )
                ),

                serving_signal_power_scale=float(
                    item.get(
                        "serving_signal_power_scale",
                        1.0,
                    )
                ),

                failed_bs_index=(
                    None
                    if item.get(
                        "failed_bs_index",
                        None,
                    ) is None
                    else int(
                        item[
                            "failed_bs_index"
                        ]
                    )
                ),
            )
        )

    return PhysicalNetworkScenario(
        phases=tuple(
            phases
        )
    )


class ScheduledPhysicalExecutionStressInputProvider:
    """
    Apply a piecewise-constant execution-time PHY
    stress to the CURRENT physical channel.

    Scheduler observation remains untouched.

    Therefore this models:

        scheduler observes state
            ->
        scheduler chooses action
            ->
        execution environment changes.

    This can represent:

        normal -> interference shock -> recovery

        normal -> serving degradation -> recovery

        normal -> BS outage -> recovery.
    """

    def __init__(
        self,
        *,
        base_provider,
        scenario: PhysicalNetworkScenario,
    ) -> None:

        self.base_provider = (
            base_provider
        )

        self.scenario = scenario


    def phase_name(
        self,
        tti_index: int,
    ) -> str:

        return (
            self.scenario
            .phase_for_tti(
                tti_index
            )
            .name
        )


    def __call__(
        self,
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:

        base = self.base_provider(
            tti_index,
            stream_index,
        )

        phase = (
            self
            .scenario
            .phase_for_tti(
                tti_index
            )
        )

        base_builder = (
            base
            .physical_inputs_builder
        )

        def stressed_builder(
            prepared,
        ):
            physical = base_builder(
                prepared
            )

            return stress_physical_inputs(
                inputs=physical,

                config=(
                    PhysicalExecutionStressConfig(
                        non_serving_interference_power_scale=(
                            phase
                            .non_serving_interference_power_scale
                        ),

                        serving_signal_power_scale=(
                            phase
                            .serving_signal_power_scale
                        ),

                        failed_bs_index=(
                            phase
                            .failed_bs_index
                        ),
                    )
                ),
            )

        return replace(
            base,
            physical_inputs_builder=(
                stressed_builder
            ),
        )
