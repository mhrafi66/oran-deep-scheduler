from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
import math

from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIStateManager,
)
from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
    TrafficBufferManager,
)


@dataclass(frozen=True)
class RuntimeScenarioPhase:
    """
    Piecewise-constant runtime network condition.

    start_tti:
        First TTI using this phase.

    ftp_rate_scale:
        Multiplier on the ORIGINAL FTP packet-arrival rate.

    ftp_packet_size_scale:
        Multiplier on the ORIGINAL FTP packet size.

    reset_pf_history:
        Reset PF history upon ENTERING this phase.

    pf_history_bps:
        PF-history reset value.

    reset_ftp_buffers:
        Clear/set finite FTP queues upon entering phase.

    ftp_buffer_bits:
        FTP backlog reset value.
    """

    start_tti: int

    name: str = ""

    ftp_rate_scale: float = 1.0

    ftp_packet_size_scale: float = 1.0

    reset_pf_history: bool = False

    pf_history_bps: float = 1.0e6

    reset_ftp_buffers: bool = False

    ftp_buffer_bits: float = 0.0


    def __post_init__(
        self,
    ) -> None:

        if self.start_tti < 0:
            raise ValueError(
                "start_tti must be non-negative."
            )

        for name, value in (
            (
                "ftp_rate_scale",
                self.ftp_rate_scale,
            ),
            (
                "ftp_packet_size_scale",
                self.ftp_packet_size_scale,
            ),
        ):
            if (
                not math.isfinite(
                    value
                )
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and "
                    "non-negative."
                )

        if (
            not math.isfinite(
                self.pf_history_bps
            )
            or self.pf_history_bps < 0.0
        ):
            raise ValueError(
                "pf_history_bps must be finite and "
                "non-negative."
            )

        if (
            not math.isfinite(
                self.ftp_buffer_bits
            )
            or self.ftp_buffer_bits < 0.0
        ):
            raise ValueError(
                "ftp_buffer_bits must be finite and "
                "non-negative."
            )


def parse_runtime_scenario_json(
    text: str,
) -> tuple[
    RuntimeScenarioPhase,
    ...,
]:
    """
    Parse an environment-variable JSON scenario.

    Example:

    [
      {
        "start_tti": 0,
        "name": "normal",
        "ftp_rate_scale": 1.0
      },
      {
        "start_tti": 30,
        "name": "overload",
        "ftp_rate_scale": 4.0
      },
      {
        "start_tti": 60,
        "name": "recovery",
        "ftp_rate_scale": 1.0,
        "reset_pf_history": true
      }
    ]
    """

    if not text.strip():
        return tuple()

    raw = json.loads(
        text
    )

    if not isinstance(
        raw,
        list,
    ):
        raise ValueError(
            "Runtime scenario JSON must be a list."
        )

    phases = tuple(
        RuntimeScenarioPhase(
            **item
        )
        for item
        in raw
    )

    starts = [
        phase.start_tti
        for phase
        in phases
    ]

    if starts != sorted(
        starts
    ):
        raise ValueError(
            "Runtime scenario phases must be sorted "
            "by start_tti."
        )

    if len(
        set(
            starts
        )
    ) != len(
        starts
    ):
        raise ValueError(
            "Runtime scenario start_tti values "
            "must be unique."
        )

    if (
        phases
        and phases[0].start_tti != 0
    ):
        raise ValueError(
            "A non-empty runtime scenario must start "
            "at TTI 0."
        )

    return phases


class RuntimeScenarioController:
    """
    Apply piecewise traffic/state conditions between TTIs.

    This controller never modifies radio/PHY truth.

    It currently supports:

        * FTP offered-load changes
        * FTP packet-size changes
        * PF-history reset
        * FTP queue reset

    These are intentionally implemented between TTIs.
    """

    def __init__(
        self,
        *,
        phases: tuple[
            RuntimeScenarioPhase,
            ...,
        ],
        traffic_managers: tuple[
            TrafficBufferManager,
            ...,
        ],
        state_managers: tuple[
            OneLDSCellTTIStateManager,
            ...,
        ],
    ) -> None:

        if len(
            traffic_managers
        ) != len(
            state_managers
        ):
            raise ValueError(
                "Traffic/state manager counts differ."
            )

        self.phases = phases

        self.traffic_managers = (
            traffic_managers
        )

        self.state_managers = (
            state_managers
        )

        self._base_ftp_configs = tuple(
            manager.ftp3_config
            for manager
            in traffic_managers
        )

        self._current_phase_index: (
            int | None
        ) = None


    @property
    def current_phase_name(
        self,
    ) -> str:

        if (
            self._current_phase_index
            is None
        ):
            return "static"

        phase = self.phases[
            self._current_phase_index
        ]

        if phase.name:
            return phase.name

        return (
            f"phase_{self._current_phase_index}"
        )


    def _phase_index_for_tti(
        self,
        tti_index: int,
    ) -> int | None:

        selected: int | None = None

        for index, phase in enumerate(
            self.phases
        ):
            if phase.start_tti <= tti_index:
                selected = index

            else:
                break

        return selected


    def apply_tti(
        self,
        tti_index: int,
    ) -> None:
        """
        Apply the phase governing this TTI.

        Must be called before begin_tti()/prepare_tti().
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        phase_index = (
            self._phase_index_for_tti(
                tti_index
            )
        )

        if phase_index is None:
            return

        phase = self.phases[
            phase_index
        ]

        #
        # Traffic scaling is always derived from the
        # ORIGINAL configuration, avoiding compounded
        # scaling across phases.
        #
        for (
            manager,
            base_config,
        ) in zip(
            self.traffic_managers,
            self._base_ftp_configs,
            strict=True,
        ):

            scaled_packet_size = max(
                1,
                int(
                    round(
                        base_config.packet_size_bytes
                        * phase
                        .ftp_packet_size_scale
                    )
                ),
            )

            manager.ftp3_config = replace(
                base_config,

                packet_size_bytes=(
                    scaled_packet_size
                ),

                packet_arrival_rate_per_s=(
                    base_config
                    .packet_arrival_rate_per_s
                    * phase.ftp_rate_scale
                ),
            )

        entering_new_phase = (
            phase_index
            != self._current_phase_index
        )

        if entering_new_phase:

            if phase.reset_pf_history:
                for manager in (
                    self.state_managers
                ):
                    manager.reset_average_throughput_bps(
                        phase.pf_history_bps
                    )

            if phase.reset_ftp_buffers:
                for manager in (
                    self.traffic_managers
                ):
                    manager.reset_ftp_buffers(
                        phase.ftp_buffer_bits
                    )

        self._current_phase_index = (
            phase_index
        )
