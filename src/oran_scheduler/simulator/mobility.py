from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from oran_scheduler.simulator.topology import (
    TopologyData,
)


_ALLOWED_TRAJECTORY_MODES = {
    "static",
    "constant_velocity",
}

_ALLOWED_ASSOCIATION_MODES = {
    "fixed",
}

_ALLOWED_BOUNDARY_MODES = {
    "error",
}


@dataclass(frozen=True)
class MobilityConfig:
    """
    Persistent UE-mobility configuration.

    IMPORTANT SCIENTIFIC SCOPE
    --------------------------
    This object controls UE geometry evolution.

    It does NOT by itself claim full 3GPP spatial
    consistency of the stochastic channel.

    The first Wave-4 implementation intentionally
    separates:

        UE kinematics
            from
        temporal radio evolution.

    trajectory_mode:
        "static"
            UE positions remain fixed.

        "constant_velocity"
            UE positions evolve according to

                x(t) = x(0) + v * t

    association_mode:
        Only "fixed" is currently supported.

        This deliberately isolates mobility/channel
        aging from handover behavior.

    boundary_mode:
        Only "error" is currently supported.

        If max_horizontal_displacement_m is set,
        movement beyond that displacement raises
        rather than silently applying an arbitrary
        bounce/wrap rule.

    OPEN-REPRODUCTION:
        tti_duration_s is explicit because the Nokia
        paper does not define our simulator's Python
        scheduling-step duration.
    """

    tti_duration_s: float = 0.001

    trajectory_mode: str = (
        "constant_velocity"
    )

    association_mode: str = "fixed"

    boundary_mode: str = "error"

    preserve_ut_height: bool = True

    max_horizontal_displacement_m: (
        float | None
    ) = None


    def __post_init__(
        self,
    ) -> None:

        if (
            not math.isfinite(
                self.tti_duration_s
            )
            or self.tti_duration_s <= 0.0
        ):
            raise ValueError(
                "tti_duration_s must be finite "
                "and positive."
            )

        if (
            self.trajectory_mode
            not in _ALLOWED_TRAJECTORY_MODES
        ):
            raise ValueError(
                "Unsupported trajectory_mode. "
                f"Got {self.trajectory_mode!r}."
            )

        if (
            self.association_mode
            not in _ALLOWED_ASSOCIATION_MODES
        ):
            raise ValueError(
                "Only fixed association is "
                "implemented in Wave 4."
            )

        if (
            self.boundary_mode
            not in _ALLOWED_BOUNDARY_MODES
        ):
            raise ValueError(
                "Only boundary_mode='error' is "
                "implemented in Wave 4."
            )

        if (
            self.max_horizontal_displacement_m
            is not None
        ):
            if (
                not math.isfinite(
                    self
                    .max_horizontal_displacement_m
                )
                or (
                    self
                    .max_horizontal_displacement_m
                    <= 0.0
                )
            ):
                raise ValueError(
                    "max_horizontal_displacement_m "
                    "must be finite and positive "
                    "when provided."
                )


@dataclass(frozen=True)
class MobilitySnapshot:
    """
    UE geometry at one simulator TTI.

    topology:
        TopologyData containing the current UE
        positions and the persistent velocity vectors.

    displacement_m:
        Euclidean horizontal displacement from the
        initial location.

        Shape:
            [batch, UE]
    """

    tti_index: int

    elapsed_time_s: float

    topology: TopologyData

    displacement_m: torch.Tensor


    @property
    def mean_displacement_m(
        self,
    ) -> float:
        return float(
            self
            .displacement_m
            .mean()
            .item()
        )


    @property
    def max_displacement_m(
        self,
    ) -> float:
        return float(
            self
            .displacement_m
            .max()
            .item()
        )


def _horizontal_velocities(
    topology: TopologyData,
) -> torch.Tensor:
    """
    Return the velocity used by the mobility engine.

    UE height is kept fixed in the first implementation,
    so the z component is forced to zero.
    """

    velocity = (
        topology
        .ut_velocities
        .detach()
        .clone()
    )

    if velocity.ndim != 3:
        raise ValueError(
            "ut_velocities must have shape "
            "[batch, UE, xyz]."
        )

    if velocity.shape[-1] != 3:
        raise ValueError(
            "ut_velocities last dimension "
            "must be xyz."
        )

    velocity[..., 2] = 0.0

    return velocity


def build_mobility_snapshot(
    *,
    initial_topology: TopologyData,
    config: MobilityConfig,
    tti_index: int,
) -> MobilitySnapshot:
    """
    Deterministically construct UE geometry for one TTI.

    This function is stateless.

    Therefore:

        same topology
        same config
        same TTI

    always gives the same UE positions.
    """

    if tti_index < 0:
        raise ValueError(
            "tti_index must be non-negative."
        )

    initial_ut_loc = (
        initial_topology
        .ut_loc
    )

    if (
        initial_ut_loc.ndim != 3
        or initial_ut_loc.shape[-1] != 3
    ):
        raise ValueError(
            "ut_loc must have shape "
            "[batch, UE, xyz]."
        )

    velocity = _horizontal_velocities(
        initial_topology
    )

    if (
        velocity.shape
        != initial_ut_loc.shape
    ):
        raise ValueError(
            "UE location and velocity shapes "
            "must match."
        )

    elapsed_time_s = (
        float(tti_index)
        * config.tti_duration_s
    )

    if (
        config.trajectory_mode
        == "static"
    ):
        current_ut_loc = (
            initial_ut_loc
            .detach()
            .clone()
        )

    elif (
        config.trajectory_mode
        == "constant_velocity"
    ):
        current_ut_loc = (
            initial_ut_loc
            + velocity
            * elapsed_time_s
        )

    else:
        raise RuntimeError(
            "Unreachable trajectory mode."
        )

    if config.preserve_ut_height:
        current_ut_loc = (
            current_ut_loc
            .clone()
        )

        current_ut_loc[..., 2] = (
            initial_ut_loc[..., 2]
        )

    horizontal_delta = (
        current_ut_loc[..., :2]
        - initial_ut_loc[..., :2]
    )

    displacement_m = (
        torch.linalg.vector_norm(
            horizontal_delta,
            dim=-1,
        )
    )

    if (
        config.max_horizontal_displacement_m
        is not None
        and torch.any(
            displacement_m
            >
            config.max_horizontal_displacement_m
        )
    ):
        raise RuntimeError(
            "UE displacement exceeded the "
            "configured short-window mobility "
            "limit. A proper boundary/wraparound "
            "model is required before continuing."
        )

    #
    # IMPORTANT:
    #
    # bs_virtual_loc is intentionally preserved.
    #
    # For the first short-window fixed-association
    # mobility experiments we do not allow movement
    # large enough to require a wraparound-image
    # reassignment.
    #
    current_topology = replace(
        initial_topology,

        ut_loc=current_ut_loc,

        ut_velocities=velocity,
    )

    return MobilitySnapshot(
        tti_index=tti_index,

        elapsed_time_s=(
            elapsed_time_s
        ),

        topology=current_topology,

        displacement_m=(
            displacement_m
        ),
    )


class MobilityEngine:
    """
    Persistent monotonic simulator-time mobility engine.

    The underlying geometry is computed from the
    INITIAL topology rather than repeatedly adding
    floating-point increments.

    This prevents accumulated numerical drift:

        x_t = x_0 + v * t

    rather than

        x_t = x_(t-1) + v * dt
    """

    def __init__(
        self,
        *,
        initial_topology: TopologyData,
        config: MobilityConfig,
        initial_tti_index: int = 0,
    ) -> None:

        if initial_tti_index < 0:
            raise ValueError(
                "initial_tti_index must be "
                "non-negative."
            )

        self._initial_topology = (
            initial_topology
        )

        self._config = config

        self._current_snapshot = (
            build_mobility_snapshot(
                initial_topology=(
                    initial_topology
                ),
                config=config,
                tti_index=(
                    initial_tti_index
                ),
            )
        )


    @property
    def config(
        self,
    ) -> MobilityConfig:
        return self._config


    @property
    def current_snapshot(
        self,
    ) -> MobilitySnapshot:
        return self._current_snapshot


    @property
    def current_tti_index(
        self,
    ) -> int:
        return (
            self
            ._current_snapshot
            .tti_index
        )


    @property
    def current_topology(
        self,
    ) -> TopologyData:
        return (
            self
            ._current_snapshot
            .topology
        )


    def advance_to(
        self,
        tti_index: int,
    ) -> MobilitySnapshot:

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if (
            tti_index
            < self.current_tti_index
        ):
            raise ValueError(
                "MobilityEngine cannot move "
                "backward in simulator time."
            )

        if (
            tti_index
            == self.current_tti_index
        ):
            return self.current_snapshot

        self._current_snapshot = (
            build_mobility_snapshot(
                initial_topology=(
                    self
                    ._initial_topology
                ),
                config=self._config,
                tti_index=tti_index,
            )
        )

        return self.current_snapshot
