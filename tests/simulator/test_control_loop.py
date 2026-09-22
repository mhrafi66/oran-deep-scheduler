from oran_scheduler.simulator.control_loop import (
    ControlLoopTimingConfig,
    ControlLoopTimingModel,
)


def test_control_loop_deadline_hit() -> None:

    model = ControlLoopTimingModel(
        ControlLoopTimingConfig(
            deadline_ms=1.0,
            base_compute_ms=0.5,
            jitter_std_ms=0.0,
        )
    )

    decision = model.decision(
        0
    )

    assert not decision.deadline_missed


def test_control_loop_deadline_miss() -> None:

    model = ControlLoopTimingModel(
        ControlLoopTimingConfig(
            deadline_ms=0.5,
            base_compute_ms=1.0,
            jitter_std_ms=0.0,
            miss_policy="empty",
        )
    )

    decision = model.decision(
        0
    )

    assert decision.deadline_missed
