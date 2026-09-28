from oran_scheduler.simulator.network_events import (
    PhysicalNetworkPhase,
    PhysicalNetworkScenario,
)


def test_network_event_phase_selection() -> None:

    scenario = PhysicalNetworkScenario(
        phases=(
            PhysicalNetworkPhase(
                start_tti=0,
                name="normal",
            ),

            PhysicalNetworkPhase(
                start_tti=20,
                name="interference",
                non_serving_interference_power_scale=(
                    4.0
                ),
            ),

            PhysicalNetworkPhase(
                start_tti=40,
                name="bs_failure",
                failed_bs_index=0,
            ),

            PhysicalNetworkPhase(
                start_tti=60,
                name="recovery",
            ),
        )
    )

    assert (
        scenario.phase_for_tti(0).name
        == "normal"
    )

    assert (
        scenario.phase_for_tti(39).name
        == "interference"
    )

    assert (
        scenario.phase_for_tti(40).name
        == "bs_failure"
    )

    assert (
        scenario.phase_for_tti(100).name
        == "recovery"
    )
