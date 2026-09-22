import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIStateManager,
)
from oran_scheduler.simulator.runtime_scenario import (
    RuntimeScenarioController,
    RuntimeScenarioPhase,
    parse_runtime_scenario_json,
)
from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
    TrafficBufferManager,
)


def _traffic_manager():
    return TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [True, False],
            dtype=torch.bool,
        ),
        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=100,
            packet_arrival_rate_per_s=10.0,
            tti_duration_s=0.001,
        ),
        full_buffer_state_bits=999.0,
        initial_ftp_buffer_bits=torch.tensor(
            [0.0, 50.0],
            dtype=torch.float32,
        ),
        seed=1,
    )


def _state_manager():
    return OneLDSCellTTIStateManager(
        initial_average_throughput_bps=torch.tensor(
            [10.0, 20.0],
            dtype=torch.float32,
        ),
        serving_global_ue_indices=torch.tensor(
            [1, 2],
            dtype=torch.long,
        ),
        serving_ue_valid_mask=torch.tensor(
            [True, True],
            dtype=torch.bool,
        ),
        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),
        throughput_forgetting_factor=0.9,
    )


def test_parse_runtime_scenario() -> None:

    phases = parse_runtime_scenario_json(
        """
        [
          {
            "start_tti": 0,
            "name": "normal"
          },
          {
            "start_tti": 10,
            "name": "overload",
            "ftp_rate_scale": 4.0
          }
        ]
        """
    )

    assert len(
        phases
    ) == 2

    assert phases[
        1
    ].ftp_rate_scale == 4.0


def test_dynamic_ftp_scaling_uses_original_config() -> None:

    traffic = _traffic_manager()

    state = _state_manager()

    controller = RuntimeScenarioController(
        phases=(
            RuntimeScenarioPhase(
                start_tti=0,
                ftp_rate_scale=2.0,
            ),
            RuntimeScenarioPhase(
                start_tti=5,
                ftp_rate_scale=3.0,
            ),
        ),
        traffic_managers=(
            traffic,
        ),
        state_managers=(
            state,
        ),
    )

    controller.apply_tti(
        0
    )

    assert (
        traffic
        .ftp3_config
        .packet_arrival_rate_per_s
        == 20.0
    )

    controller.apply_tti(
        5
    )

    #
    # 3 x ORIGINAL 10, not 3 x previous 20.
    #
    assert (
        traffic
        .ftp3_config
        .packet_arrival_rate_per_s
        == 30.0
    )


def test_phase_entry_can_reset_pf_and_queue() -> None:

    traffic = _traffic_manager()

    state = _state_manager()

    controller = RuntimeScenarioController(
        phases=(
            RuntimeScenarioPhase(
                start_tti=0,
                name="before",
            ),
            RuntimeScenarioPhase(
                start_tti=5,
                name="restart",
                reset_pf_history=True,
                pf_history_bps=7.0,
                reset_ftp_buffers=True,
                ftp_buffer_bits=3.0,
            ),
        ),
        traffic_managers=(
            traffic,
        ),
        state_managers=(
            state,
        ),
    )

    controller.apply_tti(
        0
    )

    controller.apply_tti(
        5
    )

    torch.testing.assert_close(
        state
        .current_average_throughput_bps,
        torch.tensor(
            [7.0, 7.0]
        ),
    )

    torch.testing.assert_close(
        traffic.current_buffer_bits,
        torch.tensor(
            [999.0, 3.0]
        ),
    )

    assert (
        controller.current_phase_name
        == "restart"
    )
