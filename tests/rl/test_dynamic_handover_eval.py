from types import SimpleNamespace

import torch

import oran_scheduler.rl.dynamic_handover_eval as eval_module

from oran_scheduler.rl.dynamic_handover_eval import (
    run_dynamic_multicell_ppo_evaluation,
)

from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)

from oran_scheduler.simulator.dynamic_handover_coordinator import (
    DynamicHandoverCoordinator,
)

from oran_scheduler.simulator.global_traffic_arrivals import (
    GlobalUETrafficArrivalProcess,
)

from oran_scheduler.simulator.global_ue_state import (
    GlobalUESchedulerStateRegistry,
)

from oran_scheduler.simulator.handover import (
    HandoverConfig,
    HandoverController,
)

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
)


FULL_BUFFER_BITS = 8000.0


def _ftp_config():
    return FTP3TrafficConfig(
        packet_size_bytes=100,
        packet_arrival_rate_per_s=0.0,
        tti_duration_s=0.0005,
    )


def _system():

    registry = GlobalUESchedulerStateRegistry(
        global_ue_indices=torch.tensor(
            [
                10,
                20,
                30,
                40,
            ],
            dtype=torch.long,
        ),

        serving_bs=torch.tensor(
            [
                0,
                0,
                1,
                1,
            ],
            dtype=torch.long,
        ),

        average_throughput_bps=torch.tensor(
            [
                1.0e6,
                2.0e6,
                3.0e6,
                4.0e6,
            ],
            dtype=torch.float32,
        ),

        buffer_bits=torch.tensor(
            [
                FULL_BUFFER_BITS,
                1000.0,
                FULL_BUFFER_BITS,
                2000.0,
            ],
            dtype=torch.float32,
        ),

        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                True,
                False,
            ],
            dtype=torch.bool,
        ),
    )

    handover = HandoverController(
        initial_serving_bs=(
            registry
            .snapshot()
            .serving_bs
        ),

        num_bs=2,

        config=HandoverConfig(
            hysteresis_db=3.0,
            time_to_trigger_ttis=1,
        ),
    )

    arrivals = GlobalUETrafficArrivalProcess(
        global_ue_indices=(
            registry
            .snapshot()
            .global_ue_indices
        ),

        full_buffer_mask=(
            registry
            .snapshot()
            .full_buffer_mask
        ),

        ftp3_config=_ftp_config(),

        seed=123,
    )

    coordinator = DynamicHandoverCoordinator(
        registry=registry,

        handover_controller=handover,

        selected_cell_indices=(
            0,
            1,
        ),

        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),

        throughput_forgetting_factor=0.9,

        ftp3_config=_ftp_config(),

        full_buffer_state_bits=(
            FULL_BUFFER_BITS
        ),

        traffic_arrival_process=(
            arrivals
        ),
    )

    return (
        registry,
        coordinator,
    )


def test_dynamic_eval_rebuilds_membership_after_handover(
    monkeypatch,
):

    (
        registry,
        coordinator,
    ) = _system()

    observed_calls = []

    def link_power_provider(
        tti_index,
    ):

        if tti_index == 0:

            return torch.tensor(
                [
                    [5.0, 1.0],
                    [5.0, 1.0],
                    [1.0, 5.0],
                    [1.0, 5.0],
                ],
                dtype=torch.float32,
            )

        return torch.tensor(
            [
                [5.0, 1.0],
                [1.0, 5.0],
                [1.0, 5.0],
                [1.0, 5.0],
            ],
            dtype=torch.float32,
        )

    def radio_provider(
        tti_index,
        stream_index,
    ):

        ids = (
            coordinator
            .membership_provider(
                tti_index,
                stream_index,
            )
        )

        observation = SimpleNamespace(
            serving_global_ue_indices=(
                ids.clone()
            )
        )

        return PPOTrainingTTIInputs(
            observation=observation,

            physical_inputs_builder=(
                lambda prepared: prepared
            ),

            packet_arrivals=None,
        )

    def fake_cell_step(
        **kwargs,
    ):

        ids = (
            kwargs[
                "observation"
            ]
            .serving_global_ue_indices
        )

        observed_calls.append(
            (
                kwargs[
                    "tti_index"
                ],

                tuple(
                    ids.tolist()
                ),

                tuple(
                    kwargs[
                        "packet_arrivals"
                    ].tolist()
                ),
            )
        )

        assert (
            kwargs[
                "collect_experience"
            ]
            is False
        )

        return SimpleNamespace(
            tti_index=(
                kwargs[
                    "tti_index"
                ]
            )
        )

    monkeypatch.setattr(
        eval_module,
        "run_traffic_aware_ppo_cell_tti_step",
        fake_cell_step,
    )

    result = (
        run_dynamic_multicell_ppo_evaluation(
            start_tti_index=0,

            num_ttis=2,

            radio_input_provider=(
                radio_provider
            ),

            handover_link_power_provider=(
                link_power_provider
            ),

            coordinator=coordinator,

            rollout_controllers=(
                object(),
                object(),
            ),

            tds_eligibility_config=object(),

            state_config=object(),

            greedy_config=object(),

            reward_config=object(),

            reward_population=(
                "serving_ues"
            ),

            device="cpu",
        )
    )

    assert observed_calls[
        0
    ][1] == (
        10,
        20,
    )

    assert observed_calls[
        1
    ][1] == (
        30,
        40,
    )

    assert observed_calls[
        2
    ][1] == (
        10,
    )

    assert observed_calls[
        3
    ][1] == (
        20,
        30,
        40,
    )

    assert (
        result.num_completed_handovers
        == 1
    )

    ue20 = registry.gather(
        torch.tensor(
            [
                20,
            ],
            dtype=torch.long,
        )
    )

    assert (
        ue20
        .serving_bs[
            0
        ]
        .item()
        == 1
    )


def test_dynamic_eval_rejects_radio_membership_mismatch(
    monkeypatch,
):

    (
        _,
        coordinator,
    ) = _system()

    def link_power_provider(
        tti_index,
    ):

        del tti_index

        return torch.tensor(
            [
                [5.0, 1.0],
                [5.0, 1.0],
                [1.0, 5.0],
                [1.0, 5.0],
            ],
            dtype=torch.float32,
        )

    def bad_radio_provider(
        tti_index,
        stream_index,
    ):

        del tti_index

        ids = (
            torch.tensor(
                [999],
                dtype=torch.long,
            )
            if stream_index == 0
            else torch.tensor(
                [
                    30,
                    40,
                ],
                dtype=torch.long,
            )
        )

        return PPOTrainingTTIInputs(
            observation=(
                SimpleNamespace(
                    serving_global_ue_indices=ids
                )
            ),

            physical_inputs_builder=(
                lambda prepared: prepared
            ),
        )

    def impossible_step(
        **kwargs,
    ):

        del kwargs

        raise AssertionError(
            "Cell step must not execute."
        )

    monkeypatch.setattr(
        eval_module,
        "run_traffic_aware_ppo_cell_tti_step",
        impossible_step,
    )

    try:
        run_dynamic_multicell_ppo_evaluation(
            start_tti_index=0,

            num_ttis=1,

            radio_input_provider=(
                bad_radio_provider
            ),

            handover_link_power_provider=(
                link_power_provider
            ),

            coordinator=coordinator,

            rollout_controllers=(
                object(),
                object(),
            ),

            tds_eligibility_config=object(),

            state_config=object(),

            greedy_config=object(),

            reward_config=object(),

            reward_population=(
                "serving_ues"
            ),

            device="cpu",
        )

    except RuntimeError as error:

        assert (
            "membership"
            in str(error)
        )

        return

    raise AssertionError(
        "Expected membership mismatch."
    )


def test_transition_observer_runs_before_cell_execution(
    monkeypatch,
):
    (
        _,
        coordinator,
    ) = _system()

    order = []

    def link_power_provider(
        tti_index,
    ):
        del tti_index

        return torch.tensor(
            [
                [5.0, 1.0],
                [5.0, 1.0],
                [1.0, 5.0],
                [1.0, 5.0],
            ],
            dtype=torch.float32,
        )

    def radio_provider(
        tti_index,
        stream_index,
    ):
        ids = (
            coordinator
            .membership_provider(
                tti_index,
                stream_index,
            )
        )

        return PPOTrainingTTIInputs(
            observation=SimpleNamespace(
                serving_global_ue_indices=(
                    ids.clone()
                )
            ),

            physical_inputs_builder=(
                lambda prepared: prepared
            ),
        )

    def transition_observer(
        tti_index,
        transition,
    ):
        del transition

        order.append(
            (
                "transition",
                tti_index,
            )
        )

    def fake_cell_step(
        **kwargs,
    ):
        order.append(
            (
                "cell",
                kwargs[
                    "tti_index"
                ],
            )
        )

        return SimpleNamespace(
            tti_index=(
                kwargs[
                    "tti_index"
                ]
            )
        )

    monkeypatch.setattr(
        eval_module,
        "run_traffic_aware_ppo_cell_tti_step",
        fake_cell_step,
    )

    run_dynamic_multicell_ppo_evaluation(
        start_tti_index=0,

        num_ttis=1,

        radio_input_provider=(
            radio_provider
        ),

        handover_link_power_provider=(
            link_power_provider
        ),

        coordinator=coordinator,

        rollout_controllers=(
            object(),
            object(),
        ),

        tds_eligibility_config=object(),

        state_config=object(),

        greedy_config=object(),

        reward_config=object(),

        reward_population=(
            "serving_ues"
        ),

        transition_observer=(
            transition_observer
        ),

        device="cpu",
    )

    assert order == [
        (
            "transition",
            0,
        ),
        (
            "cell",
            0,
        ),
        (
            "cell",
            0,
        ),
    ]
