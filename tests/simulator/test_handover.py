import pytest
import torch

from oran_scheduler.simulator.handover import (
    HandoverConfig,
    HandoverController,
    NO_PENDING_TARGET,
)


def test_handover_requires_hysteresis() -> None:

    controller = HandoverController(
        initial_serving_bs=torch.tensor(
            [0],
            dtype=torch.long,
        ),
        num_bs=2,
        config=HandoverConfig(
            hysteresis_db=3.0,
            time_to_trigger_ttis=1,
        ),
    )

    #
    # BS 1 is only ~0.41 dB stronger.
    #
    result = controller.step(
        tti_index=0,
        link_power=torch.tensor(
            [
                [
                    1.0,
                    1.1,
                ]
            ],
            dtype=torch.float32,
        ),
    )

    assert result.serving_bs.item() == 0
    assert not result.handover_mask.item()

    assert (
        result.pending_target_bs.item()
        == NO_PENDING_TARGET
    )


def test_handover_requires_consecutive_ttt() -> None:

    controller = HandoverController(
        initial_serving_bs=torch.tensor(
            [0],
            dtype=torch.long,
        ),
        num_bs=2,
        config=HandoverConfig(
            hysteresis_db=3.0,
            time_to_trigger_ttis=2,
        ),
    )

    power = torch.tensor(
        [
            [
                1.0,
                4.0,
            ]
        ],
        dtype=torch.float32,
    )

    first = controller.step(
        tti_index=0,
        link_power=power,
    )

    assert first.serving_bs.item() == 0
    assert not first.handover_mask.item()
    assert first.pending_target_bs.item() == 1
    assert first.pending_count.item() == 1

    second = controller.step(
        tti_index=1,
        link_power=power,
    )

    assert second.handover_mask.item()
    assert second.serving_bs.item() == 1

    assert (
        second.pending_target_bs.item()
        == NO_PENDING_TARGET
    )

    assert second.pending_count.item() == 0


def test_changed_target_restarts_ttt() -> None:

    controller = HandoverController(
        initial_serving_bs=torch.tensor(
            [0],
            dtype=torch.long,
        ),
        num_bs=3,
        config=HandoverConfig(
            hysteresis_db=3.0,
            time_to_trigger_ttis=2,
        ),
    )

    first = controller.step(
        tti_index=0,
        link_power=torch.tensor(
            [
                [
                    1.0,
                    4.0,
                    2.0,
                ]
            ],
            dtype=torch.float32,
        ),
    )

    assert first.pending_target_bs.item() == 1
    assert first.pending_count.item() == 1

    second = controller.step(
        tti_index=1,
        link_power=torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    5.0,
                ]
            ],
            dtype=torch.float32,
        ),
    )

    #
    # Target changed 1 -> 2, so TTT restarts.
    #
    assert not second.handover_mask.item()
    assert second.serving_bs.item() == 0
    assert second.pending_target_bs.item() == 2
    assert second.pending_count.item() == 1


def test_multiple_ues_are_independent() -> None:

    controller = HandoverController(
        initial_serving_bs=torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
        ),
        num_bs=3,
        config=HandoverConfig(
            hysteresis_db=3.0,
            time_to_trigger_ttis=1,
        ),
    )

    result = controller.step(
        tti_index=0,
        link_power=torch.tensor(
            [
                [
                    1.0,
                    5.0,
                    0.5,
                ],
                [
                    0.5,
                    4.0,
                    1.0,
                ],
            ],
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        result.serving_bs,
        torch.tensor(
            [
                1,
                1,
            ],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        result.handover_mask,
        torch.tensor(
            [
                True,
                False,
            ],
            dtype=torch.bool,
        ),
    )


def test_nonconsecutive_tti_is_rejected() -> None:

    controller = HandoverController(
        initial_serving_bs=torch.tensor(
            [0],
            dtype=torch.long,
        ),
        num_bs=2,
        config=HandoverConfig(),
    )

    power = torch.tensor(
        [
            [
                1.0,
                1.0,
            ]
        ],
        dtype=torch.float32,
    )

    controller.step(
        tti_index=0,
        link_power=power,
    )

    with pytest.raises(
        ValueError,
        match="strictly",
    ):
        controller.step(
            tti_index=0,
            link_power=power,
        )


def test_invalid_power_shape_is_rejected() -> None:

    controller = HandoverController(
        initial_serving_bs=torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
        ),
        num_bs=3,
        config=HandoverConfig(),
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        controller.step(
            tti_index=0,
            link_power=torch.ones(
                2,
                2,
            ),
        )
