import torch

from oran_scheduler.simulator.ue_availability import (
    UEAvailabilityPhase,
    UEAvailabilityScenario,
)


def test_explicit_ue_availability_phase() -> None:

    scenario = UEAvailabilityScenario(
        phases=(
            UEAvailabilityPhase(
                start_tti=0,
                name="normal",
            ),

            UEAvailabilityPhase(
                start_tti=10,
                name="outage",
                unavailable_global_ue_indices=(
                    20,
                ),
            ),

            UEAvailabilityPhase(
                start_tti=20,
                name="recovery",
            ),
        )
    )

    ids = torch.tensor(
        [
            10,
            20,
            30,
        ],
        dtype=torch.long,
    )

    valid = torch.tensor(
        [
            True,
            True,
            True,
        ]
    )

    mask = scenario.unavailable_mask(
        global_ue_indices=ids,
        existing_valid_mask=valid,
        tti_index=15,
    )

    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                False,
                True,
                False,
            ]
        ),
    )


def test_fraction_is_deterministic() -> None:

    scenario = UEAvailabilityScenario(
        phases=(
            UEAvailabilityPhase(
                start_tti=0,
                name="outage",
                unavailable_fraction=0.5,
            ),
        ),
        seed=123,
    )

    ids = torch.arange(
        100,
        dtype=torch.long,
    )

    valid = torch.ones(
        100,
        dtype=torch.bool,
    )

    first = scenario.unavailable_mask(
        global_ue_indices=ids,
        existing_valid_mask=valid,
        tti_index=0,
    )

    second = scenario.unavailable_mask(
        global_ue_indices=ids,
        existing_valid_mask=valid,
        tti_index=10,
    )

    torch.testing.assert_close(
        first,
        second,
    )

    assert (
        20
        < int(first.sum().item())
        < 80
    )
