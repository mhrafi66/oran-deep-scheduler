from types import SimpleNamespace

import torch

from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.simulator.ue_availability import (
    UEAvailabilityInputProvider,
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
        ],
        dtype=torch.bool,
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
            ],
            dtype=torch.bool,
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


def test_provider_preserves_association_and_arrivals() -> None:

    observation = SimpleNamespace(
        serving_global_ue_indices=(
            torch.tensor(
                [
                    10,
                    20,
                    30,
                ],
                dtype=torch.long,
            )
        ),

        serving_ue_valid_mask=(
            torch.tensor(
                [
                    True,
                    True,
                    True,
                ],
                dtype=torch.bool,
            )
        ),
    )

    arrivals = torch.tensor(
        [
            1,
            2,
            3,
        ],
        dtype=torch.long,
    )

    base = PPOTrainingTTIInputs(
        observation=observation,
        physical_inputs_builder=(
            lambda prepared: None
        ),
        packet_arrivals=arrivals,
    )

    def base_provider(
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:
        del tti_index
        del stream_index
        return base

    scenario = UEAvailabilityScenario(
        phases=(
            UEAvailabilityPhase(
                start_tti=0,
                name="normal",
            ),

            UEAvailabilityPhase(
                start_tti=1,
                name="outage",
                unavailable_global_ue_indices=(
                    20,
                ),
            ),

            UEAvailabilityPhase(
                start_tti=3,
                name="recovery",
            ),
        )
    )

    provider = UEAvailabilityInputProvider(
        base_provider=base_provider,
        scenario=scenario,
    )

    stressed = provider(
        1,
        0,
    )

    #
    # Association/layout is untouched.
    #
    assert (
        stressed.observation
        is observation
    )

    torch.testing.assert_close(
        stressed
        .observation
        .serving_ue_valid_mask,
        torch.tensor(
            [
                True,
                True,
                True,
            ],
            dtype=torch.bool,
        ),
    )

    #
    # Traffic continues arriving while unavailable.
    #
    torch.testing.assert_close(
        stressed.packet_arrivals,
        arrivals,
    )

    #
    # Only schedulability changes.
    #
    torch.testing.assert_close(
        stressed
        .tds_eligibility_override_mask,
        torch.tensor(
            [
                True,
                False,
                True,
            ],
            dtype=torch.bool,
        ),
    )


def test_provider_recovery_restores_schedulability() -> None:

    observation = SimpleNamespace(
        serving_global_ue_indices=(
            torch.tensor(
                [
                    10,
                    20,
                    30,
                ],
                dtype=torch.long,
            )
        ),

        serving_ue_valid_mask=(
            torch.tensor(
                [
                    True,
                    True,
                    True,
                ],
                dtype=torch.bool,
            )
        ),
    )

    base = PPOTrainingTTIInputs(
        observation=observation,
        physical_inputs_builder=(
            lambda prepared: None
        ),
        packet_arrivals=None,
    )

    provider = UEAvailabilityInputProvider(
        base_provider=(
            lambda tti_index, stream_index: base
        ),

        scenario=UEAvailabilityScenario(
            phases=(
                UEAvailabilityPhase(
                    start_tti=0,
                    name="normal",
                ),

                UEAvailabilityPhase(
                    start_tti=1,
                    name="outage",
                    unavailable_global_ue_indices=(
                        20,
                    ),
                ),

                UEAvailabilityPhase(
                    start_tti=3,
                    name="recovery",
                ),
            )
        ),
    )

    recovered = provider(
        3,
        0,
    )

    torch.testing.assert_close(
        recovered
        .tds_eligibility_override_mask,
        torch.tensor(
            [
                True,
                True,
                True,
            ],
            dtype=torch.bool,
        ),
    )
