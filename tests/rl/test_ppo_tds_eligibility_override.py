import pytest
import torch

from oran_scheduler.rl.ppo_traffic_cell_step import (
    _apply_tds_eligibility_override,
)
from oran_scheduler.simulator.tds_eligibility import (
    TDSEligibilityData,
)


def _base_eligibility() -> TDSEligibilityData:

    return TDSEligibilityData(
        serving_ue_valid_mask=torch.tensor(
            [
                True,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
        ),

        has_finite_buffer_data=torch.tensor(
            [
                True,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
        ),

        eligible_mask=torch.tensor(
            [
                True,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
        ),

        num_valid_serving_ues=3,

        num_eligible_ues=3,
    )


def test_override_removes_temporarily_unavailable_ue():

    result = (
        _apply_tds_eligibility_override(
            eligibility=(
                _base_eligibility()
            ),

            override_mask=torch.tensor(
                [
                    True,
                    False,
                    True,
                    False,
                ],
                dtype=torch.bool,
            ),
        )
    )

    #
    # Persistent association stays unchanged.
    #
    torch.testing.assert_close(
        result.serving_ue_valid_mask,
        torch.tensor(
            [
                True,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
        ),
    )

    #
    # Scheduling eligibility changes.
    #
    torch.testing.assert_close(
        result.eligible_mask,
        torch.tensor(
            [
                True,
                False,
                True,
                False,
            ],
            dtype=torch.bool,
        ),
    )

    assert (
        result.num_valid_serving_ues
        == 3
    )

    assert (
        result.num_eligible_ues
        == 2
    )


def test_override_cannot_enable_padding():

    with pytest.raises(
        ValueError,
        match="cannot enable",
    ):

        _apply_tds_eligibility_override(
            eligibility=(
                _base_eligibility()
            ),

            override_mask=torch.tensor(
                [
                    True,
                    True,
                    True,
                    True,
                ],
                dtype=torch.bool,
            ),
        )
