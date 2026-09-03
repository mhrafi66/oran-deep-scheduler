import pytest
import torch

from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
    build_tds_eligibility,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def test_all_serving_ues_keeps_empty_ftp_eligible():
    device = preferred_device()

    result = build_tds_eligibility(
        serving_ue_valid_mask=torch.tensor(
            [
                True,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        dl_buffer_bits=torch.tensor(
            [
                4000.0,
                0.0,
                8000.0,
                0.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                False,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        config=TDSBufferEligibilityConfig(
            mode="all_serving_ues",
        ),
    )

    torch.testing.assert_close(
        result.eligible_mask,
        torch.tensor(
            [
                True,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
    )

    assert result.num_valid_serving_ues == 3
    assert result.num_eligible_ues == 3

def test_data_available_only_excludes_empty_ftp():
    device = preferred_device()

    result = build_tds_eligibility(
        serving_ue_valid_mask=torch.tensor(
            [
                True,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        dl_buffer_bits=torch.tensor(
            [
                4000.0,
                0.0,
                8000.0,
                999999.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                False,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        config=TDSBufferEligibilityConfig(
            mode="data_available_only",
        ),
    )

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
            device=device,
        ),
    )

    assert result.num_valid_serving_ues == 3
    assert result.num_eligible_ues == 2

def test_full_buffer_is_always_data_eligible():
    device = preferred_device()

    result = build_tds_eligibility(
        serving_ue_valid_mask=torch.tensor(
            [
                True,
            ],
            dtype=torch.bool,
            device=device,
        ),
        dl_buffer_bits=torch.tensor(
            [
                0.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        full_buffer_mask=torch.tensor(
            [
                True,
            ],
            dtype=torch.bool,
            device=device,
        ),
        config=TDSBufferEligibilityConfig(
            mode="data_available_only",
        ),
    )

    assert bool(
        result.eligible_mask[0].item()
    )

def test_finite_buffer_minimum_threshold():
    device = preferred_device()

    result = build_tds_eligibility(
        serving_ue_valid_mask=torch.ones(
            3,
            dtype=torch.bool,
            device=device,
        ),
        dl_buffer_bits=torch.tensor(
            [
                0.0,
                1.0,
                100.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        full_buffer_mask=torch.zeros(
            3,
            dtype=torch.bool,
            device=device,
        ),
        config=TDSBufferEligibilityConfig(
            mode="data_available_only",
            finite_buffer_min_bits=1.0,
        ),
    )

    torch.testing.assert_close(
        result.eligible_mask,
        torch.tensor(
            [
                False,
                False,
                True,
            ],
            dtype=torch.bool,
            device=device,
        ),
    )

def test_tds_eligibility_config_rejects_invalid_values():
    with pytest.raises(
        ValueError
    ):
        TDSBufferEligibilityConfig(
            mode="something_else",
        )

    with pytest.raises(
        ValueError
    ):
        TDSBufferEligibilityConfig(
            mode="data_available_only",
            finite_buffer_min_bits=-1.0,
        )

