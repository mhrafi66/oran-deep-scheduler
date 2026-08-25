import torch

from oran_scheduler.state.cqi_features import (
    CQISurrogateConfig,
    build_cqi_surrogate,
)


def test_cqi_surrogate_maps_mcs_and_bler():
    device = "cuda:0"

    config = CQISurrogateConfig(
        max_mcs_index=27,
        min_reported_cqi=1,
        max_reported_cqi=15,
    )

    mcs_index = torch.tensor(
        [
            [0, 9, 18, 27],
            [3, 6, 9, 12],
        ],
        dtype=torch.long,
        device=device,
    )


    meets_bler_target = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, False],
        ],
        dtype=torch.bool,
        device=device,
    )

    valid_mask = torch.tensor(
        [True, True],
        dtype=torch.bool,
        device=device,
    )

    result = build_cqi_surrogate(
        mcs_index=mcs_index,
        meets_bler_target=(
            meets_bler_target
        ),
        candidate_valid_mask=valid_mask,
        config=config,
    )

    expected_candidate_0 = torch.tensor(
        [1, 6, 10, 15],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result.subband_cqi[
            0
        ],
        expected_candidate_0,
    )

    assert int(
        result.subband_cqi[
            1,
            1,
        ].item()
    ) == 0

    assert int(
        result.subband_cqi[
            1,
            3,
        ].item()
    ) == 0

def test_wideband_cqi_is_rounded_subband_mean():
    device = "cuda:0"

    config = CQISurrogateConfig()


    mcs_index = torch.tensor(
        [
            [0, 27, 0, 27],
        ],
        dtype=torch.long,
        device=device,
    )

    meets_target = torch.ones(
        (
            1,
            4,
        ),
        dtype=torch.bool,
        device=device,
    )

    valid_mask = torch.tensor(
        [True],
        dtype=torch.bool,
        device=device,
    )

    result = build_cqi_surrogate(
        mcs_index=mcs_index,
        meets_bler_target=meets_target,
        candidate_valid_mask=valid_mask,
        config=config,
    )

    expected_subband = torch.tensor(
        [
            [1, 15, 1, 15],
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result.subband_cqi,
        expected_subband,
    )

    assert int(
        result.wideband_cqi[
            0
        ].item()
    ) == 8

def test_invalid_candidate_has_zero_cqi():
    device = "cuda:0"

    config = CQISurrogateConfig()

    mcs_index = torch.tensor(
        [
            [10, 20],
            [-1, -1],
        ],
        dtype=torch.long,
        device=device,
    )

    valid_mask = torch.tensor(
        [True, False],
        dtype=torch.bool,
        device=device,
    )

    meets_target = torch.tensor(
        [
            [True, True],
            [True, True],
        ],
        dtype=torch.bool,
        device=device,
    )

    result = build_cqi_surrogate(
        mcs_index=mcs_index,
        meets_bler_target=meets_target,
        candidate_valid_mask=valid_mask,
        config=config,
    )

    torch.testing.assert_close(
        result.subband_cqi[
            1
        ],
        torch.zeros(
            2,
            dtype=torch.long,
            device=device,
        ),
    )

    assert int(
        result.wideband_cqi[
            1
        ].item()
    ) == 0

def test_cqi_surrogate_paper_dimensions():
    device = "cuda:0"

    batch_size = 2
    num_cells = 3
    num_candidates = 10
    num_rbgs = 18

    mcs_shape = (
        batch_size,
        num_cells,
        num_candidates,
        num_rbgs,
    )

    candidate_shape = (
        batch_size,
        num_cells,
        num_candidates,
    )

    mcs_index = torch.full(
        mcs_shape,
        fill_value=15,
        dtype=torch.long,
        device=device,
    )

    meets_target = torch.ones(
        mcs_shape,
        dtype=torch.bool,
        device=device,
    )

    valid_mask = torch.ones(
        candidate_shape,
        dtype=torch.bool,
        device=device,
    )

    result = build_cqi_surrogate(
        mcs_index=mcs_index,
        meets_bler_target=meets_target,
        candidate_valid_mask=valid_mask,
        config=CQISurrogateConfig(),
    )

    assert result.subband_cqi.shape == (
        2,
        3,
        10,
        18,
    )

    assert result.wideband_cqi.shape == (
        2,
        3,
        10,
    )