import torch

from oran_scheduler.state.one_lds import (
    OneLDSRawFeatures,
    OneLDSStateConfig,
    build_1lds_state,
)

def test_1lds_state_feature_ordering():
    device = "cuda:0"

    config = OneLDSStateConfig(
        throughput_normalization_bps=4.0,
        buffer_normalization=100.0,
        subband_cqi_normalization=10.0,
        num_candidates=2,
        num_rbgs=2,
        max_rank=2,
        wideband_cqi_min=0.0,
        wideband_cqi_max=15.0,
    )


    features = OneLDSRawFeatures(
        past_average_throughput=torch.tensor(
            [2.0, 4.0],
            device=device,
        ),
        rank=torch.tensor(
            [1, 2],
            dtype=torch.long,
            device=device,
        ),
        allocated_rbg_count=torch.tensor(
            [1, 2],
            dtype=torch.long,
            device=device,
        ),
        dl_buffer=torch.tensor(
            [25.0, 50.0],
            device=device,
        ),
        wideband_cqi=torch.tensor(
            [5.0, 10.0],
            device=device,
        ),

        subband_cqi=torch.tensor(
            [
                [2.0, 4.0],
                [6.0, 8.0],
            ],
            device=device,
        ),
        max_precoder_cross_correlation=torch.tensor(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            device=device,
        ),
        candidate_valid_mask=torch.tensor(
            [True, True],
            dtype=torch.bool,
            device=device,
        ),
    )

    result = build_1lds_state(
        features=features,
        config=config,
    )

    assert result.ue_feature_segments.shape == (
        2,
        9,
    )

    assert result.state.shape == (
        18,
    )

    expected_candidate_0 = torch.tensor(
        [
            0.5,
            0.5,
            0.5,
            0.25,
            5.0 / 15.0,
            0.2,
            0.4,
            0.1,
            0.2,
        ],
        dtype=torch.float32,
        device=device,
    )

    torch.testing.assert_close(
        result.ue_feature_segments[
            0
        ],
        expected_candidate_0,
    )

def test_1lds_blanks_invalid_candidate_segment():
    device = "cuda:0"

    config = OneLDSStateConfig(
        throughput_normalization_bps=10.0,
        buffer_normalization=100.0,
        subband_cqi_normalization=15.0,
        num_candidates=2,
        num_rbgs=2,
    )

    features = OneLDSRawFeatures(
        past_average_throughput=torch.tensor(
            [5.0, 9.0],
            device=device,
        ),
        rank=torch.tensor(
            [1, 0],
            dtype=torch.long,
            device=device,
        ),
        allocated_rbg_count=torch.tensor(
            [0, 0],
            dtype=torch.long,
            device=device,
        ),
        dl_buffer=torch.tensor(
            [50.0, 0.0],
            device=device,
        ),
        wideband_cqi=torch.tensor(
            [10.0, 0.0],
            device=device,
        ),
        subband_cqi=torch.tensor(
            [
                [10.0, 12.0],
                [0.0, 0.0],
            ],
            device=device,
        ),
        max_precoder_cross_correlation=torch.tensor(
            [
                [0.2, 0.4],
                [0.0, 0.0],
            ],
            device=device,
        ),
        candidate_valid_mask=torch.tensor(
            [True, False],
            dtype=torch.bool,
            device=device,
        ),
    )

    result = build_1lds_state(
        features=features,
        config=config,
    )

    assert torch.any(
        result.ue_feature_segments[
            0
        ] != 0.0
    )

    torch.testing.assert_close(
        result.ue_feature_segments[
            1
        ],
        torch.zeros(
            config.ue_feature_size,
            dtype=torch.float32,
            device=device,
        ),
    )

def test_1lds_paper_state_dimensions():
    device = "cuda:0"

    config = OneLDSStateConfig(
        throughput_normalization_bps=100.0e6,
        buffer_normalization=1.0e6,
        subband_cqi_normalization=15.0,
        num_candidates=10,
        num_rbgs=18,
        max_rank=2,
    )

    assert config.ue_feature_size == 41
    assert config.state_size == 410
    assert config.actor_output_size == 198

    batch_size = 2
    num_cells = 3

    ue_shape = (
        batch_size,
        num_cells,
        10,
    )

    rbg_shape = (
        batch_size,
        num_cells,
        10,
        18,
    )

    features = OneLDSRawFeatures(
        past_average_throughput=torch.full(
            ue_shape,
            10.0e6,
            device=device,
        ),
        rank=torch.ones(
            ue_shape,
            dtype=torch.long,
            device=device,
        ),
        allocated_rbg_count=torch.zeros(
            ue_shape,
            dtype=torch.long,
            device=device,
        ),
        dl_buffer=torch.full(
            ue_shape,
            1000.0,
            device=device,
        ),
        wideband_cqi=torch.full(
            ue_shape,
            10.0,
            device=device,
        ),
        subband_cqi=torch.full(
            rbg_shape,
            10.0,
            device=device,
        ),
        max_precoder_cross_correlation=torch.zeros(
            rbg_shape,
            device=device,
        ),
        candidate_valid_mask=torch.ones(
            ue_shape,
            dtype=torch.bool,
            device=device,
        ),
    )

    result = build_1lds_state(
        features=features,
        config=config,
    )

    assert result.ue_feature_segments.shape == (
        2,
        3,
        10,
        41,
    )

    assert result.state.shape == (
        2,
        3,
        410,
    )

