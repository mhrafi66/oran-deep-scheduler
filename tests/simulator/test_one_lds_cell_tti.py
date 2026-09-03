import pytest
import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
    OneLDSCellTTIStateManager,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def build_observation(
    *,
    device: torch.device,
    instantaneous_rate: torch.Tensor,
) -> OneLDSCellTTIObservation:
    num_ues = 3

    num_rbgs = 2

    directions = torch.zeros(
        (
            num_ues,
            num_rbgs,
            1,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        :,
        :,
        0,
        0,
    ] = 1.0

    return OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            torch.tensor(
                [
                    100,
                    101,
                    102,
                ],
                dtype=torch.long,
                device=device,
            )
        ),
        serving_ue_valid_mask=(
            torch.ones(
                3,
                dtype=torch.bool,
                device=device,
            )
        ),
        td_instantaneous_rate_bps=(
            instantaneous_rate
        ),
        rank=torch.tensor(
            [
                1,
                2,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
        dl_buffer=torch.tensor(
            [
                10.0,
                20.0,
                30.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        wideband_cqi=torch.tensor(
            [
                10.0,
                11.0,
                12.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        subband_cqi=torch.tensor(
            [
                [
                    10.0,
                    9.0,
                ],
                [
                    11.0,
                    10.0,
                ],
                [
                    12.0,
                    11.0,
                ],
            ],
            dtype=torch.float32,
            device=device,
        ),
        precoder_directions=(
            directions
        ),
    )


def test_prepare_tti_builds_pf_candidate_inputs():
    device = preferred_device()

    manager = OneLDSCellTTIStateManager(
        initial_average_throughput_bps=(
            torch.ones(
                3,
                dtype=torch.float32,
                device=device,
            )
        ),
        serving_global_ue_indices=torch.tensor(
            [
                100,
                101,
                102,
            ],
            dtype=torch.long,
            device=device,
        ),
        serving_ue_valid_mask=torch.ones(
            3,
            dtype=torch.bool,
            device=device,
        ),
        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),
        throughput_forgetting_factor=0.5,
    )

    observation = build_observation(
        device=device,
        instantaneous_rate=torch.tensor(
            [
                10.0,
                8.0,
                6.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    prepared = manager.prepare_tti(
        tti_index=0,
        observation=observation,
    )

    torch.testing.assert_close(
        prepared.candidate_serving_indices,
        torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    torch.testing.assert_close(
        prepared.candidate_global_ue_indices,
        torch.tensor(
            [
                100,
                101,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    assert tuple(
        prepared
        .decision_inputs
        .past_average_throughput
        .shape
    ) == (
        2,
    )

    assert tuple(
        prepared
        .decision_inputs
        .subband_cqi
        .shape
    ) == (
        2,
        2,
    )

    assert tuple(
        prepared
        .decision_inputs
        .candidate_precoder_directions
        .shape
    ) == (
        2,
        2,
        1,
        2,
    )

    assert (
        prepared
        .decision_inputs
        .past_average_throughput
        .device
        == device
    )


def test_next_tti_uses_updated_history_and_can_change_candidates():
    device = preferred_device()

    manager = OneLDSCellTTIStateManager(
        initial_average_throughput_bps=(
            torch.ones(
                3,
                dtype=torch.float32,
                device=device,
            )
        ),
        serving_global_ue_indices=torch.tensor(
            [
                100,
                101,
                102,
            ],
            dtype=torch.long,
            device=device,
        ),
        serving_ue_valid_mask=torch.ones(
            3,
            dtype=torch.bool,
            device=device,
        ),
        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),
        throughput_forgetting_factor=0.5,
    )

    observation = build_observation(
        device=device,
        instantaneous_rate=torch.tensor(
            [
                10.0,
                8.0,
                6.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    tti_0 = manager.prepare_tti(
        tti_index=0,
        observation=observation,
    )

    torch.testing.assert_close(
        tti_0.candidate_serving_indices,
        torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    manager.complete_tti(
        candidate_delivered_rate_bps=(
            torch.tensor(
                [
                    100.0,
                    0.0,
                ],
                dtype=torch.float32,
                device=device,
            )
        )
    )

    expected_history = torch.tensor(
        [
            50.5,
            0.5,
            0.5,
        ],
        dtype=torch.float32,
        device=device,
    )

    torch.testing.assert_close(
        manager.current_average_throughput_bps,
        expected_history,
    )

    tti_1 = manager.prepare_tti(
        tti_index=1,
        observation=observation,
    )

    torch.testing.assert_close(
        tti_1.candidate_serving_indices,
        torch.tensor(
            [
                1,
                2,
            ],
            dtype=torch.long,
            device=device,
        ),
    )


def test_prepare_tti_requires_previous_tti_completion():
    device = preferred_device()

    manager = OneLDSCellTTIStateManager(
        initial_average_throughput_bps=(
            torch.ones(
                3,
                device=device,
            )
        ),
        serving_global_ue_indices=torch.tensor(
            [
                100,
                101,
                102,
            ],
            device=device,
        ),
        serving_ue_valid_mask=torch.ones(
            3,
            dtype=torch.bool,
            device=device,
        ),
        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),
        throughput_forgetting_factor=0.5,
    )

    observation = build_observation(
        device=device,
        instantaneous_rate=torch.tensor(
            [
                10.0,
                8.0,
                6.0,
            ],
            device=device,
        ),
    )

    manager.prepare_tti(
        tti_index=0,
        observation=observation,
    )

    with pytest.raises(
        RuntimeError
    ):
        manager.prepare_tti(
            tti_index=1,
            observation=observation,
        )

def test_state_manager_rejects_changed_serving_identity_layout():
    device = preferred_device()

    manager = OneLDSCellTTIStateManager(
        initial_average_throughput_bps=(
            torch.ones(
                3,
                device=device,
            )
        ),
        serving_global_ue_indices=torch.tensor(
            [
                100,
                101,
                102,
            ],
            device=device,
        ),
        serving_ue_valid_mask=torch.ones(
            3,
            dtype=torch.bool,
            device=device,
        ),
        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),
        throughput_forgetting_factor=0.5,
    )

    observation = build_observation(
        device=device,
        instantaneous_rate=torch.tensor(
            [
                10.0,
                8.0,
                6.0,
            ],
            device=device,
        ),
    )

    broken = OneLDSCellTTIObservation(
        serving_global_ue_indices=torch.tensor(
            [
                100,
                999,
                102,
            ],
            dtype=torch.long,
            device=device,
        ),
        serving_ue_valid_mask=(
            observation.serving_ue_valid_mask
        ),
        td_instantaneous_rate_bps=(
            observation
            .td_instantaneous_rate_bps
        ),
        rank=observation.rank,
        dl_buffer=observation.dl_buffer,
        wideband_cqi=(
            observation.wideband_cqi
        ),
        subband_cqi=(
            observation.subband_cqi
        ),
        precoder_directions=(
            observation.precoder_directions
        ),
    )

    with pytest.raises(
        ValueError
    ):
        manager.prepare_tti(
            tti_index=0,
            observation=broken,
        )


def test_paper_shaped_candidate_inputs_stay_on_gpu():
    device = preferred_device()

    num_serving_ues = 12

    num_rbgs = 18

    num_modes = 2

    num_tx_ant = 8

    global_ids = torch.arange(
        100,
        100 + num_serving_ues,
        dtype=torch.long,
        device=device,
    )

    valid_mask = torch.ones(
        num_serving_ues,
        dtype=torch.bool,
        device=device,
    )

    manager = OneLDSCellTTIStateManager(
        initial_average_throughput_bps=(
            torch.ones(
                num_serving_ues,
                dtype=torch.float32,
                device=device,
            )
        ),
        serving_global_ue_indices=(
            global_ids
        ),
        serving_ue_valid_mask=(
            valid_mask
        ),
        tds_config=PFTimeDomainConfig(
            num_candidates=10,
        ),
        throughput_forgetting_factor=0.9,
    )

    observation = OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            global_ids
        ),
        serving_ue_valid_mask=(
            valid_mask
        ),
        td_instantaneous_rate_bps=(
            torch.rand(
                num_serving_ues,
                device=device,
            )
            + 0.1
        ),
        rank=torch.ones(
            num_serving_ues,
            dtype=torch.long,
            device=device,
        ),
        dl_buffer=torch.ones(
            num_serving_ues,
            device=device,
        ),
        wideband_cqi=torch.full(
            (
                num_serving_ues,
            ),
            10.0,
            device=device,
        ),
        subband_cqi=torch.full(
            (
                num_serving_ues,
                num_rbgs,
            ),
            10.0,
            device=device,
        ),
        precoder_directions=(
            torch.zeros(
                (
                    num_serving_ues,
                    num_rbgs,
                    num_modes,
                    num_tx_ant,
                ),
                dtype=torch.complex64,
                device=device,
            )
        ),
    )

    prepared = manager.prepare_tti(
        tti_index=0,
        observation=observation,
    )

    assert tuple(
        prepared
        .decision_inputs
        .past_average_throughput
        .shape
    ) == (
        10,
    )

    assert tuple(
        prepared
        .decision_inputs
        .subband_cqi
        .shape
    ) == (
        10,
        18,
    )

    assert tuple(
        prepared
        .decision_inputs
        .candidate_precoder_directions
        .shape
    ) == (
        10,
        18,
        2,
        8,
    )

    assert (
        prepared
        .decision_inputs
        .subband_cqi
        .device
        == device
    )

def test_prepare_tti_uses_separate_tds_eligibility_mask():
    device = preferred_device()

    manager = OneLDSCellTTIStateManager(
        initial_average_throughput_bps=(
            torch.ones(
                3,
                dtype=torch.float32,
                device=device,
            )
        ),
        serving_global_ue_indices=torch.tensor(
            [
                100,
                101,
                102,
            ],
            dtype=torch.long,
            device=device,
        ),
        serving_ue_valid_mask=torch.ones(
            3,
            dtype=torch.bool,
            device=device,
        ),
        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),
        throughput_forgetting_factor=0.5,
    )

    observation = build_observation(
        device=device,
        instantaneous_rate=torch.tensor(
            [
                100.0,
                10.0,
                8.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    #
    # UE 0 has by far the highest PF metric, but
    # traffic policy says it is temporarily
    # ineligible.
    #
    prepared = manager.prepare_tti(
        tti_index=0,
        observation=observation,
        tds_eligible_mask=torch.tensor(
            [
                False,
                True,
                True,
            ],
            dtype=torch.bool,
            device=device,
        ),
    )

    torch.testing.assert_close(
        prepared.candidate_serving_indices,
        torch.tensor(
            [
                1,
                2,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    torch.testing.assert_close(
        prepared.candidate_global_ue_indices,
        torch.tensor(
            [
                101,
                102,
            ],
            dtype=torch.long,
            device=device,
        ),
    )


