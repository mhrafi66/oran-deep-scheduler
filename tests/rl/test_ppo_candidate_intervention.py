import pytest
import torch

from oran_scheduler.rl.ppo_candidate_intervention import (
    CandidateInterventionConfig,
    CandidateInterventionInputProvider,
    build_candidate_intervention_observation,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
)


def _observation(
    *,
    td_value: float,
    rank_value: int,
    cqi_value: float,
    precoder_value: float,
) -> OneLDSCellTTIObservation:

    return OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            torch.tensor(
                [10, 11],
                dtype=torch.long,
            )
        ),

        serving_ue_valid_mask=(
            torch.tensor(
                [True, True],
                dtype=torch.bool,
            )
        ),

        td_instantaneous_rate_bps=(
            torch.tensor(
                [
                    td_value,
                    td_value + 1.0,
                ],
                dtype=torch.float32,
            )
        ),

        rank=(
            torch.tensor(
                [
                    rank_value,
                    rank_value,
                ],
                dtype=torch.long,
            )
        ),

        dl_buffer=(
            torch.tensor(
                [100.0, 200.0],
                dtype=torch.float32,
            )
        ),

        wideband_cqi=(
            torch.tensor(
                [
                    cqi_value,
                    cqi_value + 1.0,
                ],
                dtype=torch.float32,
            )
        ),

        subband_cqi=(
            torch.full(
                (2, 3),
                fill_value=cqi_value,
                dtype=torch.float32,
            )
        ),

        precoder_directions=(
            torch.full(
                (2, 3, 1, 2),
                fill_value=complex(
                    precoder_value,
                    0.0,
                ),
                dtype=torch.complex64,
            )
        ),
    )


def test_native_uses_all_stressed_radio_fields() -> None:
    fresh = _observation(
        td_value=100.0,
        rank_value=1,
        cqi_value=10.0,
        precoder_value=1.0,
    )

    stressed = _observation(
        td_value=500.0,
        rank_value=2,
        cqi_value=20.0,
        precoder_value=2.0,
    )

    output = (
        build_candidate_intervention_observation(
            fresh=fresh,
            stressed=stressed,
            mode="native",
        )
    )

    torch.testing.assert_close(
        output.td_instantaneous_rate_bps,
        stressed.td_instantaneous_rate_bps,
    )

    torch.testing.assert_close(
        output.rank,
        stressed.rank,
    )

    torch.testing.assert_close(
        output.wideband_cqi,
        stressed.wideband_cqi,
    )

    torch.testing.assert_close(
        output.subband_cqi,
        stressed.subband_cqi,
    )

    torch.testing.assert_close(
        output.precoder_directions,
        stressed.precoder_directions,
    )


def test_fresh_candidates_only_replaces_td_rate() -> None:
    fresh = _observation(
        td_value=100.0,
        rank_value=1,
        cqi_value=10.0,
        precoder_value=1.0,
    )

    stressed = _observation(
        td_value=500.0,
        rank_value=2,
        cqi_value=20.0,
        precoder_value=2.0,
    )

    output = (
        build_candidate_intervention_observation(
            fresh=fresh,
            stressed=stressed,
            mode="fresh_candidates",
        )
    )

    torch.testing.assert_close(
        output.td_instantaneous_rate_bps,
        fresh.td_instantaneous_rate_bps,
    )

    torch.testing.assert_close(
        output.rank,
        stressed.rank,
    )

    torch.testing.assert_close(
        output.wideband_cqi,
        stressed.wideband_cqi,
    )

    torch.testing.assert_close(
        output.subband_cqi,
        stressed.subband_cqi,
    )

    torch.testing.assert_close(
        output.precoder_directions,
        stressed.precoder_directions,
    )


def test_fresh_features_keeps_stressed_candidate_rate() -> None:
    fresh = _observation(
        td_value=100.0,
        rank_value=1,
        cqi_value=10.0,
        precoder_value=1.0,
    )

    stressed = _observation(
        td_value=500.0,
        rank_value=2,
        cqi_value=20.0,
        precoder_value=2.0,
    )

    output = (
        build_candidate_intervention_observation(
            fresh=fresh,
            stressed=stressed,
            mode="fresh_features",
        )
    )

    torch.testing.assert_close(
        output.td_instantaneous_rate_bps,
        stressed.td_instantaneous_rate_bps,
    )

    torch.testing.assert_close(
        output.rank,
        fresh.rank,
    )

    torch.testing.assert_close(
        output.wideband_cqi,
        fresh.wideband_cqi,
    )

    torch.testing.assert_close(
        output.subband_cqi,
        fresh.subband_cqi,
    )

    torch.testing.assert_close(
        output.precoder_directions,
        fresh.precoder_directions,
    )


def test_fresh_both_uses_fresh_radio_information() -> None:
    fresh = _observation(
        td_value=100.0,
        rank_value=1,
        cqi_value=10.0,
        precoder_value=1.0,
    )

    stressed = _observation(
        td_value=500.0,
        rank_value=2,
        cqi_value=20.0,
        precoder_value=2.0,
    )

    output = (
        build_candidate_intervention_observation(
            fresh=fresh,
            stressed=stressed,
            mode="fresh_both",
        )
    )

    torch.testing.assert_close(
        output.td_instantaneous_rate_bps,
        fresh.td_instantaneous_rate_bps,
    )

    torch.testing.assert_close(
        output.rank,
        fresh.rank,
    )

    torch.testing.assert_close(
        output.wideband_cqi,
        fresh.wideband_cqi,
    )

    torch.testing.assert_close(
        output.subband_cqi,
        fresh.subband_cqi,
    )

    torch.testing.assert_close(
        output.precoder_directions,
        fresh.precoder_directions,
    )


def test_serving_identity_mismatch_is_rejected() -> None:
    fresh = _observation(
        td_value=100.0,
        rank_value=1,
        cqi_value=10.0,
        precoder_value=1.0,
    )

    stressed = _observation(
        td_value=500.0,
        rank_value=2,
        cqi_value=20.0,
        precoder_value=2.0,
    )

    stressed = OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            torch.tensor(
                [99, 100],
                dtype=torch.long,
            )
        ),
        serving_ue_valid_mask=(
            stressed.serving_ue_valid_mask
        ),
        td_instantaneous_rate_bps=(
            stressed.td_instantaneous_rate_bps
        ),
        rank=stressed.rank,
        dl_buffer=stressed.dl_buffer,
        wideband_cqi=(
            stressed.wideband_cqi
        ),
        subband_cqi=(
            stressed.subband_cqi
        ),
        precoder_directions=(
            stressed.precoder_directions
        ),
    )

    with pytest.raises(
        ValueError,
        match="different serving UE identities",
    ):
        build_candidate_intervention_observation(
            fresh=fresh,
            stressed=stressed,
            mode="native",
        )


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported candidate intervention mode",
    ):
        CandidateInterventionConfig(
            mode="bad_mode",  # type: ignore[arg-type]
        )


def test_provider_preserves_physical_builder_and_arrivals() -> None:
    fresh_observation = _observation(
        td_value=100.0,
        rank_value=1,
        cqi_value=10.0,
        precoder_value=1.0,
    )

    stressed_observation = _observation(
        td_value=500.0,
        rank_value=2,
        cqi_value=20.0,
        precoder_value=2.0,
    )

    arrivals = torch.tensor(
        [1, 2],
        dtype=torch.long,
    )

    def fresh_builder(prepared):
        return prepared

    def stressed_builder(prepared):
        return prepared

    call_order: list[str] = []

    def fresh_provider(
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:
        call_order.append(
            "fresh"
        )

        return PPOTrainingTTIInputs(
            observation=(
                fresh_observation
            ),
            physical_inputs_builder=(
                fresh_builder
            ),
            packet_arrivals=arrivals,
        )

    def stressed_provider(
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:
        call_order.append(
            "stressed"
        )

        return PPOTrainingTTIInputs(
            observation=(
                stressed_observation
            ),
            physical_inputs_builder=(
                stressed_builder
            ),
            packet_arrivals=arrivals,
        )

    provider = CandidateInterventionInputProvider(
        fresh_provider=(
            fresh_provider
        ),
        stressed_provider=(
            stressed_provider
        ),
        config=CandidateInterventionConfig(
            mode="fresh_candidates",
        ),
    )

    output = provider(
        7,
        3,
    )

    #
    # Temporal stress provider must advance first.
    #
    assert call_order == [
        "stressed",
        "fresh",
    ]

    #
    # Candidate selection gets fresh TD rate.
    #
    torch.testing.assert_close(
        output
        .observation
        .td_instantaneous_rate_bps,
        fresh_observation
        .td_instantaneous_rate_bps,
    )

    #
    # PPO radio feature remains stressed.
    #
    torch.testing.assert_close(
        output
        .observation
        .wideband_cqi,
        stressed_observation
        .wideband_cqi,
    )

    #
    # Real PHY builder and arrivals must remain the
    # ones from the actual stressed evaluation path.
    #
    assert (
        output.physical_inputs_builder
        is stressed_builder
    )

    assert (
        output.packet_arrivals
        is arrivals
    )


def test_provider_is_idempotent_for_same_key() -> None:
    fresh_observation = _observation(
        td_value=100.0,
        rank_value=1,
        cqi_value=10.0,
        precoder_value=1.0,
    )

    stressed_observation = _observation(
        td_value=500.0,
        rank_value=2,
        cqi_value=20.0,
        precoder_value=2.0,
    )

    calls = {
        "fresh": 0,
        "stressed": 0,
    }

    def builder(prepared):
        return prepared

    def fresh_provider(
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:
        calls[
            "fresh"
        ] += 1

        return PPOTrainingTTIInputs(
            observation=(
                fresh_observation
            ),
            physical_inputs_builder=(
                builder
            ),
            packet_arrivals=None,
        )

    def stressed_provider(
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:
        calls[
            "stressed"
        ] += 1

        return PPOTrainingTTIInputs(
            observation=(
                stressed_observation
            ),
            physical_inputs_builder=(
                builder
            ),
            packet_arrivals=None,
        )

    provider = CandidateInterventionInputProvider(
        fresh_provider=(
            fresh_provider
        ),
        stressed_provider=(
            stressed_provider
        ),
        config=CandidateInterventionConfig(
            mode="fresh_both",
        ),
    )

    first = provider(
        2,
        4,
    )

    second = provider(
        2,
        4,
    )

    assert first is second

    assert calls == {
        "fresh": 1,
        "stressed": 1,
    }
