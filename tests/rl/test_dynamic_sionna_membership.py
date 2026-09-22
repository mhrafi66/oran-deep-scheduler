from types import SimpleNamespace

import pytest
import torch

import oran_scheduler.rl.ppo_sionna_chunked as chunked_module

from oran_scheduler.rl.ppo_sionna_chunked import (
    CellChunkedSionnaPPOInputProvider,
    ChunkedSionnaPPOConfig,
    _identity_stable_independent_ue_seed,
    _identity_stable_temporal_ue_seed,
)


def _seed_context():
    return SimpleNamespace(
        num_global_ues=420,

        config=SimpleNamespace(
            mimo_channel_seed=2000,
        ),
    )


def test_identity_stable_mode_requires_single_ue_microbatch():
    with pytest.raises(
        ValueError,
        match="ue_microbatch_size=1",
    ):
        ChunkedSionnaPPOConfig(
            ue_microbatch_size=2,

            identity_stable_ue_channel_rng=True,
        )


def test_independent_seed_is_global_ue_owned():
    context = _seed_context()

    first = (
        _identity_stable_independent_ue_seed(
            context=context,

            tti_index=4,

            global_ue_index=123,
        )
    )

    same = (
        _identity_stable_independent_ue_seed(
            context=context,

            tti_index=4,

            global_ue_index=123,
        )
    )

    other_ue = (
        _identity_stable_independent_ue_seed(
            context=context,

            tti_index=4,

            global_ue_index=124,
        )
    )

    next_tti = (
        _identity_stable_independent_ue_seed(
            context=context,

            tti_index=5,

            global_ue_index=123,
        )
    )

    assert first == same
    assert first != other_ue
    assert first != next_tti


def test_temporal_seed_is_global_ue_and_window_owned():
    context = _seed_context()

    first = (
        _identity_stable_temporal_ue_seed(
            context=context,

            window_index=3,

            global_ue_index=123,
        )
    )

    same = (
        _identity_stable_temporal_ue_seed(
            context=context,

            window_index=3,

            global_ue_index=123,
        )
    )

    other_ue = (
        _identity_stable_temporal_ue_seed(
            context=context,

            window_index=3,

            global_ue_index=124,
        )
    )

    other_window = (
        _identity_stable_temporal_ue_seed(
            context=context,

            window_index=4,

            global_ue_index=123,
        )
    )

    assert first == same
    assert first != other_ue
    assert first != other_window


def test_dynamic_provider_passes_current_membership(
    monkeypatch,
):
    calls = []

    context = SimpleNamespace(
        num_streams=2,

        config=SimpleNamespace(
            device=None,

            identity_stable_ue_channel_rng=True,
        ),
    )

    memberships = {
        (
            0,
            0,
        ): torch.tensor(
            [
                10,
                20,
            ],
            dtype=torch.long,
        ),

        (
            1,
            0,
        ): torch.tensor(
            [
                10,
            ],
            dtype=torch.long,
        ),
    }

    def membership_provider(
        tti_index,
        stream_index,
    ):
        return memberships[
            (
                tti_index,
                stream_index,
            )
        ]

    def fake_build(
        *,
        context,
        tti_index,
        stream_index,
        cell_global_ue_indices_override=None,
    ):
        del context

        assert (
            cell_global_ue_indices_override
            is not None
        )

        calls.append(
            (
                tti_index,
                stream_index,
                tuple(
                    cell_global_ue_indices_override
                    .tolist()
                ),
            )
        )

        return SimpleNamespace(
            observation=(
                tti_index,
                stream_index,
                tuple(
                    cell_global_ue_indices_override
                    .tolist()
                ),
            )
        )

    monkeypatch.setattr(
        chunked_module,
        "_build_cell_training_inputs",
        fake_build,
    )

    provider = (
        CellChunkedSionnaPPOInputProvider(
            context=context,

            global_ue_indices_provider=(
                membership_provider
            ),
        )
    )

    first = provider(
        0,
        0,
    )

    assert first.observation == (
        0,
        0,
        (
            10,
            20,
        ),
    )

    #
    # Same TTI/stream is cached.
    #
    again = provider(
        0,
        0,
    )

    assert again is first

    assert calls == [
        (
            0,
            0,
            (
                10,
                20,
            ),
        )
    ]

    #
    # Next TTI gets new association membership.
    #
    second = provider(
        1,
        0,
    )

    assert second.observation == (
        1,
        0,
        (
            10,
        ),
    )

    assert calls == [
        (
            0,
            0,
            (
                10,
                20,
            ),
        ),
        (
            1,
            0,
            (
                10,
            ),
        ),
    ]


def test_dynamic_provider_rejects_legacy_rng():
    context = SimpleNamespace(
        num_streams=1,

        config=SimpleNamespace(
            identity_stable_ue_channel_rng=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match="identity-stable",
    ):
        CellChunkedSionnaPPOInputProvider(
            context=context,

            global_ue_indices_provider=(
                lambda tti, stream: torch.tensor(
                    [
                        10,
                    ],
                    dtype=torch.long,
                )
            ),
        )
