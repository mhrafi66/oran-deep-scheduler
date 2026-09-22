from __future__ import annotations

import torch

from oran_scheduler.rl.ppo_sionna_chunked import (
    CellChunkedSionnaPPOInputProvider,
    ChunkedSionnaPPOConfig,
    build_chunked_sionna_ppo_context,
    build_chunked_state_managers,
)


def main() -> None:

    device = torch.device(
        "cuda:0"
    )

    torch.cuda.set_device(
        device
    )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    print("=" * 76)
    print(
        "WAVE-4 CHUNKED TEMPORAL RADIO PROVIDER SMOKE"
    )
    print("=" * 76)

    print(
        f"GPU: {torch.cuda.get_device_name(device)}"
    )

    context = (
        build_chunked_sionna_ppo_context(
            config=(
                ChunkedSionnaPPOConfig(
                    num_training_cells=2,

                    #
                    # Smaller population for capability
                    # validation only.
                    #
                    num_ut_per_sector=4,

                    num_rbs=18,

                    num_rbgs=18,

                    subcarriers_per_rb=12,

                    ue_microbatch_size=2,

                    topology_seed=42,

                    association_channel_seed=1000,

                    mimo_channel_seed=2000,

                    device="cuda:0",

                    ut_speed_kmh=30.0,

                    temporal_radio_mode=(
                        "velocity_window"
                    ),

                    temporal_window_ttis=4,

                    tti_duration_s=0.001,

                    temporal_max_displacement_m=(
                        20.0
                    ),
                )
            )
        )
    )

    print(
        "Global UEs:          "
        f"{context.num_global_ues}"
    )

    print(
        "Cells:               "
        f"{context.num_cells}"
    )

    print(
        "Selected streams:    "
        f"{context.selected_cell_indices}"
    )

    print(
        "Temporal mode:       "
        f"{context.config.temporal_radio_mode}"
    )

    print(
        "Temporal window:     "
        f"{context.config.temporal_window_ttis} TTIs"
    )

    print(
        "UE speed:            "
        f"{context.config.ut_speed_kmh:.1f} km/h"
    )

    provider = (
        CellChunkedSionnaPPOInputProvider(
            context=context
        )
    )

    #
    # TTI 0
    #
    tti0 = provider(
        0,
        0,
    )

    ids0 = (
        tti0
        .observation
        .serving_global_ue_indices
        .clone()
    )

    td0 = (
        tti0
        .observation
        .td_instantaneous_rate_bps
        .clone()
    )

    cqi0 = (
        tti0
        .observation
        .wideband_cqi
        .clone()
    )

    #
    # Exercise PF-TDS + candidate physical
    # compaction on the temporal H.
    #
    managers = (
        build_chunked_state_managers(
            context=context,

            initial_average_throughput_bps=(
                1.0e6
            ),

            throughput_forgetting_factor=(
                0.99
            ),

            num_candidates=10,
        )
    )

    prepared0 = (
        managers[
            0
        ]
        .prepare_tti(
            tti_index=0,

            observation=(
                tti0.observation
            ),
        )
    )

    physical0 = (
        tti0
        .physical_inputs_builder(
            prepared0
        )
    )

    print(
        "TTI 0 serving UEs:   "
        f"{ids0.numel()}"
    )

    print(
        "Candidate H shape:   "
        f"{tuple(physical0.h_freq.shape)}"
    )

    if int(
        physical0
        .h_freq
        .shape[3]
    ) != 21:
        raise RuntimeError(
            "Temporal candidate PHY lost "
            "one or more BS links."
        )

    if int(
        physical0
        .h_freq
        .shape[-2]
    ) != 1:
        raise RuntimeError(
            "Temporal provider did not return "
            "one TTI-shaped channel slice."
        )

    #
    # TTI 1 from the SAME temporal window.
    #
    tti1 = provider(
        1,
        0,
    )

    ids1 = (
        tti1
        .observation
        .serving_global_ue_indices
    )

    td1 = (
        tti1
        .observation
        .td_instantaneous_rate_bps
    )

    cqi1 = (
        tti1
        .observation
        .wideband_cqi
    )

    if not torch.equal(
        ids0,
        ids1,
    ):
        raise RuntimeError(
            "Fixed-association temporal mode "
            "changed serving UE identities."
        )

    if not (
        torch.isfinite(td0).all()
        and torch.isfinite(td1).all()
        and torch.isfinite(cqi0).all()
        and torch.isfinite(cqi1).all()
    ):
        raise RuntimeError(
            "Temporal provider produced "
            "non-finite scheduler observations."
        )

    td_change = (
        torch.mean(
            torch.abs(
                td1
                - td0
            )
        )
        .item()
    )

    cqi_change = (
        torch.mean(
            torch.abs(
                cqi1
                - cqi0
            )
        )
        .item()
    )

    print(
        "Mean |TD rate1-rate0|: "
        f"{td_change:.6f} bps"
    )

    print(
        "Mean |CQI1-CQI0|:       "
        f"{cqi_change:.6f}"
    )

    temporal_runtime = (
        context.temporal_channel_runtime
    )

    if temporal_runtime is None:
        raise RuntimeError(
            "Temporal runtime was not created."
        )

    print(
        "Temporal generations:   "
        f"{temporal_runtime.num_generations}"
    )

    peak_mib = (
        torch.cuda.max_memory_allocated(
            device
        )
        / (1024.0 ** 2)
    )

    print(
        "Peak GPU allocated:     "
        f"{peak_mib:.2f} MiB"
    )

    print()
    print(
        "WAVE4_CHUNKED_TEMPORAL_PROVIDER_PASS"
    )


if __name__ == "__main__":
    main()
