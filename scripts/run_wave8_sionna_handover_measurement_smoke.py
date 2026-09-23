from __future__ import annotations

import torch

from oran_scheduler.rl.ppo_sionna_chunked import (
    ChunkedSionnaPPOConfig,
    build_chunked_sionna_ppo_context,
)

from oran_scheduler.simulator.sionna_handover_measurement import (
    SionnaHandoverMeasurementConfig,
    SionnaPathlossHandoverMeasurementProvider,
)


DEVICE = torch.device(
    "cuda:0"
)


def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    torch.cuda.set_device(
        DEVICE
    )

    torch.cuda.reset_peak_memory_stats(
        DEVICE
    )

    print("=" * 76)
    print(
        "WAVE-8 REAL SIONNA HANDOVER "
        "MEASUREMENT SMOKE"
    )
    print("=" * 76)

    context = (
        build_chunked_sionna_ppo_context(
            config=(
                ChunkedSionnaPPOConfig(
                    num_training_cells=2,

                    num_ut_per_sector=2,

                    num_rbs=18,

                    num_rbgs=18,

                    subcarriers_per_rb=12,

                    ue_microbatch_size=1,

                    identity_stable_ue_channel_rng=True,

                    topology_seed=42,

                    association_channel_seed=1000,

                    mimo_channel_seed=2000,

                    device="cuda:0",

                    ut_speed_kmh=30.0,

                    temporal_radio_mode=(
                        "velocity_window"
                    ),

                    temporal_window_ttis=4,

                    tti_duration_s=0.5e-3,

                    temporal_max_displacement_m=20.0,
                )
            )
        )
    )

    cohort = torch.cat(
        context.global_ue_indices_by_stream,
        dim=0,
    )

    provider = (
        SionnaPathlossHandoverMeasurementProvider(
            initial_topology=(
                context.topology
            ),

            topology_config=(
                context.topology_config
            ),

            global_ue_indices=(
                cohort
            ),

            config=(
                SionnaHandoverMeasurementConfig(
                    num_rbs=18,

                    subcarriers_per_rb=12,

                    tti_duration_s=0.5e-3,

                    channel_seed=9100,

                    max_horizontal_displacement_m=(
                        20.0
                    ),

                    device="cuda:0",
                )
            ),
        )
    )

    measurement_0 = provider(
        0
    )

    measurement_0_again = provider(
        0
    )

    measurement_1 = provider(
        1
    )

    if tuple(
        measurement_0.shape
    ) != (
        int(
            cohort.numel()
        ),
        context.num_cells,
    ):
        raise RuntimeError(
            "Unexpected measurement shape."
        )

    if not torch.isfinite(
        measurement_0
    ).all():
        raise RuntimeError(
            "TTI-0 measurement is non-finite."
        )

    if torch.any(
        measurement_0 <= 0.0
    ):
        raise RuntimeError(
            "TTI-0 measurement is non-positive."
        )

    if not torch.equal(
        measurement_0,
        measurement_0_again,
    ):
        raise RuntimeError(
            "Same-TTI handover measurement "
            "is not idempotent."
        )

    strongest_0 = torch.argmax(
        measurement_0,
        dim=1,
    )

    strongest_1 = torch.argmax(
        measurement_1,
        dim=1,
    )

    changed = int(
        torch.count_nonzero(
            strongest_0
            != strongest_1
        ).item()
    )

    print(
        "Selected scheduling cells: "
        f"{context.selected_cell_indices}"
    )

    print(
        "Closed cohort size: "
        f"{int(cohort.numel())}"
    )

    print(
        "Measurement shape: "
        f"{tuple(measurement_0.shape)}"
    )

    print(
        "Strongest-cell changes over "
        "one 0.5-ms step: "
        f"{changed}"
    )

    print(
        "TTI-0 strongest BSs: "
        f"{strongest_0.detach().cpu().tolist()}"
    )

    print(
        "TTI-1 strongest BSs: "
        f"{strongest_1.detach().cpu().tolist()}"
    )

    torch.cuda.synchronize(
        DEVICE
    )

    peak_mib = (
        torch.cuda.max_memory_allocated(
            DEVICE
        )
        / (
            1024.0
            ** 2
        )
    )

    print(
        "Peak allocated GPU: "
        f"{peak_mib:.1f} MiB"
    )

    print("=" * 76)

    print(
        "WAVE8_SIONNA_HANDOVER_MEASUREMENT_PASS"
    )


if __name__ == "__main__":
    main()
