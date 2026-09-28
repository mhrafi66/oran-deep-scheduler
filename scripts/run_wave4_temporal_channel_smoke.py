from __future__ import annotations

import argparse

import torch

from oran_scheduler.simulator.channel import (
    ChannelConfig,
)
from oran_scheduler.simulator.mobility import (
    MobilityConfig,
    MobilityEngine,
)
from oran_scheduler.simulator.temporal_channel import (
    VelocityWindowFrequencyChannelRuntime,
    normalized_temporal_correlation,
)
from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
)


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--num-ttis",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--tti-duration-ms",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def _selected_lags(
    num_ttis: int,
) -> list[int]:

    candidates = [
        0,
        1,
        2,
        4,
        8,
        num_ttis - 1,
    ]

    return sorted(
        {
            value
            for value in candidates
            if 0 <= value < num_ttis
        }
    )


def main() -> None:

    args = _parse_args()

    if args.num_ttis < 2:
        raise ValueError(
            "num_ttis must be at least 2."
        )

    tti_duration_s = (
        args.tti_duration_ms
        / 1000.0
    )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA smoke requested but CUDA "
            "is unavailable."
        )

    if device.type == "cuda":
        torch.cuda.set_device(
            device
        )

        torch.cuda.reset_peak_memory_stats(
            device
        )

    speeds_kmh = (
        0.0,
        3.0,
        30.0,
        120.0,
    )

    print("=" * 76)
    print(
        "WAVE-4 TEMPORAL SIONNA CHANNEL SMOKE"
    )
    print("=" * 76)

    print(
        f"Device:             {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:                "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"Temporal samples:   {args.num_ttis}"
    )

    print(
        "Sample spacing:     "
        f"{args.tti_duration_ms:.3f} ms"
    )

    print(
        "Model:              "
        "short-window velocity-driven UMa"
    )

    print(
        "Association:        fixed"
    )

    print(
        "Geometry in radio window: fixed"
    )

    print()

    moving_correlations = []

    for speed_kmh in speeds_kmh:

        topology_config = (
            TopologyConfig(
                batch_size=1,

                num_rings=1,

                num_ut_per_sector=1,

                scenario="uma",

                isd_m=200.0,

                bs_height_m=25.0,

                ut_height_m=1.5,

                ut_speed_kmh=(
                    speed_kmh
                ),

                seed=42,

                precision="single",

                device=str(device),
            )
        )

        topology = generate_topology(
            topology_config
        )

        #
        # Geometry evolution is validated separately
        # from the short-window Doppler radio.
        #
        mobility = MobilityEngine(
            initial_topology=topology,

            config=MobilityConfig(
                tti_duration_s=(
                    tti_duration_s
                ),

                trajectory_mode=(
                    "constant_velocity"
                    if speed_kmh > 0.0
                    else "static"
                ),

                association_mode="fixed",

                max_horizontal_displacement_m=(
                    100.0
                ),
            ),
        )

        final_mobility = (
            mobility.advance_to(
                args.num_ttis - 1
            )
        )

        channel_config = (
            ChannelConfig(
                carrier_frequency_hz=4.0e9,

                subcarrier_spacing_hz=30.0e3,

                #
                # Tiny radio for capability validation.
                # Not paper-scale scheduler evaluation.
                #
                num_rbs=2,

                subcarriers_per_rb=12,

                num_ofdm_symbols=1,

                antenna_mode="sanity",

                direction="downlink",

                o2i_model="low",

                enable_pathloss=True,

                enable_shadow_fading=True,

                precision="single",

                device=str(device),

                seed=9000,
            )
        )

        runtime = (
            VelocityWindowFrequencyChannelRuntime(
                config=channel_config
            )
        )

        window = runtime.generate_window(
            topology=topology,

            seed=9000,

            batch_size=1,

            num_ttis=(
                args.num_ttis
            ),

            tti_duration_s=(
                tti_duration_s
            ),
        )

        correlation = (
            normalized_temporal_correlation(
                window.h_freq
            )
        )

        if (
            not torch.isfinite(
                correlation
            ).all()
        ):
            raise RuntimeError(
                "Non-finite temporal correlation."
            )

        print(
            f"Speed {speed_kmh:6.1f} km/h"
        )

        print(
            "  mean UE displacement: "
            f"{final_mobility.mean_displacement_m:.6f} m"
        )

        print(
            "  max UE displacement:  "
            f"{final_mobility.max_displacement_m:.6f} m"
        )

        for lag in _selected_lags(
            args.num_ttis
        ):
            print(
                f"  corr lag {lag:2d}:          "
                f"{correlation[lag].item():.6f}"
            )

        print()

        if speed_kmh == 0.0:
            if not torch.all(
                correlation
                > 0.9999
            ):
                raise RuntimeError(
                    "Zero-speed UMa channel was "
                    "not temporally static inside "
                    "one channel realization."
                )

        else:
            moving_correlations.append(
                correlation
            )

    #
    # At least one moving case must show actual
    # temporal channel evolution.
    #
    if not any(
        torch.any(
            correlation[1:] < 0.999
        ).item()

        for correlation
        in moving_correlations
    ):
        raise RuntimeError(
            "UE velocity did not produce detectable "
            "temporal Sionna channel evolution."
        )

    if device.type == "cuda":
        peak_mib = (
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024.0 ** 2)
        )

        print(
            "Peak GPU allocated: "
            f"{peak_mib:.2f} MiB"
        )

    print()
    print(
        "WAVE4_TEMPORAL_CHANNEL_SMOKE_PASS"
    )


if __name__ == "__main__":
    main()
