import pytest
import torch

from oran_scheduler.simulator.channel import (
    ChannelConfig,
)
from oran_scheduler.simulator.temporal_channel import (
    VelocityWindowFrequencyChannelRuntime,
)
from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
    subset_topology_ues,
)


def preferred_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def build_small_temporal_case():

    device = preferred_device()

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,
        num_ut_per_sector=1,
        device=str(
            device
        ),
        seed=321,
    )

    topology = generate_topology(
        topology_config
    )

    indices = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.long,
        device=device,
    )

    topology = subset_topology_ues(
        topology=topology,
        global_ue_indices=indices,
    )

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,

        subcarrier_spacing_hz=30.0e3,

        #
        # Tiny frequency dimension:
        # this test validates ordering/equivalence,
        # not paper-scale GPU memory.
        #
        num_rbs=1,

        subcarriers_per_rb=12,

        num_ofdm_symbols=1,

        antenna_mode="sanity",

        direction="downlink",

        o2i_model="low",

        enable_pathloss=True,

        enable_shadow_fading=True,

        precision="single",

        device=str(
            device
        ),

        seed=12345,
    )

    return (
        topology,
        channel_config,
    )


def test_single_tti_conversion_matches_full_window_slice():
    (
        topology,
        channel_config,
    ) = build_small_temporal_case()

    seed = 8675309

    num_ttis = 4

    tti_duration_s = 0.0005

    full_runtime = (
        VelocityWindowFrequencyChannelRuntime(
            config=channel_config,
        )
    )

    sliced_runtime = (
        VelocityWindowFrequencyChannelRuntime(
            config=channel_config,
        )
    )

    full = full_runtime.generate_window(
        topology=topology,

        seed=seed,

        batch_size=1,

        num_ttis=num_ttis,

        tti_duration_s=(
            tti_duration_s
        ),
    )

    for offset in range(
        num_ttis
    ):

        sliced = (
            sliced_runtime
            .generate_tti_slice(
                topology=topology,

                seed=seed,

                batch_size=1,

                num_ttis=num_ttis,

                tti_duration_s=(
                    tti_duration_s
                ),

                tti_offset=offset,
            )
        )

        assert (
            int(
                sliced.shape[-2]
            )
            == 1
        )

        torch.testing.assert_close(
            sliced,

            full.tti_slice(
                offset
            ),

            rtol=1.0e-5,

            atol=1.0e-6,
        )


def test_single_tti_conversion_rejects_invalid_offset():
    (
        topology,
        channel_config,
    ) = build_small_temporal_case()

    runtime = (
        VelocityWindowFrequencyChannelRuntime(
            config=channel_config,
        )
    )

    with pytest.raises(
        ValueError,
        match="tti_offset",
    ):
        runtime.generate_tti_slice(
            topology=topology,

            seed=123,

            batch_size=1,

            num_ttis=4,

            tti_duration_s=0.0005,

            tti_offset=4,
        )
