import torch

from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
    validate_frequency_channel,
    FrequencyChannelRuntime,
)

from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
    subset_topology_ues,
)


def test_4ghz_uma_frequency_channel() -> None:
    topology_config = TopologyConfig(
        device="cpu",
    )

    channel_config = ChannelConfig()

    topology = generate_topology(
        config=topology_config,
    )

    channel = generate_frequency_channel(
        topology=topology,
        topology_config=topology_config,
        channel_config=channel_config,
    )

    validate_frequency_channel(
        channel=channel,
        topology=topology,
        channel_config=channel_config,
    )


def test_persistent_frequency_channel_runtime_matches_fresh_generation():
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,
        num_ut_per_sector=1,
        device=device,
        seed=321,
    )

    topology = generate_topology(
        topology_config
    )

    first_indices = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.long,
        device=device,
    )

    second_indices = torch.tensor(
        [
            2,
            3,
            4,
        ],
        dtype=torch.long,
        device=device,
    )

    first_topology = subset_topology_ues(
        topology=topology,
        global_ue_indices=first_indices,
    )

    second_topology = subset_topology_ues(
        topology=topology,
        global_ue_indices=second_indices,
    )

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,
        subcarrier_spacing_hz=30.0e3,

        #
        # Keep this test inexpensive while still
        # exercising the paper antenna arrays.
        #
        num_rbs=2,
        subcarriers_per_rb=12,
        num_ofdm_symbols=1,

        antenna_mode="paper",

        direction="downlink",
        o2i_model="low",

        enable_pathloss=True,
        enable_shadow_fading=True,

        precision="single",
        device=device,

        seed=12345,
    )

    # ----------------------------------------------------------
    # Existing implementation:
    #
    # freshly constructs every Sionna object.
    # ----------------------------------------------------------

    fresh = generate_frequency_channel(
        topology=first_topology,
        topology_config=topology_config,
        channel_config=channel_config,
    )

    fresh_h = (
        fresh.h_freq
        .detach()
        .clone()
    )

    # ----------------------------------------------------------
    # New implementation:
    #
    # construct Sionna objects once.
    # ----------------------------------------------------------

    runtime = FrequencyChannelRuntime(
        config=channel_config
    )

    persistent_first = runtime.generate(
        topology=first_topology,
        seed=12345,
        batch_size=1,
    )

    torch.testing.assert_close(
        persistent_first.h_freq,
        fresh_h,
    )

    # ----------------------------------------------------------
    # Force the runtime to change both topology and
    # number of UEs.
    # ----------------------------------------------------------

    persistent_second = runtime.generate(
        topology=second_topology,
        seed=54321,
        batch_size=1,
    )

    assert (
        persistent_second
        .h_freq
        .shape[1]
        == 3
    )

    # ----------------------------------------------------------
    # Return to the original topology and seed.
    #
    # This catches stale topology state as well as
    # RNG-state leakage.
    # ----------------------------------------------------------

    persistent_first_again = (
        runtime.generate(
            topology=first_topology,
            seed=12345,
            batch_size=1,
        )
    )

    torch.testing.assert_close(
        persistent_first_again.h_freq,
        fresh_h,
    )

    assert (
        runtime.num_generations
        == 3
    )

    #
    # Shapes were:
    #
    #     2 UEs
    #       ->
    #     3 UEs     reset #1
    #       ->
    #     2 UEs     reset #2
    #
    assert (
        runtime.num_topology_resets
        == 2
    )