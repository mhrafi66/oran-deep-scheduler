from oran_scheduler.simulator.channel import (
    ChannelConfig,
    create_paper_arrays,
)

from oran_scheduler.simulator.channel import (
    create_channel_arrays,
)


def test_paper_array_dimensions():
    config = ChannelConfig(
        antenna_mode="paper",
        device="cuda:0",
    )

    bs_array, ut_array = create_paper_arrays(
        config
    )

    assert int(bs_array.num_ant) == 192

    assert int(ut_array.num_ant) == 4

    assert bs_array.polarization == "dual"
    assert ut_array.polarization == "dual"

def test_sanity_array_dimensions_unchanged():
    config = ChannelConfig(
        antenna_mode="sanity",
        device="cuda:0",
    )

    bs_array, ut_array = create_channel_arrays(
        config
    )

    assert int(bs_array.num_ant) == 1
    assert int(ut_array.num_ant) == 1