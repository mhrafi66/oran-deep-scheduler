import torch


from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
    validate_evaluation_topology
)

from oran_scheduler.simulator.topology import (
    TopologyData,
    subset_topology_ues,
)

def test_bell_labs_evaluation_topology() -> None:
    config = TopologyConfig(device="cuda:0")
    topology = generate_topology(config)
    validate_evaluation_topology(topology, config)



def test_subset_topology_preserves_all_bs_and_global_ue_order():
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    batch_size = 1
    num_ues = 5
    num_bs = 3

    topology = TopologyData(
        ut_loc=torch.arange(
            batch_size
            * num_ues
            * 3,
            dtype=torch.float32,
            device=device,
        ).reshape(
            batch_size,
            num_ues,
            3,
        ),

        bs_loc=torch.arange(
            batch_size
            * num_bs
            * 3,
            dtype=torch.float32,
            device=device,
        ).reshape(
            batch_size,
            num_bs,
            3,
        ),

        ut_orientations=torch.arange(
            batch_size
            * num_ues
            * 3,
            dtype=torch.float32,
            device=device,
        ).reshape(
            batch_size,
            num_ues,
            3,
        ),

        bs_orientations=torch.zeros(
            (
                batch_size,
                num_bs,
                3,
            ),
            device=device,
        ),

        ut_velocities=torch.arange(
            batch_size
            * num_ues
            * 3,
            dtype=torch.float32,
            device=device,
        ).reshape(
            batch_size,
            num_ues,
            3,
        ),

        in_state=torch.tensor(
            [
                [
                    False,
                    True,
                    False,
                    True,
                    False,
                ]
            ],
            dtype=torch.bool,
            device=device,
        ),

        los=torch.arange(
            batch_size
            * num_bs
            * num_ues,
            dtype=torch.float32,
            device=device,
        ).reshape(
            batch_size,
            num_bs,
            num_ues,
        ),

        bs_virtual_loc=torch.arange(
            batch_size
            * num_bs
            * num_ues
            * 3,
            dtype=torch.float32,
            device=device,
        ).reshape(
            batch_size,
            num_bs,
            num_ues,
            3,
        ),

        #
        # subset_topology_ues() preserves this object
        # but does not inspect it.
        #
        grid=object(),
    )

    selected_global_ues = torch.tensor(
        [
            4,
            1,
        ],
        dtype=torch.long,
        device=device,
    )

    subset = subset_topology_ues(
        topology=topology,
        global_ue_indices=(
            selected_global_ues
        ),
    )

    assert subset.ut_loc.shape == (
        1,
        2,
        3,
    )

    assert subset.bs_loc.shape == (
        1,
        3,
        3,
    )

    assert (
        subset.bs_virtual_loc.shape
        == (
            1,
            3,
            2,
            3,
        )
    )

    torch.testing.assert_close(
        subset.ut_loc[
            0,
            0,
        ],
        topology.ut_loc[
            0,
            4,
        ],
    )

    torch.testing.assert_close(
        subset.ut_loc[
            0,
            1,
        ],
        topology.ut_loc[
            0,
            1,
        ],
    )

    #
    # BS tensor must be untouched.
    #
    torch.testing.assert_close(
        subset.bs_loc,
        topology.bs_loc,
    )

    torch.testing.assert_close(
        subset.bs_virtual_loc[
            :,
            :,
            0,
            :,
        ],
        topology.bs_virtual_loc[
            :,
            :,
            4,
            :,
        ],
    )

    torch.testing.assert_close(
        subset.bs_virtual_loc[
            :,
            :,
            1,
            :,
        ],
        topology.bs_virtual_loc[
            :,
            :,
            1,
            :,
        ],
    )




