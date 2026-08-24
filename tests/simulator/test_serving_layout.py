import torch

from oran_scheduler.simulator.cell_association import (
    CellAssociationData,
)
from oran_scheduler.simulator.rbg import (
    RBGChannelData,
)
from oran_scheduler.simulator.serving_layout import (
    ServingCellData,
    build_serving_cell_data,
    gather_serving_ue_rbg_values,
)

def test_variable_size_serving_layout():
    device = "cuda:0"

    batch_size = 1
    num_ues = 6
    num_cells = 3
    num_rbgs = 2

    power = torch.zeros(
        batch_size,
        num_ues,
        num_cells,
        num_rbgs,
        device=device,
    )

    for ue in range(num_ues):
        for cell in range(num_cells):
            for rbg in range(num_rbgs):
                power[0, ue, cell, rbg] = (
                    1000 * ue
                    + 100 * cell
                    + rbg
                )

    rbg_data = RBGChannelData(
        power=power,
    )

    serving_bs = torch.tensor(
        [
            [0, 1, 1, 2, 1, 0]
        ],
        device=device,
    )

    association_mask = torch.nn.functional.one_hot(
        serving_bs,
        num_classes=num_cells,
    ).to(torch.bool)

    num_ues_per_cell = association_mask.sum(
        dim=1,
    )

    association = CellAssociationData(
        serving_bs=serving_bs,
        association_mask=association_mask,
        num_ues_per_cell=num_ues_per_cell,
        association_metric=torch.ones(
            batch_size,
            num_ues,
            num_cells,
            device=device,
        ),
    )

    serving_data = build_serving_cell_data(
        rbg_data=rbg_data,
        association=association,
    )

    assert serving_data.power.shape == (
        1,
        3,
        3,
        2,
    )

    assert serving_data.global_ue_indices.shape == (
        1,
        3,
        3,
    )

    assert serving_data.valid_ue_mask.shape == (
        1,
        3,
        3,
    )

    expected_global_indices = torch.tensor(
        [
            [
                [0, 5, -1],
                [1, 2, 4],
                [3, -1, -1],
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        serving_data.global_ue_indices,
        expected_global_indices,
    )

    expected_mask = torch.tensor(
        [
            [
                [True, True, False],
                [True, True, True],
                [True, False, False],
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        serving_data.valid_ue_mask,
        expected_mask,
    )

    actual = serving_data.power[
        0,
        1,
        2,
        1,
    ]

    expected = torch.tensor(
        4101.0,
        device=device,
    )

    torch.testing.assert_close(
        actual,
        expected,
    )

def test_gather_serving_ue_rbg_values():
    device = "cuda:0"

    global_values = torch.tensor(
        [
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
            ]
        ],
        device=device,
    )
    serving_data = ServingCellData(
        power=torch.zeros(
            1,
            2,
            3,
            2,
            device=device,
        ),
        global_ue_indices=torch.tensor(
            [
                [
                    [2, 0, -1],
                    [3, 1, -1],
                ]
            ],
            dtype=torch.long,
            device=device,
        ),
        valid_ue_mask=torch.tensor(
            [
                [
                    [True, True, False],
                    [True, True, False],
                ]
            ],
            dtype=torch.bool,
            device=device,
        ),
        num_ues_per_cell=torch.tensor(
            [
                [2, 2]
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    serving_values = (
        gather_serving_ue_rbg_values(
            global_ue_rbg_values=global_values,
            serving_data=serving_data,
        )
    )

    assert serving_values.shape == (
        1,
        2,
        3,
        2,
    )

    expected_cell_0 = torch.tensor(
        [
            [30.0, 31.0],
            [10.0, 11.0],
            [0.0, 0.0],
        ],
        device=device,
    )

    expected_cell_1 = torch.tensor(
        [
            [40.0, 41.0],
            [20.0, 21.0],
            [0.0, 0.0],
        ],
        device=device,
    )

    torch.testing.assert_close(
        serving_values[
            0,
            0,
            :,
            :,
        ],
        expected_cell_0,
    )

    torch.testing.assert_close(
        serving_values[
            0,
            1,
            :,
            :,
        ],
        expected_cell_1,
    )

def test_gather_serving_ue_rbg_values_zeroes_padding():
    device = "cuda:0"

    global_values = torch.full(
        (
            1,
            2,
            3,
        ),
        fill_value=9999.0,
        device=device,
    )

    serving_data = ServingCellData(
        power=torch.zeros(
            1,
            1,
            2,
            3,
            device=device,
        ),
        global_ue_indices=torch.tensor(
            [
                [
                    [1, -1]
                ]
            ],
            dtype=torch.long,
            device=device,
        ),
        valid_ue_mask=torch.tensor(
            [
                [
                    [True, False]
                ]
            ],
            dtype=torch.bool,
            device=device,
        ),
        num_ues_per_cell=torch.tensor(
            [
                [1]
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    serving_values = (
        gather_serving_ue_rbg_values(
            global_ue_rbg_values=global_values,
            serving_data=serving_data,
        )
    )

    assert torch.all(
        serving_values[
            0,
            0,
            1,
            :,
        ]
        == 0.0
    )





      