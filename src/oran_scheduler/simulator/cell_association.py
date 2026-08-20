from dataclasses import dataclass

import torch

from sionna.sys.utils import get_pathloss

@dataclass
class CellAssociationData:
    """
    UE-to-cell association information.

    serving_bs:
        [batch, UE]

        serving_bs[b, u] gives the BS index serving UE u.

    association_mask:
        [batch, UE, BS]

        Boolean one-hot representation of serving_bs.

    num_ues_per_cell:
        [batch, BS]

        Number of UEs associated with each BS.

    association_metric:
        [batch, UE, BS]

        Metric used to make the association decision.

        For the current implementation this is channel-averaged
        pathloss in linear scale, where smaller is better.
    """

    serving_bs: torch.Tensor

    association_mask: torch.Tensor

    num_ues_per_cell: torch.Tensor

    association_metric: torch.Tensor

def compute_channel_averaged_pathloss(
    h_freq: torch.Tensor,
    precision: str = "single",
) -> torch.Tensor:
    """
    Compute an all-UE/all-BS pathloss metric from the OFDM channel.

    Input channel shape:

        [batch,
         UE,
         RX antenna,
         BS,
         TX antenna,
         OFDM symbol,
         subcarrier]

    Current temporary-array training example:

        [1, 420, 1, 21, 1, 1, 216]

    Sionna get_pathloss() returns:

        [batch, UE, BS, OFDM symbol]

    We then average over the OFDM-symbol dimension to obtain:

        [batch, UE, BS]

    Important reproduction note:
        This is an open-reproduction association metric.

        It averages the generated channel power over antennas and
        subcarriers and is therefore substantially less sensitive to
        individual RBG fading than selecting the strongest instantaneous
        RBG link.

        However, it should not be interpreted as the exact proprietary
        Nokia cell-association procedure.
    """

    if h_freq.ndim != 7:
        raise ValueError(
            "Expected h_freq shape "
            "[batch, UE, RX_ant, BS, TX_ant, OFDM_symbol, subcarrier], "
            f"got {tuple(h_freq.shape)}."
        )


    pathloss_all_pairs, _ = get_pathloss(
        h_freq=h_freq,
        precision=precision,
    )

    pathloss = pathloss_all_pairs.mean(
        dim=-1,
    )
    if not torch.isfinite(pathloss).all():
        raise ValueError(
            "Pathloss contains non-finite values."
        )

    if torch.any(pathloss <= 0):
        raise ValueError(
            "Pathloss must be strictly positive."
        )

    return pathloss

def associate_by_minimum_pathloss(
    pathloss: torch.Tensor,
) -> CellAssociationData:
    """
    Associate each UE with the BS having minimum pathloss.

    Input:

        pathloss:
            [batch, UE, BS]

    Returns:
        CellAssociationData

    Association rule:

        serving_bs(u) = argmin_b pathloss(u, b)

    Because all Bell Labs gNBs currently use the same transmit power,
    minimizing pathloss is equivalent to maximizing average received
    power under this association model.
    """

    if pathloss.ndim != 3:
        raise ValueError(
            "Expected pathloss shape [batch, UE, BS], "
            f"got {tuple(pathloss.shape)}."
        )

    serving_bs = torch.argmin(
        pathloss,
        dim=-1,
    )

    num_bs = pathloss.shape[-1]

    association_mask = torch.nn.functional.one_hot(
        serving_bs,
        num_classes=num_bs,
    )

    association_mask = association_mask.to(
        dtype=torch.bool,
    )

    num_ues_per_cell = association_mask.sum(
        dim=1,
    )   

    return CellAssociationData(
        serving_bs=serving_bs,
        association_mask=association_mask,
        num_ues_per_cell=num_ues_per_cell,
        association_metric=pathloss,
    )

def build_cell_association(
    h_freq: torch.Tensor,
    precision: str = "single",
) -> CellAssociationData:
    """
    Compute the current open-reproduction cell association.

    Pipeline:

        full all-pair channel
            ->
        channel-averaged pathloss
            ->
        minimum-pathloss serving BS
    """

    pathloss = compute_channel_averaged_pathloss(
        h_freq=h_freq,
        precision=precision,
    )

    return associate_by_minimum_pathloss(
        pathloss=pathloss,
    )

def validate_cell_association(
    association: CellAssociationData,
    expected_batch_size: int,
    expected_num_ues: int,
    expected_num_bs: int,
) -> None:
    """
    Validate dimensions and one-serving-cell-per-UE invariants.
    """

    expected_serving_shape = (
        expected_batch_size,
        expected_num_ues,
    )

    expected_mask_shape = (
        expected_batch_size,
        expected_num_ues,
        expected_num_bs,
    )

    expected_count_shape = (
        expected_batch_size,
        expected_num_bs,
    )

    expected_metric_shape = (
        expected_batch_size,
        expected_num_ues,
        expected_num_bs,
    )

    assert (
        tuple(association.serving_bs.shape)
        == expected_serving_shape
    ), (
        f"Expected serving BS shape "
        f"{expected_serving_shape}, got "
        f"{tuple(association.serving_bs.shape)}."
    )

    assert (
        tuple(association.association_mask.shape)
        == expected_mask_shape
    ), (
        f"Expected association-mask shape "
        f"{expected_mask_shape}, got "
        f"{tuple(association.association_mask.shape)}."
    )

    assert (
        tuple(association.num_ues_per_cell.shape)
        == expected_count_shape
    ), (
        f"Expected UE-count shape "
        f"{expected_count_shape}, got "
        f"{tuple(association.num_ues_per_cell.shape)}."
    )

    assert (
        tuple(association.association_metric.shape)
        == expected_metric_shape
    )

    associations_per_ue = (
        association.association_mask.sum(
            dim=-1,
        )
    )

    assert torch.all(
        associations_per_ue == 1
    ), (
        "Every UE must be associated with exactly one BS."
    )

    total_associated_ues = (
        association.num_ues_per_cell.sum(
            dim=-1,
        )
    )

    assert torch.all(
        total_associated_ues == expected_num_ues
    ), (
        "Total associated UEs does not equal the "
        "number of generated UEs."
    )

    assert torch.all(
        association.serving_bs >= 0
    )

    assert torch.all(
        association.serving_bs < expected_num_bs
    )

def print_cell_association_summary(
    association: CellAssociationData,
) -> None:
    """
    Print UE distribution across serving cells.
    """

    print("=" * 72)
    print("Cell Association Sanity Check")
    print("=" * 72)

    print(
        f"Serving-BS tensor shape:   "
        f"{tuple(association.serving_bs.shape)}"
    )

    print(
        f"Association mask shape:    "
        f"{tuple(association.association_mask.shape)}"
    )

    print(
        f"Association metric shape:  "
        f"{tuple(association.association_metric.shape)}"
    )

    print()

    counts = association.num_ues_per_cell[0]

    for bs_index, count in enumerate(counts):
        print(
            f"BS {bs_index:2d}: "
            f"{int(count.item()):3d} associated UEs"
        )

    print()

    print(
        "Minimum UEs in a cell:    "
        f"{int(counts.min().item())}"
    )

    print(
        "Maximum UEs in a cell:    "
        f"{int(counts.max().item())}"
    )

    print(
        "Mean UEs per cell:        "
        f"{counts.float().mean().item():.2f}"
    )

    print(
        "Total associated UEs:     "
        f"{int(counts.sum().item())}"
    )

    print("=" * 72)

