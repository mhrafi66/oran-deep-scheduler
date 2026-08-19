from dataclasses import dataclass
from pathlib import Path
import matplotlib
import torch
from sionna.sys import HexGrid, gen_hexgrid_topology

matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class TopologyConfig:
    """Configuration for the Bell Labs evaluation topology."""

    batch_size: int = 1

    # One center site plus one surrounding ring:
    # 7 sites total, each divided into 3 sectors.
    num_rings: int = 1

    num_sectors_per_site: int = 3

    # Paper evaluation:
    # 210 UEs / 21 sectors = 10 UEs per sector.
    num_ut_per_sector: int = 10

    scenario: str = "uma"

    isd_m: float = 200.0
    bs_height_m: float = 25.0
    ut_height_m: float = 1.5

    ut_speed_kmh: float = 3.0

    precision: str = "single"
    device: str = "cuda:0"

    @property
    def ut_speed_mps(self) -> float:
        """Convert UE speed from km/h to m/s."""
        return self.ut_speed_kmh / 3.6

@dataclass
class TopologyData:
    """Tensors and grid returned by Sionna topology generation."""

    ut_loc: torch.Tensor
    bs_loc: torch.Tensor

    ut_orientations: torch.Tensor
    bs_orientations: torch.Tensor

    ut_velocities: torch.Tensor

    in_state: torch.Tensor

    los: torch.Tensor | None

    bs_virtual_loc: torch.Tensor

    grid: HexGrid


def generate_topology(
        config: TopologyConfig,
) -> TopologyData:
    """Generate a topology for the Bell Labs evaluation scenario.

    Args:
        config: Topology configuration.

    Returns:
        Generated topology data.
    """

    topology = gen_hexgrid_topology(
        batch_size=config.batch_size,
        num_rings=config.num_rings,
        num_ut_per_sector=config.num_ut_per_sector,
        scenario=config.scenario,
        isd=config.isd_m,
        bs_height=config.bs_height_m,
        min_ut_height=config.ut_height_m,
        max_ut_height=config.ut_height_m,
        min_ut_velocity=config.ut_speed_mps,
        max_ut_velocity=config.ut_speed_mps,
        return_grid=True,
        precision=config.precision,
        device=config.device,
    )

    (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        los,
        bs_virtual_loc,
        grid,
    ) = topology

    return TopologyData(
        ut_loc= ut_loc,
        bs_loc= bs_loc,
        ut_orientations= ut_orientations,
        bs_orientations= bs_orientations,
        ut_velocities= ut_velocities,
        in_state= in_state,
        los= los,
        bs_virtual_loc= bs_virtual_loc,
        grid= grid
    )


def validate_evaluation_topology(
        topology: TopologyData,
        config: TopologyConfig
) -> None:
    """Validate the generated topology against the Bell Labs evaluation scenario.

    Args:
        topology: Generated topology data.
        config: Topology configuration.

    Raises:
        AssertionError: If any of the validation checks fail.
    """

    expected_num_cells = 21
    expected_num_ues = 210

    expected_ut_shape =(
        config.batch_size,
        expected_num_ues,
        3
    )

    expected_bs_shape = (
        config.batch_size,
        expected_num_cells,
        3
    )

    assert tuple(topology.ut_loc.shape) == expected_ut_shape, (
        f"Expected UE location shape {expected_ut_shape}, "
        f"but got {tuple(topology.ut_loc.shape)}."
    )

    assert tuple(topology.bs_loc.shape) == expected_bs_shape, (
        f"Expected BS location shape {expected_bs_shape}, "
        f"but got {tuple(topology.bs_loc.shape)}."
    )

    expected_virtual_bs_shape = (
        config.batch_size,
        expected_num_cells,
        expected_num_ues,
        3
    )

    assert tuple(topology.bs_virtual_loc.shape) == expected_virtual_bs_shape, (
        f"Expected virtual BS location shape {expected_virtual_bs_shape}, "
        f"but got {tuple(topology.bs_virtual_loc.shape)}."
    )


    bs_heights = topology.bs_loc[..., 2]

    expected_bs_height = torch.full_like(
        bs_heights,
        fill_value=config.bs_height_m
    )

    torch.testing.assert_close(
        bs_heights,
        expected_bs_height,
    )

    ut_heights = topology.ut_loc[..., 2]

    expected_ut_height = torch.full_like(
        ut_heights,
        fill_value=config.ut_height_m
    )

    torch.testing.assert_close(
        ut_heights,
        expected_ut_height,
    )

    ue_speeds = torch.linalg.norm(topology.ut_velocities, dim=-1)

    expected_ue_speed = torch.full_like(
        ue_speeds,
        fill_value=config.ut_speed_mps
    )

    torch.testing.assert_close(
        ue_speeds,
        expected_ue_speed,
        rtol=1e-4,
        atol=1e-4
    )


def print_topology_summary(
    topology: TopologyData,
    config: TopologyConfig,
) -> None:
    """Print key dimensions and paper parameters."""

    ue_speeds = torch.linalg.vector_norm(
        topology.ut_velocities,
        dim=-1,
    )

    print("=" * 70)
    print("Bell Labs Evaluation Topology Sanity Check")
    print("=" * 70)

    print(f"Scenario:              {config.scenario}")
    print(f"Batch size:            {config.batch_size}")

    print(f"BS tensor shape:       {tuple(topology.bs_loc.shape)}")
    print(f"UE tensor shape:       {tuple(topology.ut_loc.shape)}")

    print(f"Number of cells:       {topology.bs_loc.shape[1]}")
    print(f"Number of UEs:         {topology.ut_loc.shape[1]}")

    print(f"Inter-site distance:   {config.isd_m:.1f} m")
    print(f"gNB height:            {config.bs_height_m:.1f} m")
    print(f"UE height:             {config.ut_height_m:.1f} m")

    print(
        "UE speed:              "
        f"{ue_speeds.mean().item():.6f} m/s "
        f"({config.ut_speed_kmh:.1f} km/h)"
    )

    print(
        "Virtual BS shape:       "
        f"{tuple(topology.bs_virtual_loc.shape)}"
    )

    print("=" * 70)

def save_topology_plot(
    topology: TopologyData,
    output_path: str | Path,
) -> Path:
    """Save a visual sanity-check plot of cells and UEs."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = topology.grid.show(
        show_sectors=True,
    )

    axis = figure.axes[0]

    ue_xy = (
        topology.ut_loc[0, :, :2]
        .detach()
        .cpu()
        .numpy()
    )

    bs_xy = (
        topology.bs_loc[0, :, :2]
        .detach()
        .cpu()
        .numpy()
    )

    axis.scatter(
        ue_xy[:, 0],
        ue_xy[:, 1],
        marker="x",
        s=14,
        label="UE",
    )

    axis.scatter(
        bs_xy[:, 0],
        bs_xy[:, 1],
        marker="^",
        s=28,
        label="Sector gNB",
    )

    axis.legend()

    axis.set_title(
        "Bell Labs Evaluation Topology Sanity Check"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path



