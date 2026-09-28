from dataclasses import dataclass

import torch

from sionna.phy import config as sionna_config
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.channel.tr38901 import PanelArray, UMa
from sionna.phy.ofdm import ResourceGrid

from oran_scheduler.simulator.topology import (
    TopologyConfig,
    TopologyData,
)

@dataclass(frozen = True)
class ChannelConfig:
    """Configuration for the channel model."""
    carrier_frequency_hz: float = 4.0e9
    subcarrier_spacing_hz: float = 30e3
    num_rbs: int = 18
    subcarriers_per_rb: int = 12
    num_ofdm_symbols: int = 1
    antenna_mode: str = "sanity"
    direction: str = "downlink"
    # Outdoor-to-indoor penetration-loss model required by Sionna UMa.
    #
    # The Bell Labs paper does not specify this implementation detail,
    # so "low" is currently a reproduction assumption.
    o2i_model: str = "low"
    enable_pathloss: bool = True
    enable_shadow_fading: bool = True
    precision: str = "single"
    device:str = "cuda:0"
    seed: int = 42

    @property 
    def num_subcarriers(self) -> int:
        """Number of subcarriers."""
        return self.num_rbs * self.subcarriers_per_rb

    @property
    def occupied_bandwidth_hz(self) -> float:
        """Occupied bandwidth in Hz."""
        return self.num_subcarriers * self.subcarrier_spacing_hz


@dataclass
class ChannelData:
    """Objects and tensors produced by channel generation. """

    resource_grid: ResourceGrid
    bs_array: PanelArray
    ut_array: PanelArray
    channel_model: UMa
    h_freq: torch.Tensor

def set_channel_seed(seed: int) -> None:
    """Set the seed for channel generation.

    Args:
        seed (int): Seed for channel generation.
    """
    sionna_config.seed = seed
    torch.manual_seed(seed)

def create_sanity_arrays(
        config: ChannelConfig
) -> tuple[PanelArray, PanelArray]:
    """
    Create lightweight arrays for channel-pipeline validations.
    IMPORTANT:
    There are NOT final Bell Labs antenna arrays.

    The full 12x8 dual dual-polarized BS array and paper UE array will be introduced after the channel pipeline itself has been validated.
    """

    bs_array = PanelArray(
        num_rows_per_panel = 1,
        num_cols_per_panel = 1,
        polarization = "single",
        polarization_type = "V",
        antenna_pattern = "38.901",
        carrier_frequency = config.carrier_frequency_hz,
        precision = config.precision,
        device = config.device
    )

    ut_array = PanelArray(
        num_rows_per_panel = 1,
        num_cols_per_panel = 1,
        polarization = "single",
        polarization_type = "V",
        antenna_pattern = "omni",
        carrier_frequency = config.carrier_frequency_hz,
        precision = config.precision,
        device = config.device
    )

    return bs_array, ut_array

def create_paper_arrays(
    config: ChannelConfig,
) -> tuple[PanelArray, PanelArray]:
    """
    Create the antenna arrays used for the paper-aligned MIMO path.

    Paper:
        gNB panel:
            12 x 8 physical elements,
            two polarizations.

        UE:
            2 dual-polarized Rx antennas.

        Maximum UE rank:
            2.

    Open-reproduction interpretation:
        - Sionna dual polarization creates two antenna ports
          per physical element.
        - Therefore the gNB has 12 * 8 * 2 = 192 ports.
        - We represent the UE as 1 x 2 physical elements with
          dual polarization, giving 4 receive ports.
        - Maximum rank 2 will be enforced separately.
        - Polarization type "VH" is an implementation choice.
        - UE uses the omni element pattern.
        - Element spacings use Sionna defaults because the paper
          does not publicly specify them.
    """
    bs_array = PanelArray(
        num_rows_per_panel=12,
        num_cols_per_panel=8,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="38.901",
        carrier_frequency=(
            config.carrier_frequency_hz
        ),
        precision=config.precision,
        device=config.device,
    )

    ut_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=2,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="omni",
        carrier_frequency=(
            config.carrier_frequency_hz
        ),
        precision=config.precision,
        device=config.device,
    )

    return bs_array, ut_array

def create_channel_arrays(
    config: ChannelConfig,
) -> tuple[PanelArray, PanelArray]:
    """
    Create channel arrays for the selected PHY configuration.
    """

    if config.antenna_mode == "sanity":
        return create_sanity_arrays(
            config
        )

    if config.antenna_mode == "paper":
        return create_paper_arrays(
            config
        )

    raise ValueError(
        "Unsupported antenna_mode. "
        "Expected 'sanity' or 'paper', "
        f"got {config.antenna_mode!r}."
    )

def create_training_resource_grid(
        config: ChannelConfig,
) -> ResourceGrid:
    """
    Construct the frequency grid corresponding to the paper's reduced-bandwidth training configuration. 
    """

    return ResourceGrid(
        num_ofdm_symbols = config.num_ofdm_symbols,
        fft_size = config.num_subcarriers,
        subcarrier_spacing = config.subcarrier_spacing_hz,
        num_tx = 1,
        num_streams_per_tx = 1,
        device = config.device
    )


def create_uma_channel_model(
        config: ChannelConfig,
        bs_array: PanelArray,
        ut_array: PanelArray
) -> UMa:
    """
    Create the 4-GHz 3GPP UMa channel model

    """

    return UMa(
        carrier_frequency = config.carrier_frequency_hz,
        o2i_model = config.o2i_model,
        ut_array = ut_array,
        bs_array = bs_array,
        direction = config.direction,
        enable_pathloss = config.enable_pathloss,
        enable_shadow_fading = config.enable_shadow_fading,
        precision = config.precision,
        device = config.device
    )

def attach_topology_to_channel(
        channel_model: UMa,
        topology: TopologyData
) -> None:
    """
    Attach the already generated 21-cell topology to UMa.

    """

    channel_model.set_topology(
        topology.ut_loc,
        topology.bs_loc,
        topology.ut_orientations,
        topology.bs_orientations,
        topology.ut_velocities,
        topology.in_state,
        topology.los,
        topology.bs_virtual_loc,
    )


class FrequencyChannelRuntime:
    """
    Reusable Sionna frequency-channel runtime.

    The expensive configuration-invariant objects are
    constructed once:

        PanelArray
        ResourceGrid
        UMa
        GenerateOFDMChannel

    For every channel realization we only:

        1. reset the deterministic channel seed
        2. attach the requested topology
        3. generate a fresh frequency-domain channel

    OPEN-REPRODUCTION ENGINEERING:
        This changes object lifetime only.

        It must not change:
            - topology
            - channel model
            - antenna configuration
            - random seed
            - generated channel semantics
    """

    def __init__(
        self,
        *,
        config: ChannelConfig,
    ) -> None:
        self.config = config

        (
            self.bs_array,
            self.ut_array,
        ) = create_channel_arrays(
            config
        )

        self.resource_grid = (
            create_training_resource_grid(
                config
            )
        )

        self.channel_model = (
            create_uma_channel_model(
                config=config,
                bs_array=self.bs_array,
                ut_array=self.ut_array,
            )
        )

        self.channel_generator = (
            GenerateOFDMChannel(
                channel_model=(
                    self.channel_model
                ),
                resource_grid=(
                    self.resource_grid
                ),
                precision=config.precision,
                device=config.device,
            )
        )

        self._num_generations = 0
        #
        # Sionna freezes topology tensor shapes after
        # the first set_topology() call.
        #
        # We therefore track:
        #
        #     batch size
        #     number of UEs
        #     number of BSs
        #
        # and reset only when one of those dimensions
        # changes.
        #
        self._topology_shape: (
            tuple[int, int, int]
            | None
        ) = None

        self._num_topology_resets = 0


    @property
    def num_generations(
        self,
    ) -> int:
        return self._num_generations


    @property
    def num_topology_resets(
        self,
    ) -> int:
        return self._num_topology_resets

    def generate(
        self,
        *,
        topology: TopologyData,
        seed: int,
        batch_size: int,
    ) -> ChannelData:
        """
        Generate one fresh channel realization while
        reusing the invariant Sionna objects.

        Sionna freezes the topology tensor shapes after
        the first set_topology() call.

        Therefore:

            same shape
                ->
            update topology in-place

            changed batch/UE/BS count
                ->
            reset_topology()
                ->
            attach new topology

        This preserves persistent object reuse while
        supporting variable-size UE microbatches.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        topology_batch_size = int(
            topology.ut_loc.shape[0]
        )

        num_topology_ues = int(
            topology.ut_loc.shape[1]
        )

        num_topology_bs = int(
            topology.bs_loc.shape[1]
        )

        topology_shape = (
            topology_batch_size,
            num_topology_ues,
            num_topology_bs,
        )

        if batch_size != topology_batch_size:
            raise ValueError(
                "Requested channel batch size does not "
                "match the topology batch dimension."
            )

        #
        # Sionna's topology buffers are shape-frozen
        # after first use.
        #
        # A reset is required only when one of the
        # topology dimensions changes.
        #
        if (
            self._topology_shape is not None
            and topology_shape
            != self._topology_shape
        ):
            self.channel_model.reset_topology()

            self._num_topology_resets += 1

        #
        # Preserve deterministic channel-generation
        # semantics from generate_frequency_channel().
        #
        set_channel_seed(
            seed
        )

        attach_topology_to_channel(
            channel_model=(
                self.channel_model
            ),
            topology=topology,
        )

        #
        # After set_topology() succeeds, this is now
        # the shape frozen inside Sionna.
        #
        self._topology_shape = (
            topology_shape
        )

        h_freq = (
            self.channel_generator(
                batch_size
            )
        )

        self._num_generations += 1

        return ChannelData(
            resource_grid=(
                self.resource_grid
            ),
            bs_array=(
                self.bs_array
            ),
            ut_array=(
                self.ut_array
            ),
            channel_model=(
                self.channel_model
            ),
            h_freq=h_freq,
        )


def generate_frequency_channel(
        topology: TopologyData,
        topology_config: TopologyConfig,
        channel_config: ChannelConfig
) -> ChannelData:
    """
   Generate one frequency-selective UMa channel realization.
    """
    set_channel_seed(channel_config.seed)

    bs_array, ut_array = create_channel_arrays(
        channel_config
    )

    resource_grid = create_training_resource_grid(channel_config)

    channel_model = create_uma_channel_model(
        config = channel_config,
        bs_array = bs_array,
        ut_array = ut_array
    )

    attach_topology_to_channel(
        channel_model = channel_model,
        topology = topology 
    )

    channel_generator = GenerateOFDMChannel(
        channel_model = channel_model,
        resource_grid = resource_grid,
        precision = channel_config.precision,
        device = channel_config.device
    )

    h_freq = channel_generator(
        topology_config.batch_size
    )

    return ChannelData(
        resource_grid = resource_grid,
        bs_array = bs_array,
        ut_array = ut_array,
        channel_model = channel_model,
        h_freq = h_freq
    )


def validate_frequency_channel(
        channel: ChannelData,
        topology: TopologyData,
        channel_config: ChannelConfig,
) -> None:
    """
    Basic Structural and numerical checks
    """

    h_freq = channel.h_freq

    assert torch.is_complex(h_freq), (
        "Expected a complex-valued frequency-domain channel."
    )

    assert h_freq.shape[0] == topology.ut_loc.shape[0], (
        "Unexpected batch dimension."
    )

    assert h_freq.shape[1] == topology.ut_loc.shape[1], (
        "Expected one RX dimension entry per UE."
    )

    assert h_freq.shape[3] == topology.bs_loc.shape[1], (
        "Expected one TX dimension entry per BS."
    )

    assert h_freq.shape[-1] == channel_config.num_subcarriers, (
        f"Expected {channel_config.num_subcarriers} subcarriers, "
        f"got {h_freq.shape[-1]}."
    )

    assert torch.isfinite(h_freq.real).all(), (
        "Channel contains non-finite real values."
    )

    assert torch.isfinite(h_freq.imag).all(), (
        "Channel contains non-finite imaginary values."
    )

    assert torch.any(torch.abs(h_freq) > 0), (
        "Channel is identically zero."
    )

def compute_link_power(
        channel: ChannelData
) -> torch.Tensor:
    """
    Average |H|^2 over antennas, OFDM symbols, and subcarriers.
    
    returns:
        Tensor of shape (batch_size, num_ues, num_bs) containing the average link power.
    """

    power = torch.abs(
        channel.h_freq
    ) ** 2

    link_power = power.mean(
        dim = (2,4,5,6)
    )
    return link_power

def print_channel_summary(
    channel: ChannelData,
    topology: TopologyData,
    channel_config: ChannelConfig,
) -> None:
    """Print key properties of the generated channel."""

    link_power = compute_link_power(
        channel
    )

    strongest_bs = torch.argmax(
        link_power,
        dim=-1,
    )

    num_ut_per_sector = (
        topology.ut_loc.shape[1]
        // topology.bs_loc.shape[1]
    )

    nominal_serving_bs = (
        torch.arange(
            topology.ut_loc.shape[1],
            device=strongest_bs.device,
        )
        // num_ut_per_sector
    )

    nominal_serving_bs = nominal_serving_bs.unsqueeze(
        0
    )

    strongest_matches_nominal = (
        strongest_bs == nominal_serving_bs
    ).float().mean()

    print("=" * 72)
    print("4 GHz UMa Channel Sanity Check")
    print("=" * 72)

    print(
        f"Carrier frequency:      "
        f"{channel_config.carrier_frequency_hz / 1e9:.2f} GHz"
    )

    print(
        f"Subcarrier spacing:     "
        f"{channel_config.subcarrier_spacing_hz / 1e3:.1f} kHz"
    )

    print(
        f"Number of RBs:          "
        f"{channel_config.num_rbs}"
    )

    print(
        f"Number of subcarriers:  "
        f"{channel_config.num_subcarriers}"
    )

    print(
        f"Occupied bandwidth:     "
        f"{channel_config.occupied_bandwidth_hz / 1e6:.3f} MHz"
    )

    print(
        f"Channel tensor shape:   "
        f"{tuple(channel.h_freq.shape)}"
    )

    print(
        f"Link-power shape:       "
        f"{tuple(link_power.shape)}"
    )

    print(
        "Strongest BS matches nominal sector: "
        f"{100.0 * strongest_matches_nominal.item():.2f}%"
    )

    print(
        f"Mean |H|^2:             "
        f"{link_power.mean().item():.6e}"
    )

    print("=" * 72)



