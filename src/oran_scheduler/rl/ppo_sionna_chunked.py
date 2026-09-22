from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from oran_scheduler.phy.csi import (
    compute_ideal_svd_csi,
)
from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.rank_diagnostic import (
    compute_ideal_svd_rank_diagnostic,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.phy.sinr import (
    SINRConfig,
    compute_noise_power_per_subcarrier_w,
    compute_subcarrier_tx_power_w,
)
from oran_scheduler.phy.su_mimo_reports import (
    build_single_user_phy_reports,
)

from oran_scheduler.rl.ppo_physical_score import (
    PPOPhysicalScoreInputs,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)

from oran_scheduler.simulator.cell_association import (
    CellAssociationData,
    build_cell_association,
)
from oran_scheduler.simulator.channel import (
    ChannelConfig,
    FrequencyChannelRuntime,
    generate_frequency_channel,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
    OneLDSCellTTIStateManager,
    PreparedOneLDSCellTTI,
)
from oran_scheduler.simulator.mobility import (
    MobilityConfig,
    build_mobility_snapshot,
)
from oran_scheduler.simulator.temporal_channel import (
    VelocityWindowFrequencyChannelRuntime,
)
from oran_scheduler.simulator.topology import (
    TopologyConfig,
    TopologyData,
    generate_topology,
    subset_topology_ues,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)
from oran_scheduler.state.cqi_features import (
    CQISurrogateConfig,
    build_cqi_surrogate,
)
from oran_scheduler.utils.perf_timing import (
    perf_region,
)


@dataclass(frozen=True)
class ChunkedSionnaPPOConfig:
    """
    Memory-safe paper-training radio configuration.

    PAPER-SPECIFIED:
        21 cells
        420 training UEs
        20 generated UEs / sector
        18 RB
        18 RBG
        4 GHz
        30 kHz
        paper MIMO antenna arrays

    OPEN-REPRODUCTION ENGINEERING:
        paper MIMO is generated one serving cell
        at a time, and the serving-cell UE population
        is further divided into small UE microbatches.

        The microbatch boundary exists only for radio
        generation. PF-TDS still operates over every
        UE associated with the cell.

    Each local channel still contains ALL 21 BSs.

    The scheduler continues to use persistent global
    UE identities.

    num_training_cells may be reduced for an initial
    smoke test, but num_ut_per_sector remains 20.
    """

    num_training_cells: int = 2

    num_ut_per_sector: int = 20

    num_rbs: int = 18

    num_rbgs: int = 18

    subcarriers_per_rb: int = 12

    #
    # OPEN-REPRODUCTION ENGINEERING.
    #
    # Number of serving UEs sent through one
    # paper-array Sionna channel-generation call.
    #
    # This changes only memory/runtime partitioning.
    # PF-TDS still sees the complete serving-cell
    # population after microbatch recombination.
    #
    ue_microbatch_size: int = 2

    #
    # WAVE-6 DYNAMIC ASSOCIATION.
    #
    # False:
    #     preserve the existing cell/microbatch RNG
    #     behavior exactly.
    #
    # True:
    #     require ue_microbatch_size == 1 and key
    #     radio RNG by persistent GLOBAL UE identity.
    #
    # This prevents handover from changing a UE's
    # random channel process merely because its
    # serving cell or local slot changed.
    #
    identity_stable_ue_channel_rng: bool = False

    topology_seed: int = 42

    association_channel_seed: int = 1000

    mimo_channel_seed: int = 2000

    device: str = "cuda:0"

    # ----------------------------------------------------------
    # TOPOLOGY / STATIC STRESS PARAMETERS
    #
    # Defaults reproduce the paper configuration.
    # ----------------------------------------------------------

    scenario: str = "uma"

    isd_m: float = 200.0

    bs_height_m: float = 25.0

    ut_height_m: float = 1.5

    ut_speed_kmh: float = 3.0

    # ----------------------------------------------------------
    # WAVE-4 TEMPORAL RADIO
    #
    # independent:
    #     exact previous behavior
    #
    # velocity_window:
    #     one velocity-driven Sionna temporal
    #     realization is addressed in windows.
    #
    # OPEN-REPRODUCTION:
    #     simulator scheduling-step duration.
    # ----------------------------------------------------------

    temporal_radio_mode: str = "independent"

    temporal_window_ttis: int = 8

    tti_duration_s: float = 0.001

    temporal_max_displacement_m: float = 20.0


    def __post_init__(
        self,
    ) -> None:
        if not (
            1
            <= self.num_training_cells
            <= 21
        ):
            raise ValueError(
                "num_training_cells must lie in [1, 21]."
            )

        if self.num_ut_per_sector <= 0:
            raise ValueError(
                "num_ut_per_sector must be positive."
            )

        if self.num_rbs != self.num_rbgs:
            raise ValueError(
                "Current training path requires "
                "18 RB -> 18 RBG."
            )

        if self.subcarriers_per_rb <= 0:
            raise ValueError(
                "subcarriers_per_rb must be positive."
            )
        if self.ue_microbatch_size <= 0:
            raise ValueError(
                "ue_microbatch_size must be positive."
            )

        if (
            self.identity_stable_ue_channel_rng
            and self.ue_microbatch_size != 1
        ):
            raise ValueError(
                "identity-stable UE channel RNG "
                "requires ue_microbatch_size=1."
            )


        if not self.scenario:
            raise ValueError(
                "scenario cannot be empty."
            )

        if self.isd_m <= 0.0:
            raise ValueError(
                "isd_m must be positive."
            )

        if self.bs_height_m <= 0.0:
            raise ValueError(
                "bs_height_m must be positive."
            )

        if self.ut_height_m <= 0.0:
            raise ValueError(
                "ut_height_m must be positive."
            )

        if self.ut_speed_kmh < 0.0:
            raise ValueError(
                "ut_speed_kmh cannot be negative."
            )


        if self.temporal_radio_mode not in {
            "independent",
            "velocity_window",
        }:
            raise ValueError(
                "temporal_radio_mode must be "
                "'independent' or "
                "'velocity_window'."
            )

        if self.temporal_window_ttis <= 0:
            raise ValueError(
                "temporal_window_ttis must be "
                "positive."
            )

        if self.tti_duration_s <= 0.0:
            raise ValueError(
                "tti_duration_s must be positive."
            )

        if (
            self.temporal_max_displacement_m
            <= 0.0
        ):
            raise ValueError(
                "temporal_max_displacement_m "
                "must be positive."
            )



@dataclass(frozen=True)
class ChunkedSionnaPPOContext:
    """
    Persistent information shared across TTIs.

    The expensive full paper-MIMO tensor is NOT
    stored here.

    Stored:
        420-UE topology
        lightweight cell association
        selected real cell IDs
        persistent global UE IDs per selected cell
        PHY scalar/configuration values
    """

    config: ChunkedSionnaPPOConfig

    topology_config: TopologyConfig

    topology: TopologyData

    association: CellAssociationData

    selected_cell_indices: tuple[
        int,
        ...
    ]

    global_ue_indices_by_stream: tuple[
        torch.Tensor,
        ...
    ]

    tx_power_per_subcarrier_w: torch.Tensor

    noise_power_per_subcarrier_w: torch.Tensor

    link_adaptation_config: LinkAdaptationConfig

    rate_config: RateConfig

    #
    # PERFORMANCE ENGINEERING.
    #
    # One persistent paper-array Sionna runtime is
    # shared by every serving cell / UE microbatch.
    #
    # This changes object lifetime only. The channel
    # topology and deterministic per-microbatch seed
    # are still supplied for every generation.
    #
    channel_runtime: FrequencyChannelRuntime

    #
    # None in legacy independent mode.
    #
    # In Wave-4 temporal mode this runtime directly
    # generates multi-TTI velocity-driven Sionna
    # channel windows.
    #
    temporal_channel_runtime: (
        VelocityWindowFrequencyChannelRuntime
        | None
    ) = None


    @property
    def num_global_ues(
        self,
    ) -> int:
        return int(
            self.topology.ut_loc.shape[1]
        )


    @property
    def num_cells(
        self,
    ) -> int:
        return int(
            self.topology.bs_loc.shape[1]
        )


    @property
    def num_streams(
        self,
    ) -> int:
        return len(
            self.selected_cell_indices
        )



def build_chunked_sionna_ppo_context(
    *,
    config: (
        ChunkedSionnaPPOConfig
        | None
    ) = None,
) -> ChunkedSionnaPPOContext:
    """
    Build the complete paper-sized TRAINING topology
    without constructing the complete paper-MIMO H.

    Association pipeline:

        420-UE topology
            ->
        lightweight 1x1 UMa channel
            ->
        existing minimum-pathloss association
            ->
        discard lightweight channel

    Why lightweight association?

        The current reproduction's association metric
        is channel-averaged pathloss.

        We therefore do not need a 192-port gNB
        channel merely to determine serving BS.

    OPEN-REPRODUCTION:
        exact Nokia association procedure is not
        public.

    MEMORY-SAFE REPRODUCTION:
        lightweight SISO association is used only
        for determining the persistent serving cell.
    """

    if config is None:
        config = (
            ChunkedSionnaPPOConfig()
        )

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,

        #
        # PAPER-SPECIFIED TRAINING POPULATION:
        #
        #     21 x 20 = 420 UEs
        #
        num_ut_per_sector=(
            config.num_ut_per_sector
        ),

        scenario=config.scenario,

        isd_m=config.isd_m,

        bs_height_m=config.bs_height_m,

        ut_height_m=config.ut_height_m,

        ut_speed_kmh=config.ut_speed_kmh,

        seed=config.topology_seed,

        precision="single",

        device=config.device,
    )

    topology = generate_topology(
        topology_config
    )

    if (
        topology_config.num_cells
        != 21
    ):
        raise RuntimeError(
            "Expected exactly 21 cells."
        )

    expected_num_ues = (
        21
        * config.num_ut_per_sector
    )

    if (
        topology_config.num_ues
        != expected_num_ues
    ):
        raise RuntimeError(
            "Unexpected training UE population."
        )

    # ==========================================================
    # LIGHTWEIGHT FULL-POPULATION ASSOCIATION CHANNEL
    # ==========================================================

    association_channel_config = (
        ChannelConfig(
            carrier_frequency_hz=4.0e9,
            subcarrier_spacing_hz=30.0e3,

            num_rbs=config.num_rbs,

            subcarriers_per_rb=(
                config.subcarriers_per_rb
            ),

            num_ofdm_symbols=1,

            #
            # MEMORY-SAFE association only.
            #
            antenna_mode="sanity",

            direction="downlink",

            o2i_model="low",

            enable_pathloss=True,

            enable_shadow_fading=True,

            precision="single",

            device=config.device,

            seed=(
                config
                .association_channel_seed
            ),
        )
    )

    association_channel = (
        generate_frequency_channel(
            topology=topology,

            topology_config=(
                topology_config
            ),

            channel_config=(
                association_channel_config
            ),
        )
    )

    association = build_cell_association(
        h_freq=(
            association_channel.h_freq
        ),
        precision="single",
    )

    #
    # Do not retain the full-population association H.
    #
    del association_channel

    # ==========================================================
    # SELECT TRAINING STREAMS
    # ==========================================================

    ue_counts = (
        association
        .num_ues_per_cell[
            0
        ]
    )

    if (
        config.num_training_cells
        == topology_config.num_cells
    ):
        #
        # Full paper run:
        # preserve natural gNB order 0..20.
        #
        selected_cells = tuple(
            range(
                topology_config.num_cells
            )
        )

    else:
        #
        # Smoke run:
        # pick most populated cells so PF TDS has
        # meaningful candidate populations.
        #
        sorted_cells = torch.argsort(
            ue_counts,
            descending=True,
        )

        selected_cells = tuple(
            int(
                value.item()
            )
            for value
            in sorted_cells[
                :config.num_training_cells
            ]
        )

    global_ue_indices_by_stream = []

    for real_cell_index in (
        selected_cells
    ):
        cell_global_indices = torch.nonzero(
            association.serving_bs[
                0
            ]
            == real_cell_index,
            as_tuple=False,
        ).flatten()

        if cell_global_indices.numel() == 0:
            raise RuntimeError(
                "Selected PPO cell has no serving UEs."
            )

        global_ue_indices_by_stream.append(
            cell_global_indices
        )

    # ==========================================================
    # COMMON POWER / LA CONFIG
    # ==========================================================

    sinr_config = SINRConfig(
        tx_power_dbm=44.0,

        subcarrier_spacing_hz=(
            30.0e3
        ),

        temperature_k=294.0,

        receiver_noise_figure_db=None,

        precision="single",

        device=config.device,
    )

    num_subcarriers = (
        config.num_rbs
        * config.subcarriers_per_rb
    )

    tx_power_per_subcarrier_w = (
        compute_subcarrier_tx_power_w(
            config=sinr_config,
            num_subcarriers=(
                num_subcarriers
            ),
        )
    )

    noise_power_per_subcarrier_w = (
        compute_noise_power_per_subcarrier_w(
            config=sinr_config,
        )
    )

    link_adaptation_config = (
        LinkAdaptationConfig(
            num_rbgs=(
                config.num_rbgs
            ),

            subcarriers_per_rbg=(
                config
                .subcarriers_per_rb
            ),

            precision="single",

            device=config.device,
        )
    )

    rate_config = RateConfig(
        subcarriers_per_rbg=(
            config.subcarriers_per_rb
        ),
        device=config.device,
    )

    # ==========================================================
    # PERSISTENT PAPER-MIMO SIONNA RUNTIME
    # ==========================================================
    #
    # Previously every UE microbatch recreated:
    #
    #     PanelArray
    #     ResourceGrid
    #     UMa
    #     GenerateOFDMChannel
    #
    # The runtime now creates those objects once.
    #
    # Per-microbatch topology and seed are still
    # supplied independently below.
    #

    paper_channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,

        subcarrier_spacing_hz=30.0e3,

        num_rbs=(
            config.num_rbs
        ),

        subcarriers_per_rb=(
            config.subcarriers_per_rb
        ),

        num_ofdm_symbols=1,

        antenna_mode="paper",

        direction="downlink",

        o2i_model="low",

        enable_pathloss=True,

        enable_shadow_fading=True,

        precision="single",

        device=config.device,

        #
        # The actual realization seed is supplied
        # separately to runtime.generate().
        #
        seed=config.mimo_channel_seed,
    )

    channel_runtime = (
        FrequencyChannelRuntime(
            config=(
                paper_channel_config
            )
        )
    )

    temporal_channel_runtime = None

    if (
        config.temporal_radio_mode
        == "velocity_window"
    ):
        temporal_channel_runtime = (
            VelocityWindowFrequencyChannelRuntime(
                config=paper_channel_config
            )
        )

    return ChunkedSionnaPPOContext(
        config=config,

        topology_config=(
            topology_config
        ),

        topology=topology,

        association=association,

        selected_cell_indices=(
            selected_cells
        ),

        global_ue_indices_by_stream=tuple(
            global_ue_indices_by_stream
        ),

        tx_power_per_subcarrier_w=(
            tx_power_per_subcarrier_w
        ),

        noise_power_per_subcarrier_w=(
            noise_power_per_subcarrier_w
        ),

        link_adaptation_config=(
            link_adaptation_config
        ),

        rate_config=rate_config,

        channel_runtime=(
            channel_runtime
        ),

        temporal_channel_runtime=(
            temporal_channel_runtime
        ),
    )



def _temporal_window_coordinates(
    *,
    tti_index: int,
    window_ttis: int,
) -> tuple[
    int,
    int,
    int,
]:
    """
    Map absolute TTI ->

        window index,
        window start TTI,
        offset inside window.

    Example, W=8:

        t=0  -> (0, 0, 0)
        t=7  -> (0, 0, 7)
        t=8  -> (1, 8, 0)
        t=11 -> (1, 8, 3)
    """

    if tti_index < 0:
        raise ValueError(
            "tti_index must be non-negative."
        )

    if window_ttis <= 0:
        raise ValueError(
            "window_ttis must be positive."
        )

    window_index = (
        tti_index
        // window_ttis
    )

    window_start_tti = (
        window_index
        * window_ttis
    )

    offset = (
        tti_index
        - window_start_tti
    )

    return (
        window_index,
        window_start_tti,
        offset,
    )


def _temporal_microbatch_channel_seed(
    *,
    context: ChunkedSionnaPPOContext,
    window_index: int,
    real_cell_index: int,
    microbatch_index: int,
) -> int:
    """
    Deterministic seed shared by every TTI inside
    one temporal radio window.

    This is the critical difference from the old
    independent realization path.

    Old:
        seed depends on TTI.

    New:
        seed depends on temporal WINDOW.

    Therefore offsets 0..W-1 address time samples
    from the same Sionna realization.
    """

    if window_index < 0:
        raise ValueError(
            "window_index must be non-negative."
        )

    if not (
        0
        <= real_cell_index
        < context.num_cells
    ):
        raise ValueError(
            "real_cell_index outside topology."
        )

    if microbatch_index < 0:
        raise ValueError(
            "microbatch_index must be non-negative."
        )

    seed_stride = (
        context.num_global_ues
        + 1
    )

    window_cell_ordinal = (
        window_index
        * context.num_cells
        + real_cell_index
    )

    return int(
        context.config.mimo_channel_seed
        + window_cell_ordinal
        * seed_stride
        + microbatch_index
    )




def _identity_stable_independent_ue_seed(
    *,
    context: ChunkedSionnaPPOContext,
    tti_index: int,
    global_ue_index: int,
) -> int:
    """
    Independent-radio seed owned by GLOBAL UE identity.

    Association/local scheduler position does not
    enter this seed.
    """

    if tti_index < 0:
        raise ValueError(
            "tti_index must be non-negative."
        )

    if not (
        0
        <= global_ue_index
        < context.num_global_ues
    ):
        raise ValueError(
            "global_ue_index outside topology."
        )

    seed_stride = (
        context.num_global_ues
        + 1
    )

    return int(
        context
        .config
        .mimo_channel_seed

        + tti_index
        * seed_stride

        + global_ue_index
    )


def _identity_stable_temporal_ue_seed(
    *,
    context: ChunkedSionnaPPOContext,
    window_index: int,
    global_ue_index: int,
) -> int:
    """
    Temporal-window seed owned by GLOBAL UE identity.

    Every TTI within one temporal window addresses
    the same underlying Sionna realization.

    A handover does not change this seed.
    """

    if window_index < 0:
        raise ValueError(
            "window_index must be non-negative."
        )

    if not (
        0
        <= global_ue_index
        < context.num_global_ues
    ):
        raise ValueError(
            "global_ue_index outside topology."
        )

    seed_stride = (
        context.num_global_ues
        + 1
    )

    return int(
        context
        .config
        .mimo_channel_seed

        + window_index
        * seed_stride

        + global_ue_index
    )


def _generate_microbatch_h_freq(
    *,
    context: ChunkedSionnaPPOContext,
    tti_index: int,
    real_cell_index: int,
    microbatch_index: int,
    microbatch_global_ue_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Generate exactly one TTI-shaped paper-MIMO H.

    Returned shape is identical in BOTH modes:

        [
            batch,
            microbatch UE,
            RX,
            all 21 BS,
            TX,
            1 time sample,
            subcarrier,
        ]

    Modes
    -----

    independent:
        Preserve the original implementation exactly.

    velocity_window:
        1. Find this TTI's temporal window.
        2. Move UE geometry to the START of the window.
        3. Generate W velocity-driven Sionna samples.
        4. Return only the current TTI offset.

    MEMORY POLICY
    -------------
    Production generates W correlated CIR samples,
    selects the requested CIR time sample, and only
    then performs OFDM/subcarrier expansion.

    The complete W-sample paper-MIMO frequency tensor
    is never materialized in this path.
    """

    mode = (
        context
        .config
        .temporal_radio_mode
    )

    identity_stable = bool(
        getattr(
            context.config,
            "identity_stable_ue_channel_rng",
            False,
        )
    )

    identity_global_ue = None

    if identity_stable:

        if int(
            microbatch_global_ue_indices
            .numel()
        ) != 1:
            raise ValueError(
                "Identity-stable radio generation "
                "requires exactly one UE per "
                "microbatch."
            )

        identity_global_ue = int(
            microbatch_global_ue_indices[
                0
            ].item()
        )

    if mode == "independent":

        microbatch_topology = (
            subset_topology_ues(
                topology=context.topology,

                global_ue_indices=(
                    microbatch_global_ue_indices
                ),
            )
        )

        if identity_stable:

            assert (
                identity_global_ue
                is not None
            )

            channel_seed = (
                _identity_stable_independent_ue_seed(
                    context=context,

                    tti_index=tti_index,

                    global_ue_index=(
                        identity_global_ue
                    ),
                )
            )

        else:

            channel_seed = (
                _microbatch_channel_seed(
                    context=context,

                    tti_index=tti_index,

                    real_cell_index=(
                        real_cell_index
                    ),

                    microbatch_index=(
                        microbatch_index
                    ),
                )
            )

        channel = (
            context
            .channel_runtime
            .generate(
                topology=(
                    microbatch_topology
                ),

                seed=channel_seed,

                batch_size=(
                    context
                    .topology_config
                    .batch_size
                ),
            )
        )

        return channel.h_freq


    if mode != "velocity_window":
        raise RuntimeError(
            "Unexpected temporal radio mode."
        )


    (
        window_index,
        window_start_tti,
        offset,
    ) = _temporal_window_coordinates(
        tti_index=tti_index,

        window_ttis=(
            context
            .config
            .temporal_window_ttis
        ),
    )


    #
    # Move persistent UE geometry only to the
    # beginning of the radio window.
    #
    # Sionna velocity then evolves the fast channel
    # samples INSIDE that short window.
    #
    mobility_snapshot = (
        build_mobility_snapshot(
            initial_topology=(
                context.topology
            ),

            config=MobilityConfig(
                tti_duration_s=(
                    context
                    .config
                    .tti_duration_s
                ),

                trajectory_mode=(
                    "static"
                    if (
                        context
                        .config
                        .ut_speed_kmh
                        == 0.0
                    )
                    else
                    "constant_velocity"
                ),

                association_mode="fixed",

                max_horizontal_displacement_m=(
                    context
                    .config
                    .temporal_max_displacement_m
                ),
            ),

            tti_index=(
                window_start_tti
            ),
        )
    )


    microbatch_topology = (
        subset_topology_ues(
            topology=(
                mobility_snapshot
                .topology
            ),

            global_ue_indices=(
                microbatch_global_ue_indices
            ),
        )
    )


    runtime = (
        context
        .temporal_channel_runtime
    )

    if runtime is None:
        raise RuntimeError(
            "velocity_window mode requires "
            "temporal_channel_runtime."
        )


    if identity_stable:

        assert (
            identity_global_ue
            is not None
        )

        channel_seed = (
            _identity_stable_temporal_ue_seed(
                context=context,

                window_index=window_index,

                global_ue_index=(
                    identity_global_ue
                ),
            )
        )

    else:

        channel_seed = (
            _temporal_microbatch_channel_seed(
                context=context,

                window_index=window_index,

                real_cell_index=(
                    real_cell_index
                ),

                microbatch_index=(
                    microbatch_index
                ),
            )
        )


    #
    # MEMORY-SAFE TEMPORAL PATH
    # -------------------------
    #
    # Generate the complete W-sample correlated CIR
    # realization, but select the requested temporal
    # coefficient BEFORE OFDM/subcarrier expansion.
    #
    # Therefore the production scheduler never
    # materializes a W-sample full paper-MIMO
    # frequency tensor merely to consume one TTI.
    #
    return runtime.generate_tti_slice(
        topology=microbatch_topology,

        seed=channel_seed,

        batch_size=(
            context
            .topology_config
            .batch_size
        ),

        num_ttis=(
            context
            .config
            .temporal_window_ttis
        ),

        tti_duration_s=(
            context
            .config
            .tti_duration_s
        ),

        tti_offset=offset,
    )


def _map_candidate_global_to_local_phy(
    *,
    candidate_global_ue_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    cell_global_ue_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Map PF candidate GLOBAL identities into the
    local UE positions of one cell's MIMO tensor.

    Example:

        cell global UE list:
            [17, 52, 84, 103]

        PF candidates:
            [84, 17, -1, ...]

        local PHY indices:
            [2, 0, -1, ...]

    Padding remains -1.
    """

    if tuple(
        candidate_global_ue_indices.shape
    ) != tuple(
        candidate_valid_mask.shape
    ):
        raise ValueError(
            "Candidate IDs and validity mask disagree."
        )

    output = torch.full(
        candidate_global_ue_indices.shape,
        fill_value=-1,
        dtype=torch.long,
        device=(
            candidate_global_ue_indices
            .device
        ),
    )

    valid_candidate_ids = (
        candidate_global_ue_indices[
            candidate_valid_mask
        ]
    )

    if valid_candidate_ids.numel() == 0:
        return output

    matches = (
        valid_candidate_ids[
            :,
            None,
        ]
        ==
        cell_global_ue_indices[
            None,
            :,
        ]
    )

    matches_per_candidate = (
        matches.sum(
            dim=1
        )
    )

    if not torch.all(
        matches_per_candidate == 1
    ):
        raise ValueError(
            "Every valid PF candidate must appear "
            "exactly once in the local cell PHY UE list."
        )

    local_indices = torch.argmax(
        matches.to(
            dtype=torch.int64
        ),
        dim=1,
    )

    output[
        candidate_valid_mask
    ] = local_indices

    return output

@dataclass(frozen=True)
class _CellRadioMicrobatch:
    """
    Useful outputs retained from one UE microbatch.

    Scheduler-facing tensors are later concatenated
    across ALL microbatches before PF-TDS.

    H / rank / RX combiners remain chunked only until
    PF-TDS selects <=10 candidates.
    """

    serving_start_index: int
    serving_stop_index: int

    global_ue_indices: torch.Tensor

    td_instantaneous_rate_bps: torch.Tensor

    rank: torch.Tensor

    wideband_cqi: torch.Tensor

    subband_cqi: torch.Tensor

    precoder_directions: torch.Tensor

    h_freq: torch.Tensor

    recommended_rank: torch.Tensor

    rx_combiners: torch.Tensor

    csi_subcarrier_index: int


def _partition_cell_global_ue_indices(
    *,
    cell_global_ue_indices: torch.Tensor,
    ue_microbatch_size: int,
) -> tuple[
    tuple[
        int,
        int,
        torch.Tensor,
    ],
    ...,
]:
    """
    Partition one serving-cell UE vector while
    preserving exact global UE identity and order.

    OPEN-REPRODUCTION ENGINEERING:
        this affects Sionna memory partitioning only.
    """

    if cell_global_ue_indices.ndim != 1:
        raise ValueError(
            "cell_global_ue_indices must have shape "
            "[serving_ue]."
        )

    if ue_microbatch_size <= 0:
        raise ValueError(
            "ue_microbatch_size must be positive."
        )

    num_serving_ues = int(
        cell_global_ue_indices.numel()
    )

    if num_serving_ues == 0:
        raise ValueError(
            "A PPO stream must contain at least one "
            "serving UE."
        )

    partitions: list[
        tuple[
            int,
            int,
            torch.Tensor,
        ]
    ] = []

    for start_index in range(
        0,
        num_serving_ues,
        ue_microbatch_size,
    ):
        stop_index = min(
            start_index
            + ue_microbatch_size,
            num_serving_ues,
        )

        partitions.append(
            (
                start_index,
                stop_index,
                cell_global_ue_indices[
                    start_index:stop_index
                ],
            )
        )

    return tuple(
        partitions
    )


def _microbatch_channel_seed(
    *,
    context: ChunkedSionnaPPOContext,
    tti_index: int,
    real_cell_index: int,
    microbatch_index: int,
) -> int:
    """
    Deterministic channel seed for one UE microbatch.

    OPEN-REPRODUCTION ENGINEERING:
        the Nokia paper does not specify how a
        Sionna reproduction should divide RNG streams
        across UE microbatches.

    For a fixed:
        topology
        TTI
        real cell
        microbatch size/order

    this returns the same seed on every run.
    """

    if tti_index < 0:
        raise ValueError(
            "tti_index must be non-negative."
        )

    if not (
        0
        <= real_cell_index
        < context.num_cells
    ):
        raise ValueError(
            "real_cell_index is outside the topology."
        )

    if microbatch_index < 0:
        raise ValueError(
            "microbatch_index must be non-negative."
        )

    #
    # One TTI/cell combination receives a seed
    # namespace wider than the entire 420-UE
    # population, so nearby microchunks cannot
    # collide.
    #
    seed_stride = (
        context.num_global_ues
        + 1
    )

    cell_tti_ordinal = (
        tti_index
        * context.num_cells
        + real_cell_index
    )

    return int(
        context
        .config
        .mimo_channel_seed

        + cell_tti_ordinal
        * seed_stride

        + microbatch_index
    )


def _combine_cell_radio_microbatches(
    *,
    cell_global_ue_indices: torch.Tensor,
    microbatches: tuple[
        _CellRadioMicrobatch,
        ...,
    ],
) -> OneLDSCellTTIObservation:
    """
    Reconstruct the complete scheduler observation.

    IMPORTANT:
        this happens BEFORE PF-TDS.

    Therefore, if a cell owns 27 UEs, PF-TDS still
    receives one 27-UE observation.
    """

    if len(
        microbatches
    ) == 0:
        raise ValueError(
            "microbatches cannot be empty."
        )

    recombined_global_ue_indices = (
        torch.cat(
            tuple(
                microbatch
                .global_ue_indices

                for microbatch
                in microbatches
            ),
            dim=0,
        )
    )

    if not torch.equal(
        recombined_global_ue_indices,
        cell_global_ue_indices,
    ):
        raise RuntimeError(
            "UE microbatch recombination changed "
            "global UE identity or order."
        )

    num_serving_ues = int(
        cell_global_ue_indices.numel()
    )

    return OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            cell_global_ue_indices
        ),

        serving_ue_valid_mask=(
            torch.ones(
                (
                    num_serving_ues,
                ),
                dtype=torch.bool,
                device=(
                    cell_global_ue_indices
                    .device
                ),
            )
        ),

        td_instantaneous_rate_bps=(
            torch.cat(
                tuple(
                    microbatch
                    .td_instantaneous_rate_bps

                    for microbatch
                    in microbatches
                ),
                dim=0,
            )
        ),

        rank=(
            torch.cat(
                tuple(
                    microbatch.rank

                    for microbatch
                    in microbatches
                ),
                dim=0,
            )
        ),

        #
        # TrafficBufferManager replaces this before
        # PF-TDS, exactly as in the current path.
        #
        dl_buffer=torch.zeros(
            (
                num_serving_ues,
            ),
            dtype=torch.float32,
            device=(
                cell_global_ue_indices
                .device
            ),
        ),

        wideband_cqi=(
            torch.cat(
                tuple(
                    microbatch.wideband_cqi

                    for microbatch
                    in microbatches
                ),
                dim=0,
            )
        ),

        subband_cqi=(
            torch.cat(
                tuple(
                    microbatch.subband_cqi

                    for microbatch
                    in microbatches
                ),
                dim=0,
            )
        ),

        precoder_directions=(
            torch.cat(
                tuple(
                    microbatch
                    .precoder_directions

                    for microbatch
                    in microbatches
                ),
                dim=0,
            )
        ),
    )


def _compact_candidate_phy_from_microbatches(
    *,
    prepared: PreparedOneLDSCellTTI,
    cell_global_ue_indices: torch.Tensor,
    microbatches: tuple[
        _CellRadioMicrobatch,
        ...,
    ],
    real_cell_index: int,
    context: ChunkedSionnaPPOContext,
) -> PPOPhysicalScoreInputs:
    """
    Gather PF-TDS candidates directly from retained
    UE microbatches.

    Importantly, we NEVER concatenate a full-cell H.

    Input H storage:

        chunk 0
        chunk 1
        ...
        chunk N

    Output H:

        [1,
         candidate,
         RX,
         21 BS,
         TX,
         OFDM symbol,
         subcarrier]
    """

    if len(
        microbatches
    ) == 0:
        raise ValueError(
            "microbatches cannot be empty."
        )

    candidate_valid_mask = (
        prepared
        .candidate_valid_mask
    )

    local_candidate_indices = (
        _map_candidate_global_to_local_phy(
            candidate_global_ue_indices=(
                prepared
                .candidate_global_ue_indices
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
            cell_global_ue_indices=(
                cell_global_ue_indices
            ),
        )
    )

    first_microbatch = (
        microbatches[0]
    )

    csi_subcarrier_index = (
        first_microbatch
        .csi_subcarrier_index
    )

    for microbatch in (
        microbatches[1:]
    ):
        if (
            microbatch
            .csi_subcarrier_index
            != csi_subcarrier_index
        ):
            raise RuntimeError(
                "UE microbatches disagree on the "
                "CSI subcarrier index."
            )

    h_rows: list[
        torch.Tensor
    ] = []

    rank_rows: list[
        torch.Tensor
    ] = []

    rx_rows: list[
        torch.Tensor
    ] = []

    num_candidates = int(
        candidate_valid_mask.shape[0]
    )

    for candidate_slot in range(
        num_candidates
    ):
        #
        # Padded PF candidate slot.
        #
        if not bool(
            candidate_valid_mask[
                candidate_slot
            ].item()
        ):
            h_rows.append(
                torch.zeros_like(
                    first_microbatch
                    .h_freq[
                        :,
                        0:1,
                    ]
                )
            )

            rank_rows.append(
                torch.zeros_like(
                    first_microbatch
                    .recommended_rank[
                        :,
                        0:1,
                    ]
                )
            )

            rx_rows.append(
                torch.zeros_like(
                    first_microbatch
                    .rx_combiners[
                        :,
                        0:1,
                    ]
                )
            )

            continue

        serving_local_index = int(
            local_candidate_indices[
                candidate_slot
            ].item()
        )

        owning_microbatch: (
            _CellRadioMicrobatch
            | None
        ) = None

        for microbatch in (
            microbatches
        ):
            if (
                microbatch
                .serving_start_index

                <= serving_local_index

                < microbatch
                .serving_stop_index
            ):
                owning_microbatch = (
                    microbatch
                )

                break

        if owning_microbatch is None:
            raise RuntimeError(
                "PF candidate is not covered by "
                "any retained UE microbatch."
            )

        microbatch_local_index = (
            serving_local_index

            - owning_microbatch
            .serving_start_index
        )

        h_rows.append(
            owning_microbatch
            .h_freq[
                :,
                microbatch_local_index:
                microbatch_local_index + 1,
            ]
        )

        rank_rows.append(
            owning_microbatch
            .recommended_rank[
                :,
                microbatch_local_index:
                microbatch_local_index + 1,
            ]
        )

        rx_rows.append(
            owning_microbatch
            .rx_combiners[
                :,
                microbatch_local_index:
                microbatch_local_index + 1,
            ]
        )

    candidate_h_freq = (
        torch.cat(
            tuple(
                h_rows
            ),
            dim=1,
        )
    )

    candidate_recommended_rank = (
        torch.cat(
            tuple(
                rank_rows
            ),
            dim=1,
        )
    )

    candidate_rx_combiners = (
        torch.cat(
            tuple(
                rx_rows
            ),
            dim=1,
        )
    )

    #
    # This is the physical-model invariant we must
    # not break while fixing memory.
    #
    if int(
        candidate_h_freq.shape[3]
    ) != context.num_cells:
        raise RuntimeError(
            "Candidate compaction lost one or more "
            "BS links."
        )

    #
    # Compact PHY row namespace:
    #
    # valid candidate 0 -> physical H row 0
    # valid candidate 1 -> physical H row 1
    # ...
    #
    # padding -> -1
    #
    candidate_physical_ue_indices = (
        torch.arange(
            num_candidates,
            dtype=torch.long,
            device=(
                candidate_valid_mask
                .device
            ),
        )
    )

    candidate_physical_ue_indices = (
        torch.where(
            candidate_valid_mask,

            candidate_physical_ue_indices,

            torch.full_like(
                candidate_physical_ue_indices,
                fill_value=-1,
            ),
        )
    )

    return PPOPhysicalScoreInputs(
        #
        # Persistent global scheduler identity.
        #
        candidate_global_ue_indices=(
            prepared
            .candidate_global_ue_indices
        ),

        #
        # Candidate-only physical storage mapping.
        #
        candidate_physical_ue_indices=(
            candidate_physical_ue_indices
        ),

        h_freq=(
            candidate_h_freq
        ),

        serving_cell_index=(
            real_cell_index
        ),

        recommended_rank=(
            candidate_recommended_rank
        ),

        rx_combiners=(
            candidate_rx_combiners
        ),

        csi_subcarrier_index=(
            csi_subcarrier_index
        ),

        subcarriers_per_rbg=(
            context
            .config
            .subcarriers_per_rb
        ),

        tx_power_per_subcarrier_w=(
            context
            .tx_power_per_subcarrier_w
        ),

        noise_power_per_subcarrier_w=(
            context
            .noise_power_per_subcarrier_w
        ),

        link_adaptation_config=(
            context
            .link_adaptation_config
        ),

        rate_config=(
            context.rate_config
        ),

        batch_index=0,
    )

# def _build_cell_training_inputs(
#     *,
#     context: ChunkedSionnaPPOContext,
#     tti_index: int,
#     stream_index: int,
# ) -> PPOTrainingTTIInputs:
#     """
#     Build one PPO stream's real Sionna inputs.

#     Critically:

#         local UE count
#             ~20-ish

#         BS count
#             always 21

#     Therefore h_freq remains physically multicell:

#         [1,
#          local UE,
#          4 Rx,
#          21 BS,
#          192 Tx,
#          1 symbol,
#          216 SC]

#     but no complete 420-UE paper MIMO tensor exists.
#     """

#     if not (
#         0
#         <= stream_index
#         < context.num_streams
#     ):
#         raise ValueError(
#             "stream_index is outside the configured "
#             "PPO stream range."
#         )

#     if tti_index < 0:
#         raise ValueError(
#             "tti_index must be non-negative."
#         )

#     real_cell_index = (
#         context
#         .selected_cell_indices[
#             stream_index
#         ]
#     )

#     cell_global_ue_indices = (
#         context
#         .global_ue_indices_by_stream[
#             stream_index
#         ]
#     )

#     # ==========================================================
#     # SUBSET FULL 420-UE TOPOLOGY
#     # ==========================================================

#     local_topology = subset_topology_ues(
#         topology=context.topology,

#         global_ue_indices=(
#             cell_global_ue_indices
#         ),
#     )

#     # ==========================================================
#     # PAPER MIMO FOR ONLY THIS CELL'S SERVING UEs
#     #
#     # IMPORTANT:
#     # ALL 21 BSs remain present.
#     # ==========================================================

#     channel_seed = (
#         context
#         .config
#         .mimo_channel_seed

#         + (
#             tti_index
#             * context.num_cells
#         )

#         + real_cell_index
#     )

#     channel_config = ChannelConfig(
#         carrier_frequency_hz=4.0e9,

#         subcarrier_spacing_hz=30.0e3,

#         num_rbs=(
#             context.config.num_rbs
#         ),

#         subcarriers_per_rb=(
#             context
#             .config
#             .subcarriers_per_rb
#         ),

#         num_ofdm_symbols=1,

#         antenna_mode="paper",

#         direction="downlink",

#         o2i_model="low",

#         enable_pathloss=True,

#         enable_shadow_fading=True,

#         precision="single",

#         device=(
#             context.config.device
#         ),

#         seed=channel_seed,
#     )

#     channel = generate_frequency_channel(
#         topology=local_topology,

#         topology_config=(
#             context.topology_config
#         ),

#         channel_config=(
#             channel_config
#         ),
#     )

#     #
#     # Retain ONLY H from ChannelData.
#     #
#     # Do not intentionally keep the Sionna channel
#     # model object alive after this function.
#     #
#     h_freq = channel.h_freq

#     num_local_ues = int(
#         h_freq.shape[1]
#     )

#     if num_local_ues != int(
#         cell_global_ue_indices.numel()
#     ):
#         raise RuntimeError(
#             "Local MIMO channel and cell UE mapping "
#             "have different UE dimensions."
#         )

#     if int(
#         h_freq.shape[3]
#     ) != context.num_cells:
#         raise RuntimeError(
#             "Cell chunk lost one or more BS links."
#         )

#     # ==========================================================
#     # FIXED ASSOCIATION FOR THESE LOCAL UEs
#     # ==========================================================

#     local_serving_bs = torch.full(
#         (
#             1,
#             num_local_ues,
#         ),
#         fill_value=(
#             real_cell_index
#         ),
#         dtype=torch.long,
#         device=h_freq.device,
#     )

#     # ==========================================================
#     # LOCAL RI / CSI
#     # ==========================================================

#     rank_data = (
#         compute_ideal_svd_rank_diagnostic(
#             h_freq=h_freq,

#             serving_bs=(
#                 local_serving_bs
#             ),

#             num_rbgs=(
#                 context
#                 .config
#                 .num_rbgs
#             ),

#             tx_power_per_subcarrier_w=(
#                 context
#                 .tx_power_per_subcarrier_w
#             ),

#             noise_power_per_subcarrier_w=(
#                 context
#                 .noise_power_per_subcarrier_w
#             ),
#         )
#     )

#     serving_mimo = (
#         extract_serving_mimo_rbg_channel(
#             h_freq=h_freq,

#             serving_bs=(
#                 local_serving_bs
#             ),

#             num_rbgs=(
#                 context
#                 .config
#                 .num_rbgs
#             ),
#         )
#     )

#     csi_data = compute_ideal_svd_csi(
#         h_serving_rbg=(
#             serving_mimo
#             .h_serving_rbg
#         ),

#         rank1_rbg_score=(
#             rank_data
#             .rank1_rbg_spectral_efficiency
#         ),

#         rank2_rbg_score=(
#             rank_data
#             .rank2_rbg_spectral_efficiency
#         ),
#     )

#     # ==========================================================
#     # SU-MIMO REPORTS FOR PF TDS + CQI STATE
#     # ==========================================================

#     su_report = build_single_user_phy_reports(
#         h_freq=h_freq,

#         serving_bs=(
#             local_serving_bs
#         ),

#         recommended_rank=(
#             csi_data
#             .recommended_rank
#         ),

#         precoder_directions=(
#             csi_data
#             .precoder_directions
#         ),

#         num_rbgs=(
#             context.config.num_rbgs
#         ),

#         subcarriers_per_rbg=(
#             context
#             .config
#             .subcarriers_per_rb
#         ),

#         tx_power_per_subcarrier_w=(
#             context
#             .tx_power_per_subcarrier_w
#         ),

#         noise_power_per_subcarrier_w=(
#             context
#             .noise_power_per_subcarrier_w
#         ),

#         link_adaptation_config=(
#             context
#             .link_adaptation_config
#         ),

#         rate_config=(
#             context.rate_config
#         ),
#     )

#     local_rbg_rate_bps = (
#         su_report
#         .rate
#         .target_compliant_rate_bps[
#             0
#         ]
#     )

#     td_instantaneous_rate_bps = (
#         local_rbg_rate_bps.sum(
#             dim=-1
#         )
#     )

#     local_valid_mask = torch.ones(
#         (
#             num_local_ues,
#         ),
#         dtype=torch.bool,
#         device=h_freq.device,
#     )

#     cqi = build_cqi_surrogate(
#         mcs_index=(
#             su_report
#             .link_adaptation
#             .mcs_index[
#                 0
#             ]
#         ),

#         meets_bler_target=(
#             su_report
#             .link_adaptation
#             .meets_bler_target[
#                 0
#             ]
#         ),

#         candidate_valid_mask=(
#             local_valid_mask
#         ),

#         config=CQISurrogateConfig(),
#     )

#     observation = (
#         OneLDSCellTTIObservation(
#             #
#             # GLOBAL / persistent simulator identities.
#             #
#             serving_global_ue_indices=(
#                 cell_global_ue_indices
#             ),

#             serving_ue_valid_mask=(
#                 local_valid_mask
#             ),

#             td_instantaneous_rate_bps=(
#                 td_instantaneous_rate_bps
#             ),

#             rank=(
#                 csi_data
#                 .recommended_rank[
#                     0
#                 ]
#             ),

#             #
#             # Replaced by TrafficBufferManager before
#             # PF TDS.
#             #
#             dl_buffer=torch.zeros(
#                 (
#                     num_local_ues,
#                 ),
#                 dtype=torch.float32,
#                 device=h_freq.device,
#             ),

#             wideband_cqi=(
#                 cqi
#                 .wideband_cqi
#                 .to(
#                     dtype=torch.float32
#                 )
#             ),

#             subband_cqi=(
#                 cqi
#                 .subband_cqi
#                 .to(
#                     dtype=torch.float32
#                 )
#             ),

#             precoder_directions=(
#                 csi_data
#                 .precoder_directions[
#                     0
#                 ]
#             ),
#         )
#     )

#     # ==========================================================
#     # CANDIDATE-SPECIFIC PHYSICAL BUILDER
#     # ==========================================================


#     # ==========================================================
#     # TEMPORARY FULL-CELL PHY HOLDER
#     #
#     # The full serving-cell MIMO tensors are needed only
#     # until PF-TDS has selected its <=10 candidates.
#     #
#     # physical_inputs_builder() will:
#     #
#     #     full serving-cell PHY
#     #             ↓
#     #     PF-selected candidate rows
#     #             ↓
#     #     compact candidate PHY
#     #
#     # and then drop the references to the full-cell tensors.
#     #
#     # This is an OPEN-REPRODUCTION ENGINEERING optimization.
#     # Scheduler/PHY mathematics are unchanged.
#     # ==========================================================

#     csi_subcarrier_index = (
#         csi_data.csi_subcarrier_index
#     )

#     full_phy_holder = {
#         "h_freq": h_freq,

#         "recommended_rank": (
#             csi_data.recommended_rank
#         ),

#         "rx_combiners": (
#             csi_data.rx_combiners
#         ),

#         #
#         # If the builder is called a second time for
#         # the exact same prepared candidate set, reuse
#         # the already compacted object.
#         #
#         "materialized": None,
#     }

#     def physical_inputs_builder(
#         prepared: PreparedOneLDSCellTTI,
#     ) -> PPOPhysicalScoreInputs:
#         """
#         Materialize only the PF-TDS candidate PHY.

#         Before this call:

#             h_freq UE dimension
#                 =
#             every UE associated with this cell

#         After this call:

#             physical_inputs.h_freq UE dimension
#                 =
#             num_candidates

#         Normally:

#             <= 10

#         Persistent/global UE identities remain unchanged.
#         """

#         existing = (
#             full_phy_holder[
#                 "materialized"
#             ]
#         )

#         if existing is not None:
#             if not torch.equal(
#                 existing
#                 .candidate_global_ue_indices,
#                 prepared
#                 .candidate_global_ue_indices,
#             ):
#                 raise RuntimeError(
#                     "Candidate PHY was already "
#                     "materialized for a different "
#                     "PF-TDS candidate ordering."
#                 )

#             return existing


#         # ------------------------------------------------------
#         # Global candidate identity
#         #       ->
#         # row inside this serving-cell PHY tensor.
#         # ------------------------------------------------------

#         local_candidate_indices = (
#             _map_candidate_global_to_local_phy(
#                 candidate_global_ue_indices=(
#                     prepared
#                     .candidate_global_ue_indices
#                 ),

#                 candidate_valid_mask=(
#                     prepared
#                     .candidate_valid_mask
#                 ),

#                 cell_global_ue_indices=(
#                     cell_global_ue_indices
#                 ),
#             )
#         )

#         candidate_valid_mask = (
#             prepared
#             .candidate_valid_mask
#         )

#         num_candidates = int(
#             candidate_valid_mask.shape[0]
#         )


#         # ------------------------------------------------------
#         # Padding may contain -1.
#         #
#         # Replace padded positions temporarily with zero so
#         # index_select is legal. Those rows are zeroed below.
#         # ------------------------------------------------------

#         safe_local_indices = torch.where(
#             candidate_valid_mask,

#             local_candidate_indices,

#             torch.zeros_like(
#                 local_candidate_indices
#             ),
#         )


#         full_h_freq = (
#             full_phy_holder[
#                 "h_freq"
#             ]
#         )

#         full_recommended_rank = (
#             full_phy_holder[
#                 "recommended_rank"
#             ]
#         )

#         full_rx_combiners = (
#             full_phy_holder[
#                 "rx_combiners"
#             ]
#         )

#         if not isinstance(
#             full_h_freq,
#             torch.Tensor,
#         ):
#             raise RuntimeError(
#                 "Full-cell h_freq was released "
#                 "before candidate materialization."
#             )

#         if not isinstance(
#             full_recommended_rank,
#             torch.Tensor,
#         ):
#             raise RuntimeError(
#                 "Full-cell rank tensor was released "
#                 "before candidate materialization."
#             )

#         if not isinstance(
#             full_rx_combiners,
#             torch.Tensor,
#         ):
#             raise RuntimeError(
#                 "Full-cell RX combiners were released "
#                 "before candidate materialization."
#             )


#         # ------------------------------------------------------
#         # Compact H:
#         #
#         #     [1, serving UE, RX, BS, TX, sym, SC]
#         #
#         # becomes:
#         #
#         #     [1, candidate, RX, BS, TX, sym, SC]
#         #
#         # ALL 21 BS links are retained.
#         # ------------------------------------------------------

#         candidate_h_freq = (
#             full_h_freq
#             .index_select(
#                 1,
#                 safe_local_indices,
#             )
#         )

#         h_valid_mask = (
#             candidate_valid_mask[
#                 None,
#                 :,
#                 None,
#                 None,
#                 None,
#                 None,
#                 None,
#             ]
#         )

#         candidate_h_freq = torch.where(
#             h_valid_mask,
#             candidate_h_freq,
#             torch.zeros_like(
#                 candidate_h_freq
#             ),
#         )


#         # ------------------------------------------------------
#         # Rank tensor:
#         #
#         #     [batch, local UE]
#         #         ->
#         #     [batch, candidate]
#         # ------------------------------------------------------

#         candidate_recommended_rank = (
#             full_recommended_rank
#             .index_select(
#                 1,
#                 safe_local_indices,
#             )
#         )

#         candidate_recommended_rank = (
#             torch.where(
#                 candidate_valid_mask[
#                     None,
#                     :,
#                 ],
#                 candidate_recommended_rank,
#                 torch.zeros_like(
#                     candidate_recommended_rank
#                 ),
#             )
#         )


#         # ------------------------------------------------------
#         # RX combiners:
#         #
#         #     [batch, local UE, RBG, mode, RX]
#         #         ->
#         #     [batch, candidate, RBG, mode, RX]
#         # ------------------------------------------------------

#         candidate_rx_combiners = (
#             full_rx_combiners
#             .index_select(
#                 1,
#                 safe_local_indices,
#             )
#         )

#         rx_valid_mask = (
#             candidate_valid_mask[
#                 None,
#                 :,
#                 None,
#                 None,
#                 None,
#             ]
#         )

#         candidate_rx_combiners = (
#             torch.where(
#                 rx_valid_mask,
#                 candidate_rx_combiners,
#                 torch.zeros_like(
#                     candidate_rx_combiners
#                 ),
#             )
#         )


#         # ------------------------------------------------------
#         # The new compact tensor has candidate positions
#         #
#         #     0, 1, ..., K-1
#         #
#         # as its physical UE namespace.
#         #
#         # Padding remains -1.
#         # ------------------------------------------------------

#         candidate_physical_ue_indices = (
#             torch.arange(
#                 num_candidates,
#                 dtype=torch.long,
#                 device=(
#                     candidate_valid_mask
#                     .device
#                 ),
#             )
#         )

#         candidate_physical_ue_indices = (
#             torch.where(
#                 candidate_valid_mask,

#                 candidate_physical_ue_indices,

#                 torch.full_like(
#                     candidate_physical_ue_indices,
#                     fill_value=-1,
#                 ),
#             )
#         )


#         compact_inputs = (
#             PPOPhysicalScoreInputs(
#                 #
#                 # Persistent scheduler identity.
#                 #
#                 candidate_global_ue_indices=(
#                     prepared
#                     .candidate_global_ue_indices
#                 ),

#                 #
#                 # Compact candidate-only PHY.
#                 #
#                 candidate_physical_ue_indices=(
#                     candidate_physical_ue_indices
#                 ),

#                 h_freq=(
#                     candidate_h_freq
#                 ),

#                 serving_cell_index=(
#                     real_cell_index
#                 ),

#                 recommended_rank=(
#                     candidate_recommended_rank
#                 ),

#                 rx_combiners=(
#                     candidate_rx_combiners
#                 ),

#                 csi_subcarrier_index=(
#                     csi_subcarrier_index
#                 ),

#                 subcarriers_per_rbg=(
#                     context
#                     .config
#                     .subcarriers_per_rb
#                 ),

#                 tx_power_per_subcarrier_w=(
#                     context
#                     .tx_power_per_subcarrier_w
#                 ),

#                 noise_power_per_subcarrier_w=(
#                     context
#                     .noise_power_per_subcarrier_w
#                 ),

#                 link_adaptation_config=(
#                     context
#                     .link_adaptation_config
#                 ),

#                 rate_config=(
#                     context.rate_config
#                 ),

#                 batch_index=0,
#             )
#         )


#         # ------------------------------------------------------
#         # CRITICAL MEMORY RELEASE.
#         #
#         # The compact_inputs object now owns everything
#         # required by PPO / PF expert / RZF.
#         #
#         # The full serving-cell H is no longer needed.
#         # ------------------------------------------------------

#         full_phy_holder[
#             "materialized"
#         ] = compact_inputs

#         full_phy_holder[
#             "h_freq"
#         ] = None

#         full_phy_holder[
#             "recommended_rank"
#         ] = None

#         full_phy_holder[
#             "rx_combiners"
#         ] = None

#         return compact_inputs

#     return PPOTrainingTTIInputs(
#         observation=observation,

#         physical_inputs_builder=(
#             physical_inputs_builder
#         ),

#         packet_arrivals=None,
#     )

def _build_cell_training_inputs(
    *,
    context: ChunkedSionnaPPOContext,
    tti_index: int,
    stream_index: int,

    cell_global_ue_indices_override: (
        torch.Tensor | None
    ) = None,
) -> PPOTrainingTTIInputs:
    """
    Build one PPO stream's real Sionna inputs.

    Two-level memory layout:

        level 1:
            one serving cell at a time

        level 2:
            one small serving-UE microbatch at a time

    Every UE microbatch still contains all 21 BS links.

    PF-TDS runs only after scheduler-facing reports
    from every microbatch have been recombined.
    """

    if not (
        0
        <= stream_index
        < context.num_streams
    ):
        raise ValueError(
            "stream_index is outside the configured "
            "PPO stream range."
        )

    if tti_index < 0:
        raise ValueError(
            "tti_index must be non-negative."
        )

    real_cell_index = (
        context
        .selected_cell_indices[
            stream_index
        ]
    )

    if (
        cell_global_ue_indices_override
        is None
    ):

        cell_global_ue_indices = (
            context
            .global_ue_indices_by_stream[
                stream_index
            ]
        )

    else:

        cell_global_ue_indices = (
            cell_global_ue_indices_override
        )

        if (
            cell_global_ue_indices.ndim
            != 1
        ):
            raise ValueError(
                "Dynamic cell UE membership must "
                "have shape [UE]."
            )

        if (
            cell_global_ue_indices.dtype
            == torch.bool
            or torch.is_floating_point(
                cell_global_ue_indices
            )
        ):
            raise ValueError(
                "Dynamic cell UE identities must "
                "use an integer dtype."
            )

        if (
            cell_global_ue_indices.device
            != context.topology.ut_loc.device
        ):
            raise ValueError(
                "Dynamic cell UE identities are "
                "on the wrong device."
            )

        if int(
            cell_global_ue_indices.numel()
        ) == 0:
            raise ValueError(
                "Dynamic selected cell has zero "
                "serving UEs. Empty-cell scheduler "
                "support is not implemented yet."
            )

        if torch.any(
            cell_global_ue_indices < 0
        ):
            raise ValueError(
                "Dynamic cell UE identities cannot "
                "be negative."
            )

        if torch.any(
            cell_global_ue_indices
            >= context.num_global_ues
        ):
            raise ValueError(
                "Dynamic cell UE identity outside "
                "global topology."
            )

        if (
            torch.unique(
                cell_global_ue_indices
            ).numel()
            != cell_global_ue_indices.numel()
        ):
            raise ValueError(
                "Dynamic cell UE identities must "
                "be unique."
            )

        cell_global_ue_indices = (
            cell_global_ue_indices
            .detach()
            .clone()
            .to(
                dtype=torch.long,
            )
        )

    microbatch_specs = (
        _partition_cell_global_ue_indices(
            cell_global_ue_indices=(
                cell_global_ue_indices
            ),
            ue_microbatch_size=(
                context
                .config
                .ue_microbatch_size
            ),
        )
    )

    built_microbatches: list[
        _CellRadioMicrobatch
    ] = []

    # ==========================================================
    # PAPER MIMO, UE MICROBATCH BY UE MICROBATCH
    #
    # OPEN-REPRODUCTION ENGINEERING:
    #     only the UE batching / memory lifetime.
    #
    # Every generated UE still has H to ALL 21 BSs.
    # ==========================================================

    for (
        microbatch_index,
        (
            serving_start_index,
            serving_stop_index,
            microbatch_global_ue_indices,
        ),
    ) in enumerate(
        microbatch_specs
    ):
        with perf_region(
            "radio.channel_generation",
            device=context.config.device,
        ):
            h_freq = (
                _generate_microbatch_h_freq(
                    context=context,

                    tti_index=tti_index,

                    real_cell_index=(
                        real_cell_index
                    ),

                    microbatch_index=(
                        microbatch_index
                    ),

                    microbatch_global_ue_indices=(
                        microbatch_global_ue_indices
                    ),
                )
            )

        num_microbatch_ues = int(
            h_freq.shape[1]
        )

        if (
            num_microbatch_ues
            != int(
                microbatch_global_ue_indices
                .numel()
            )
        ):
            raise RuntimeError(
                "UE microbatch channel and global "
                "UE mapping have different UE "
                "dimensions."
            )

        if int(
            h_freq.shape[3]
        ) != context.num_cells:
            raise RuntimeError(
                "UE microbatch lost one or more "
                "BS links."
            )

        # ======================================================
        # FIXED ASSOCIATION
        #
        # Association was already determined globally.
        # We are NOT reassociating per microbatch.
        # ======================================================

        local_serving_bs = (
            torch.full(
                (
                    1,
                    num_microbatch_ues,
                ),

                fill_value=(
                    real_cell_index
                ),

                dtype=torch.long,

                device=h_freq.device,
            )
        )

        # ======================================================
        # RI / CSI
        # ======================================================

        # rank_data = (
        #     compute_ideal_svd_rank_diagnostic(
        #         h_freq=h_freq,

        #         serving_bs=(
        #             local_serving_bs
        #         ),

        #         num_rbgs=(
        #             context
        #             .config
        #             .num_rbgs
        #         ),

        #         tx_power_per_subcarrier_w=(
        #             context
        #             .tx_power_per_subcarrier_w
        #         ),

        #         noise_power_per_subcarrier_w=(
        #             context
        #             .noise_power_per_subcarrier_w
        #         ),
        #     )
        # )


        with perf_region(
            "radio.rank_diagnostic",
            device=context.config.device,
        ):
            rank_data = (
                compute_ideal_svd_rank_diagnostic(
                    h_freq=h_freq,
                    serving_bs=(
                        local_serving_bs
                    ),
                    num_rbgs=(
                        context
                        .config
                        .num_rbgs
                    ),
                    tx_power_per_subcarrier_w=(
                        context
                        .tx_power_per_subcarrier_w
                    ),
                    noise_power_per_subcarrier_w=(
                        context
                        .noise_power_per_subcarrier_w
                    ),
                )
            )

        # serving_mimo = (
        #     extract_serving_mimo_rbg_channel(
        #         h_freq=h_freq,

        #         serving_bs=(
        #             local_serving_bs
        #         ),

        #         num_rbgs=(
        #             context
        #             .config
        #             .num_rbgs
        #         ),
        #     )
        # )

        # csi_data = compute_ideal_svd_csi(
        #     h_serving_rbg=(
        #         rank_data
        #         .h_serving_rbg
        #     ),

        #     rank1_rbg_score=(
        #         rank_data
        #         .rank1_rbg_spectral_efficiency
        #     ),

        #     rank2_rbg_score=(
        #         rank_data
        #         .rank2_rbg_spectral_efficiency
        #     ),
        # )

        with perf_region(
            "radio.csi",
            device=context.config.device,
        ):
            csi_data = (
                compute_ideal_svd_csi(
                    h_serving_rbg=(
                        rank_data
                        .h_serving_rbg
                    ),
                    rank1_rbg_score=(
                        rank_data
                        .rank1_rbg_spectral_efficiency
                    ),
                    rank2_rbg_score=(
                        rank_data
                        .rank2_rbg_spectral_efficiency
                    ),
                )
            )

        # ======================================================
        # SU REPORTS NEEDED FOR PF-TDS + 1LDS STATE
        # ======================================================
        with perf_region(
            "radio.su_reports",
            device=context.config.device,
        ):
            su_report = (
                build_single_user_phy_reports(
                    h_freq=h_freq,

                    serving_bs=(
                        local_serving_bs
                    ),

                    recommended_rank=(
                        csi_data
                        .recommended_rank
                    ),

                    precoder_directions=(
                        csi_data
                        .precoder_directions
                    ),

                    num_rbgs=(
                        context.config.num_rbgs
                    ),

                    subcarriers_per_rbg=(
                        context
                        .config
                        .subcarriers_per_rb
                    ),

                    tx_power_per_subcarrier_w=(
                        context
                        .tx_power_per_subcarrier_w
                    ),

                    noise_power_per_subcarrier_w=(
                        context
                        .noise_power_per_subcarrier_w
                    ),

                    link_adaptation_config=(
                        context
                        .link_adaptation_config
                    ),

                    rate_config=(
                        context.rate_config
                    ),
                    precomputed_serving_channel=(
                        rank_data
                        .h_serving_rbg
                    ),

                    precomputed_inter_cell_covariance=(
                        rank_data
                        .inter_cell_covariance
                    ),
                )
            )

        local_rbg_rate_bps = (
            su_report
            .rate
            .target_compliant_rate_bps[
                0
            ]
        )

        td_instantaneous_rate_bps = (
            local_rbg_rate_bps.sum(
                dim=-1
            )
        )

        local_valid_mask = (
            torch.ones(
                (
                    num_microbatch_ues,
                ),

                dtype=torch.bool,

                device=h_freq.device,
            )
        )
        with perf_region(
            "radio.cqi",
            device=context.config.device,
        ):
            cqi = build_cqi_surrogate(
                mcs_index=(
                    su_report
                    .link_adaptation
                    .mcs_index[
                        0
                    ]
                ),

                meets_bler_target=(
                    su_report
                    .link_adaptation
                    .meets_bler_target[
                        0
                    ]
                ),

                candidate_valid_mask=(
                    local_valid_mask
                ),

                config=CQISurrogateConfig(),
            )

        # ======================================================
        # RETAIN ONLY WHAT SURVIVES THIS MICROBATCH
        # ======================================================

        built_microbatches.append(
            _CellRadioMicrobatch(
                serving_start_index=(
                    serving_start_index
                ),

                serving_stop_index=(
                    serving_stop_index
                ),

                #
                # Persistent/global identity.
                #
                global_ue_indices=(
                    microbatch_global_ue_indices
                ),

                #
                # Lightweight scheduler observation.
                #
                td_instantaneous_rate_bps=(
                    td_instantaneous_rate_bps
                ),

                rank=(
                    csi_data
                    .recommended_rank[
                        0
                    ]
                ),

                wideband_cqi=(
                    cqi
                    .wideband_cqi
                    .to(
                        dtype=torch.float32
                    )
                ),

                subband_cqi=(
                    cqi
                    .subband_cqi
                    .to(
                        dtype=torch.float32
                    )
                ),

                precoder_directions=(
                    csi_data
                    .precoder_directions[
                        0
                    ]
                ),

                #
                # Candidate PHY may need these after
                # PF-TDS.
                #
                h_freq=h_freq,

                recommended_rank=(
                    csi_data
                    .recommended_rank
                ),

                rx_combiners=(
                    csi_data
                    .rx_combiners
                ),

                csi_subcarrier_index=(
                    csi_data
                    .csi_subcarrier_index
                ),
            )
        )

        # ======================================================
        # CRITICAL TEMPORARY LIFETIME BOUNDARY
        #
        # The dataclass above now owns the few tensors
        # that must survive.
        #
        # Everything else from this Sionna invocation
        # should become reclaimable before the next
        # microbatch starts.
        #
        # Deliberately NO torch.cuda.empty_cache().
        # PyTorch should reuse released allocator blocks.
        # ======================================================

        del cqi

        del su_report


        del rank_data

        del csi_data

        del local_rbg_rate_bps

        del td_instantaneous_rate_bps

        del local_valid_mask

        del local_serving_bs

        #
        # channel and microbatch_topology now live
        # inside _generate_microbatch_h_freq().
        #
        # They leave scope automatically when that
        # helper returns, so there is nothing to
        # delete here.
        #
        del h_freq

    microbatches = tuple(
        built_microbatches
    )

    # ==========================================================
    # RECOMBINE LIGHTWEIGHT REPORTS
    #
    # Example:
    #
    #     [UE 0, UE 1]
    #     [UE 2, UE 3]
    #     ...
    #
    #         ->
    #
    #     one complete [serving_ue, ...] observation
    #
    # PF-TDS happens AFTER this.
    # ==========================================================
    with perf_region(
        "radio.combine_microbatches",
        device=context.config.device,
    ):
        observation = (
            _combine_cell_radio_microbatches(
                cell_global_ue_indices=(
                    cell_global_ue_indices
                ),

                microbatches=(
                    microbatches
                ),
            )
        )

    # ==========================================================
    # LAZY CANDIDATE PHY COMPACTION
    #
    # Until PF-TDS:
    #     H remains in small microbatch tensors.
    #
    # After PF-TDS:
    #     gather <=10 candidate rows
    #     ->
    #     release every microbatch H.
    # ==========================================================

    phy_holder: dict[
        str,
        object,
    ] = {
        "microbatches": (
            microbatches
        ),

        "materialized": None,
    }

    def physical_inputs_builder(
        prepared: PreparedOneLDSCellTTI,
    ) -> PPOPhysicalScoreInputs:
        existing = (
            phy_holder[
                "materialized"
            ]
        )

        if existing is not None:
            if not isinstance(
                existing,
                PPOPhysicalScoreInputs,
            ):
                raise RuntimeError(
                    "Invalid cached candidate PHY "
                    "object."
                )

            if not torch.equal(
                existing
                .candidate_global_ue_indices,

                prepared
                .candidate_global_ue_indices,
            ):
                raise RuntimeError(
                    "Candidate PHY was already "
                    "materialized for a different "
                    "PF-TDS candidate ordering."
                )

            return existing

        retained_microbatches = (
            phy_holder[
                "microbatches"
            ]
        )

        if not isinstance(
            retained_microbatches,
            tuple,
        ):
            raise RuntimeError(
                "UE microbatch PHY was released "
                "before candidate materialization."
            )

        compact_inputs = (
            _compact_candidate_phy_from_microbatches(
                prepared=prepared,

                cell_global_ue_indices=(
                    cell_global_ue_indices
                ),

                microbatches=(
                    retained_microbatches
                ),

                real_cell_index=(
                    real_cell_index
                ),

                context=context,
            )
        )

        phy_holder[
            "materialized"
        ] = compact_inputs

        #
        # CRITICAL:
        #
        # candidate H/rank/RX rows now live inside
        # compact_inputs.
        #
        # Noncandidate H for the complete serving
        # population is now dead.
        #
        phy_holder[
            "microbatches"
        ] = None

        return compact_inputs

    return PPOTrainingTTIInputs(
        observation=observation,

        physical_inputs_builder=(
            physical_inputs_builder
        ),

        packet_arrivals=None,
    )


class CellChunkedSionnaPPOInputProvider:
    """
    Lazily generate one serving-cell Sionna MIMO
    chunk at a time.

    Important centralized-training sequence:

        input_provider(TTI, cell 0)
            ->
        generate cell-0 serving-UE PHY
            ->
        PF-TDS preparation
            ->
        compact to <=10 candidate UEs

        input_provider(TTI, cell 1)
            ->
        generate cell-1 serving-UE PHY
            ->
        ...

    Therefore we never intentionally generate every
    full serving-cell MIMO tensor before PF-TDS
    compaction.

    OPEN-REPRODUCTION ENGINEERING:
        memory lifecycle only.
        Scheduling semantics are unchanged.
    """

    def __init__(
        self,
        *,
        context: ChunkedSionnaPPOContext,

        global_ue_indices_provider: (
            Callable[
                [
                    int,
                    int,
                ],
                torch.Tensor,
            ]
            | None
        ) = None,
    ) -> None:
        self.context = context

        self.global_ue_indices_provider = (
            global_ue_indices_provider
        )

        #
        # Production dynamic membership must use
        # identity-stable channel RNG.
        #
        # Minimal fake contexts in unit tests may not
        # expose config, so enforce this only when a
        # real config object is available.
        #
        if (
            global_ue_indices_provider
            is not None
        ):
            config = getattr(
                context,
                "config",
                None,
            )

            if (
                config is not None
                and not bool(
                    getattr(
                        config,
                        "identity_stable_ue_channel_rng",
                        False,
                    )
                )
            ):
                raise ValueError(
                    "Dynamic Sionna membership "
                    "requires identity-stable "
                    "UE channel RNG."
                )

        self._current_tti_index: (
            int | None
        ) = None

        self._current_inputs_by_stream: dict[
            int,
            PPOTrainingTTIInputs,
        ] = {}

        self._num_tti_builds = 0

        self._num_stream_builds = 0


    @property
    def num_tti_builds(
        self,
    ) -> int:
        return (
            self._num_tti_builds
        )


    @property
    def num_stream_builds(
        self,
    ) -> int:
        return (
            self._num_stream_builds
        )

    @property
    def num_channel_generations(
        self,
    ) -> int:
        return (
            self.context
            .channel_runtime
            .num_generations
        )


    @property
    def num_channel_topology_resets(
        self,
    ) -> int:
        return (
            self.context
            .channel_runtime
            .num_topology_resets
        )


    @property
    def current_tti_index(
        self,
    ) -> int | None:
        return (
            self._current_tti_index
        )


    @property
    def current_inputs(
        self,
    ) -> tuple[
        PPOTrainingTTIInputs,
        ...,
    ]:
        return tuple(
            self
            ._current_inputs_by_stream[
                stream_index
            ]

            for stream_index
            in sorted(
                self
                ._current_inputs_by_stream
            )
        )


    @property
    def current_observations(
        self,
    ) -> tuple[
        OneLDSCellTTIObservation,
        ...,
    ]:
        if len(
            self
            ._current_inputs_by_stream
        ) != self.context.num_streams:
            raise RuntimeError(
                "Not every PPO stream has been "
                "materialized for the current TTI."
            )

        return tuple(
            self
            ._current_inputs_by_stream[
                stream_index
            ]
            .observation

            for stream_index
            in range(
                self.context.num_streams
            )
        )


    def prepare_tti(
        self,
        tti_index: int,
    ) -> None:
        """
        Enter a new TTI WITHOUT generating cell PHY.

        Actual radio generation is lazy and occurs
        only when __call__ requests a particular
        stream.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if (
            self._current_tti_index
            is not None
            and tti_index
            < self._current_tti_index
        ):
            raise ValueError(
                "Chunked Sionna provider cannot "
                "move backward in TTI time."
            )

        if (
            tti_index
            == self._current_tti_index
        ):
            return

        #
        # Drop all provider-owned references to the
        # previous TTI.
        #
        self._current_inputs_by_stream = {}

        self._current_tti_index = (
            tti_index
        )

        self._num_tti_builds += 1


    def __call__(
        self,
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:
        self.prepare_tti(
            tti_index
        )

        if not (
            0
            <= stream_index
            < self.context.num_streams
        ):
            raise ValueError(
                "stream_index outside configured "
                "PPO streams."
            )

        existing = (
            self
            ._current_inputs_by_stream
            .get(
                stream_index
            )
        )

        if existing is not None:
            return existing

        # inputs = (
        #     _build_cell_training_inputs(
        #         context=self.context,

        #         tti_index=tti_index,

        #         stream_index=(
        #             stream_index
        #         ),
        #     )
        # )

        #
        # Production ChunkedSionnaPPOContext has
        # context.config.device.
        #
        # Some pure-logic unit tests intentionally use
        # a minimal fake context containing only
        # num_streams. Profiling must not impose extra
        # runtime requirements on that interface.
        #
        profile_config = getattr(
            self.context,
            "config",
            None,
        )

        profile_device = getattr(
            profile_config,
            "device",
            None,
        )

        with perf_region(
            "radio.cell_input_total",
            device=profile_device,
        ):

            if (
                self.global_ue_indices_provider
                is None
            ):

                #
                # Preserve the original static path.
                #
                inputs = (
                    _build_cell_training_inputs(
                        context=self.context,
                        tti_index=tti_index,
                        stream_index=stream_index,
                    )
                )

            else:

                dynamic_global_ue_indices = (
                    self
                    .global_ue_indices_provider(
                        tti_index,
                        stream_index,
                    )
                )

                inputs = (
                    _build_cell_training_inputs(
                        context=self.context,
                        tti_index=tti_index,
                        stream_index=stream_index,

                        cell_global_ue_indices_override=(
                            dynamic_global_ue_indices
                        ),
                    )
                )

        self._current_inputs_by_stream[
            stream_index
        ] = inputs

        self._num_stream_builds += 1

        return inputs


def build_chunked_state_managers(
    *,
    context: ChunkedSionnaPPOContext,
    initial_average_throughput_bps: float,
    throughput_forgetting_factor: float,
    num_candidates: int = 10,
) -> tuple[
    OneLDSCellTTIStateManager,
    ...,
]:
    """
    Build persistent PF-history state directly from
    the fixed serving-UE identities.

    No Sionna MIMO channel needs to be generated.
    """

    if (
        initial_average_throughput_bps
        < 0.0
    ):
        raise ValueError(
            "initial_average_throughput_bps "
            "cannot be negative."
        )

    managers = []

    for global_ue_indices in (
        context
        .global_ue_indices_by_stream
    ):
        valid_mask = torch.ones(
            global_ue_indices.shape,
            dtype=torch.bool,
            device=(
                global_ue_indices.device
            ),
        )

        initial_history = torch.full(
            valid_mask.shape,

            fill_value=(
                initial_average_throughput_bps
            ),

            dtype=torch.float32,

            device=(
                valid_mask.device
            ),
        )

        managers.append(
            OneLDSCellTTIStateManager(
                initial_average_throughput_bps=(
                    initial_history
                ),

                serving_global_ue_indices=(
                    global_ue_indices
                ),

                serving_ue_valid_mask=(
                    valid_mask
                ),

                tds_config=(
                    PFTimeDomainConfig(
                        num_candidates=(
                            num_candidates
                        )
                    )
                ),

                throughput_forgetting_factor=(
                    throughput_forgetting_factor
                ),
            )
        )

    return tuple(
        managers
    )

