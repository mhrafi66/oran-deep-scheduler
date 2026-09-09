from __future__ import annotations

from dataclasses import dataclass

import torch

from oran_scheduler.phy.csi import (
    compute_ideal_svd_csi,
    extract_serving_mimo_rbg_channel,
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
    generate_frequency_channel,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
    OneLDSCellTTIStateManager,
    PreparedOneLDSCellTTI,
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

    MEMORY ENGINEERING:
        paper MIMO is generated one serving-cell
        UE population at a time.

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

    topology_seed: int = 42

    association_channel_seed: int = 1000

    mimo_channel_seed: int = 2000

    device: str = "cuda:0"


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

        scenario="uma",

        isd_m=200.0,

        bs_height_m=25.0,

        ut_height_m=1.5,

        ut_speed_kmh=3.0,

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


def _build_cell_training_inputs(
    *,
    context: ChunkedSionnaPPOContext,
    tti_index: int,
    stream_index: int,
) -> PPOTrainingTTIInputs:
    """
    Build one PPO stream's real Sionna inputs.

    Critically:

        local UE count
            ~20-ish

        BS count
            always 21

    Therefore h_freq remains physically multicell:

        [1,
         local UE,
         4 Rx,
         21 BS,
         192 Tx,
         1 symbol,
         216 SC]

    but no complete 420-UE paper MIMO tensor exists.
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

    cell_global_ue_indices = (
        context
        .global_ue_indices_by_stream[
            stream_index
        ]
    )

    # ==========================================================
    # SUBSET FULL 420-UE TOPOLOGY
    # ==========================================================

    local_topology = subset_topology_ues(
        topology=context.topology,

        global_ue_indices=(
            cell_global_ue_indices
        ),
    )

    # ==========================================================
    # PAPER MIMO FOR ONLY THIS CELL'S SERVING UEs
    #
    # IMPORTANT:
    # ALL 21 BSs remain present.
    # ==========================================================

    channel_seed = (
        context
        .config
        .mimo_channel_seed

        + (
            tti_index
            * context.num_cells
        )

        + real_cell_index
    )

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,

        subcarrier_spacing_hz=30.0e3,

        num_rbs=(
            context.config.num_rbs
        ),

        subcarriers_per_rb=(
            context
            .config
            .subcarriers_per_rb
        ),

        num_ofdm_symbols=1,

        antenna_mode="paper",

        direction="downlink",

        o2i_model="low",

        enable_pathloss=True,

        enable_shadow_fading=True,

        precision="single",

        device=(
            context.config.device
        ),

        seed=channel_seed,
    )

    channel = generate_frequency_channel(
        topology=local_topology,

        topology_config=(
            context.topology_config
        ),

        channel_config=(
            channel_config
        ),
    )

    #
    # Retain ONLY H from ChannelData.
    #
    # Do not intentionally keep the Sionna channel
    # model object alive after this function.
    #
    h_freq = channel.h_freq

    num_local_ues = int(
        h_freq.shape[1]
    )

    if num_local_ues != int(
        cell_global_ue_indices.numel()
    ):
        raise RuntimeError(
            "Local MIMO channel and cell UE mapping "
            "have different UE dimensions."
        )

    if int(
        h_freq.shape[3]
    ) != context.num_cells:
        raise RuntimeError(
            "Cell chunk lost one or more BS links."
        )

    # ==========================================================
    # FIXED ASSOCIATION FOR THESE LOCAL UEs
    # ==========================================================

    local_serving_bs = torch.full(
        (
            1,
            num_local_ues,
        ),
        fill_value=(
            real_cell_index
        ),
        dtype=torch.long,
        device=h_freq.device,
    )

    # ==========================================================
    # LOCAL RI / CSI
    # ==========================================================

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

    serving_mimo = (
        extract_serving_mimo_rbg_channel(
            h_freq=h_freq,

            serving_bs=(
                local_serving_bs
            ),

            num_rbgs=(
                context
                .config
                .num_rbgs
            ),
        )
    )

    csi_data = compute_ideal_svd_csi(
        h_serving_rbg=(
            serving_mimo
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

    # ==========================================================
    # SU-MIMO REPORTS FOR PF TDS + CQI STATE
    # ==========================================================

    su_report = build_single_user_phy_reports(
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

    local_valid_mask = torch.ones(
        (
            num_local_ues,
        ),
        dtype=torch.bool,
        device=h_freq.device,
    )

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

    observation = (
        OneLDSCellTTIObservation(
            #
            # GLOBAL / persistent simulator identities.
            #
            serving_global_ue_indices=(
                cell_global_ue_indices
            ),

            serving_ue_valid_mask=(
                local_valid_mask
            ),

            td_instantaneous_rate_bps=(
                td_instantaneous_rate_bps
            ),

            rank=(
                csi_data
                .recommended_rank[
                    0
                ]
            ),

            #
            # Replaced by TrafficBufferManager before
            # PF TDS.
            #
            dl_buffer=torch.zeros(
                (
                    num_local_ues,
                ),
                dtype=torch.float32,
                device=h_freq.device,
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
        )
    )

    # ==========================================================
    # CANDIDATE-SPECIFIC PHYSICAL BUILDER
    # ==========================================================


    # ==========================================================
    # TEMPORARY FULL-CELL PHY HOLDER
    #
    # The full serving-cell MIMO tensors are needed only
    # until PF-TDS has selected its <=10 candidates.
    #
    # physical_inputs_builder() will:
    #
    #     full serving-cell PHY
    #             ↓
    #     PF-selected candidate rows
    #             ↓
    #     compact candidate PHY
    #
    # and then drop the references to the full-cell tensors.
    #
    # This is an OPEN-REPRODUCTION ENGINEERING optimization.
    # Scheduler/PHY mathematics are unchanged.
    # ==========================================================

    csi_subcarrier_index = (
        csi_data.csi_subcarrier_index
    )

    full_phy_holder = {
        "h_freq": h_freq,

        "recommended_rank": (
            csi_data.recommended_rank
        ),

        "rx_combiners": (
            csi_data.rx_combiners
        ),

        #
        # If the builder is called a second time for
        # the exact same prepared candidate set, reuse
        # the already compacted object.
        #
        "materialized": None,
    }

    def physical_inputs_builder(
        prepared: PreparedOneLDSCellTTI,
    ) -> PPOPhysicalScoreInputs:
        """
        Materialize only the PF-TDS candidate PHY.

        Before this call:

            h_freq UE dimension
                =
            every UE associated with this cell

        After this call:

            physical_inputs.h_freq UE dimension
                =
            num_candidates

        Normally:

            <= 10

        Persistent/global UE identities remain unchanged.
        """

        existing = (
            full_phy_holder[
                "materialized"
            ]
        )

        if existing is not None:
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


        # ------------------------------------------------------
        # Global candidate identity
        #       ->
        # row inside this serving-cell PHY tensor.
        # ------------------------------------------------------

        local_candidate_indices = (
            _map_candidate_global_to_local_phy(
                candidate_global_ue_indices=(
                    prepared
                    .candidate_global_ue_indices
                ),

                candidate_valid_mask=(
                    prepared
                    .candidate_valid_mask
                ),

                cell_global_ue_indices=(
                    cell_global_ue_indices
                ),
            )
        )

        candidate_valid_mask = (
            prepared
            .candidate_valid_mask
        )

        num_candidates = int(
            candidate_valid_mask.shape[0]
        )


        # ------------------------------------------------------
        # Padding may contain -1.
        #
        # Replace padded positions temporarily with zero so
        # index_select is legal. Those rows are zeroed below.
        # ------------------------------------------------------

        safe_local_indices = torch.where(
            candidate_valid_mask,

            local_candidate_indices,

            torch.zeros_like(
                local_candidate_indices
            ),
        )


        full_h_freq = (
            full_phy_holder[
                "h_freq"
            ]
        )

        full_recommended_rank = (
            full_phy_holder[
                "recommended_rank"
            ]
        )

        full_rx_combiners = (
            full_phy_holder[
                "rx_combiners"
            ]
        )

        if not isinstance(
            full_h_freq,
            torch.Tensor,
        ):
            raise RuntimeError(
                "Full-cell h_freq was released "
                "before candidate materialization."
            )

        if not isinstance(
            full_recommended_rank,
            torch.Tensor,
        ):
            raise RuntimeError(
                "Full-cell rank tensor was released "
                "before candidate materialization."
            )

        if not isinstance(
            full_rx_combiners,
            torch.Tensor,
        ):
            raise RuntimeError(
                "Full-cell RX combiners were released "
                "before candidate materialization."
            )


        # ------------------------------------------------------
        # Compact H:
        #
        #     [1, serving UE, RX, BS, TX, sym, SC]
        #
        # becomes:
        #
        #     [1, candidate, RX, BS, TX, sym, SC]
        #
        # ALL 21 BS links are retained.
        # ------------------------------------------------------

        candidate_h_freq = (
            full_h_freq
            .index_select(
                1,
                safe_local_indices,
            )
        )

        h_valid_mask = (
            candidate_valid_mask[
                None,
                :,
                None,
                None,
                None,
                None,
                None,
            ]
        )

        candidate_h_freq = torch.where(
            h_valid_mask,
            candidate_h_freq,
            torch.zeros_like(
                candidate_h_freq
            ),
        )


        # ------------------------------------------------------
        # Rank tensor:
        #
        #     [batch, local UE]
        #         ->
        #     [batch, candidate]
        # ------------------------------------------------------

        candidate_recommended_rank = (
            full_recommended_rank
            .index_select(
                1,
                safe_local_indices,
            )
        )

        candidate_recommended_rank = (
            torch.where(
                candidate_valid_mask[
                    None,
                    :,
                ],
                candidate_recommended_rank,
                torch.zeros_like(
                    candidate_recommended_rank
                ),
            )
        )


        # ------------------------------------------------------
        # RX combiners:
        #
        #     [batch, local UE, RBG, mode, RX]
        #         ->
        #     [batch, candidate, RBG, mode, RX]
        # ------------------------------------------------------

        candidate_rx_combiners = (
            full_rx_combiners
            .index_select(
                1,
                safe_local_indices,
            )
        )

        rx_valid_mask = (
            candidate_valid_mask[
                None,
                :,
                None,
                None,
                None,
            ]
        )

        candidate_rx_combiners = (
            torch.where(
                rx_valid_mask,
                candidate_rx_combiners,
                torch.zeros_like(
                    candidate_rx_combiners
                ),
            )
        )


        # ------------------------------------------------------
        # The new compact tensor has candidate positions
        #
        #     0, 1, ..., K-1
        #
        # as its physical UE namespace.
        #
        # Padding remains -1.
        # ------------------------------------------------------

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


        compact_inputs = (
            PPOPhysicalScoreInputs(
                #
                # Persistent scheduler identity.
                #
                candidate_global_ue_indices=(
                    prepared
                    .candidate_global_ue_indices
                ),

                #
                # Compact candidate-only PHY.
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
        )


        # ------------------------------------------------------
        # CRITICAL MEMORY RELEASE.
        #
        # The compact_inputs object now owns everything
        # required by PPO / PF expert / RZF.
        #
        # The full serving-cell H is no longer needed.
        # ------------------------------------------------------

        full_phy_holder[
            "materialized"
        ] = compact_inputs

        full_phy_holder[
            "h_freq"
        ] = None

        full_phy_holder[
            "recommended_rank"
        ] = None

        full_phy_holder[
            "rx_combiners"
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
    ) -> None:
        self.context = context

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

        inputs = (
            _build_cell_training_inputs(
                context=self.context,

                tti_index=tti_index,

                stream_index=(
                    stream_index
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

