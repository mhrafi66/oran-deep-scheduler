from __future__ import annotations

from collections.abc import Callable
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

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)

from oran_scheduler.simulator.cell_association import (
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
from oran_scheduler.simulator.rbg import (
    RBGConfig,
    compute_rbg_channel_power,
)
from oran_scheduler.simulator.serving_layout import (
    ServingCellData,
    build_serving_cell_data,
    gather_serving_ue_rbg_values,
)
from oran_scheduler.simulator.topology import (
    TopologyConfig,
    TopologyData,
    generate_topology,
)
from oran_scheduler.state.cqi_features import (
    CQISurrogateConfig,
    build_cqi_surrogate,
)


@dataclass(frozen=True)
class FixedSionnaPPOSnapshotConfig:
    """
    Configuration for the first real-Sionna PPO
    integration snapshot.

    PAPER-SPECIFIED TRAINING FREQUENCY SHAPE:
        18 RB
        18 RBG
        12 subcarriers / RB

    PAPER-SPECIFIED TOPOLOGY SHAPE:
        21 cells

    TEMPORARY SCAFFOLDING:
        num_ut_per_sector defaults to 1 rather than
        the paper's 20 training UEs/sector.

        This keeps the 12x8 dual-polarized paper-array
        MIMO channel memory-safe while validating the
        complete Sionna -> PPO training path.

    TEMPORARY SCAFFOLDING:
        Only num_training_cells cells are exposed to
        the centralized PPO smoke run.

    The complete 21-cell/420-UE training experiment
    comes only after this integration is validated.
    """

    num_training_cells: int = 2

    num_ut_per_sector: int = 1

    num_rbs: int = 18

    num_rbgs: int = 18

    subcarriers_per_rb: int = 12

    topology_seed: int = 42

    channel_seed: int = 42

    device: str = "cuda:0"


    def __post_init__(
        self,
    ) -> None:
        if self.num_training_cells <= 0:
            raise ValueError(
                "num_training_cells must be positive."
            )

        if self.num_training_cells > 21:
            raise ValueError(
                "The current one-ring topology has "
                "only 21 cells."
            )

        if self.num_ut_per_sector <= 0:
            raise ValueError(
                "num_ut_per_sector must be positive."
            )

        if self.num_rbs <= 0:
            raise ValueError(
                "num_rbs must be positive."
            )

        if self.num_rbgs <= 0:
            raise ValueError(
                "num_rbgs must be positive."
            )

        if self.subcarriers_per_rb <= 0:
            raise ValueError(
                "subcarriers_per_rb must be positive."
            )

        #
        # Current reduced-bandwidth training mapping:
        #
        #     18 RB
        #       ->
        #     18 RBG
        #
        # therefore:
        #
        #     1 RB / RBG
        #
        if self.num_rbs != self.num_rbgs:
            raise ValueError(
                "The current training snapshot requires "
                "one RB per RBG, so num_rbs must equal "
                "num_rbgs."
            )

@dataclass(frozen=True)
class FixedSionnaPPOSnapshot:
    """
    One real Sionna radio snapshot prepared for the
    existing centralized PPO training runner.

    selected_cell_indices:
        Actual Sionna/gNB indices corresponding to
        PPO stream 0, stream 1, ...

    observations:
        One scheduler-facing serving-UE observation
        per selected PPO stream.

    h_freq:
        Complete real Sionna multi-cell MIMO channel.

    recommended_rank / rx_combiners:
        Global-UE CSI quantities required by the
        existing common MU-MIMO evaluator.

    IMPORTANT:
        The radio snapshot itself is currently fixed
        across TTIs.

        Traffic queues, PF history, PPO allocations,
        rewards, and learning are NOT fixed.
    """

    config: FixedSionnaPPOSnapshotConfig

    #
    # Stable radio/environment identity.
    #
    # These are retained so later TTIs can regenerate
    # the channel while preserving UE identities and
    # the originally established serving-cell layout.
    #
    topology_config: TopologyConfig

    topology: TopologyData

    frozen_serving_bs: torch.Tensor

    #
    # Exact channel RNG seed used for THIS snapshot.
    #
    channel_seed_used: int

    selected_cell_indices: tuple[
        int,
        ...
    ]

    selected_num_ues_per_cell: tuple[
        int,
        ...
    ]

    all_num_ues_per_cell: torch.Tensor

    observations: tuple[
        OneLDSCellTTIObservation,
        ...
    ]

    h_freq: torch.Tensor

    recommended_rank: torch.Tensor

    rx_combiners: torch.Tensor

    csi_subcarrier_index: int

    subcarriers_per_rbg: int

    tx_power_per_subcarrier_w: torch.Tensor

    noise_power_per_subcarrier_w: torch.Tensor

    link_adaptation_config: LinkAdaptationConfig

    rate_config: RateConfig


    @property
    def num_training_cells(
        self,
    ) -> int:
        return len(
            self.selected_cell_indices
        )


    def training_inputs(
        self,
        *,
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:
        """
        Convert one selected real Sionna cell into
        the exact interface expected by the existing
        multi-cell PPO training runner.

        TEMPORARY SCAFFOLDING:
            tti_index does not regenerate the radio
            snapshot yet.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if not (
            0
            <= stream_index
            < self.num_training_cells
        ):
            raise ValueError(
                "stream_index is outside the "
                "selected training-cell range."
            )

        serving_cell_index = (
            self.selected_cell_indices[
                stream_index
            ]
        )

        observation = (
            self.observations[
                stream_index
            ]
        )

        def physical_inputs_builder(
            prepared: PreparedOneLDSCellTTI,
        ) -> PPOPhysicalScoreInputs:
            """
            PF-TDS determines the candidate ordering
            first.

            Only then can we construct the
            candidate-specific real PHY input.
            """

            return PPOPhysicalScoreInputs(
                candidate_global_ue_indices=(
                    prepared
                    .candidate_global_ue_indices
                ),
                h_freq=self.h_freq,
                serving_cell_index=(
                    serving_cell_index
                ),
                recommended_rank=(
                    self.recommended_rank
                ),
                rx_combiners=(
                    self.rx_combiners
                ),
                csi_subcarrier_index=(
                    self.csi_subcarrier_index
                ),
                subcarriers_per_rbg=(
                    self.subcarriers_per_rbg
                ),
                tx_power_per_subcarrier_w=(
                    self.tx_power_per_subcarrier_w
                ),
                noise_power_per_subcarrier_w=(
                    self.noise_power_per_subcarrier_w
                ),
                link_adaptation_config=(
                    self.link_adaptation_config
                ),
                rate_config=(
                    self.rate_config
                ),
                batch_index=0,
            )

        return PPOTrainingTTIInputs(
            observation=observation,
            physical_inputs_builder=(
                physical_inputs_builder
            ),
            packet_arrivals=None,
        )


def _gather_global_ue_values_to_serving(
    *,
    global_ue_values: torch.Tensor,
    serving_data: ServingCellData,
) -> torch.Tensor:
    """
    Gather arbitrary global-UE tensors into the
    padded serving-cell layout.

    Input:

        global_ue_values:
            [batch, global_UE, ...]

    Serving mapping:

        serving_data.global_ue_indices:
            [batch, cell, serving_UE]

    Output:

        [batch, cell, serving_UE, ...]

    Examples:

        rank:
            [1, U]
                ->
            [1, C, S]

        precoder directions:
            [1, U, RBG, mode, TX]
                ->
            [1, C, S, RBG, mode, TX]

    Invalid padded serving slots are zeroed.
    """

    if global_ue_values.ndim < 2:
        raise ValueError(
            "global_ue_values must have shape "
            "[batch, global_UE, ...]."
        )

    if (
        serving_data
        .global_ue_indices
        .ndim
        != 3
    ):
        raise ValueError(
            "serving_data.global_ue_indices must "
            "have shape [batch, cell, serving_UE]."
        )

    if (
        global_ue_values.shape[0]
        != serving_data.global_ue_indices.shape[0]
    ):
        raise ValueError(
            "Batch dimensions do not match."
        )

    if (
        global_ue_values.device
        != serving_data.global_ue_indices.device
    ):
        raise ValueError(
            "Global values and serving layout must "
            "be on the same device."
        )

    if (
        serving_data.valid_ue_mask.device
        != global_ue_values.device
    ):
        raise ValueError(
            "Serving mask is on the wrong device."
        )

    batch_size = (
        global_ue_values.shape[0]
    )

    num_global_ues = (
        global_ue_values.shape[1]
    )

    (
        _,
        num_cells,
        num_serving_ues,
    ) = (
        serving_data
        .global_ue_indices
        .shape
    )

    feature_shape = tuple(
        global_ue_values.shape[2:]
    )

    valid_global_indices = (
        serving_data
        .global_ue_indices[
            serving_data.valid_ue_mask
        ]
    )

    if valid_global_indices.numel() > 0:
        if torch.any(
            valid_global_indices < 0
        ):
            raise ValueError(
                "A valid serving slot contains a "
                "negative global UE index."
            )

        if torch.any(
            valid_global_indices
            >= num_global_ues
        ):
            raise ValueError(
                "Serving layout references a UE "
                "outside global_ue_values."
            )

    #
    # Padding stores -1.
    #
    # torch.gather cannot consume -1 here, so use
    # zero as a temporary safe index and blank that
    # result afterward.
    #
    safe_global_indices = torch.where(
        serving_data.valid_ue_mask,
        serving_data.global_ue_indices,
        torch.zeros_like(
            serving_data.global_ue_indices
        ),
    )

    expanded_global_values = (
        global_ue_values
        .unsqueeze(1)
        .expand(
            batch_size,
            num_cells,
            num_global_ues,
            *feature_shape,
        )
    )

    gather_indices = (
        safe_global_indices
    )

    for _ in feature_shape:
        gather_indices = (
            gather_indices.unsqueeze(-1)
        )

    gather_indices = gather_indices.expand(
        batch_size,
        num_cells,
        num_serving_ues,
        *feature_shape,
    )

    serving_values = torch.gather(
        expanded_global_values,
        dim=2,
        index=gather_indices,
    )

    serving_mask = (
        serving_data.valid_ue_mask
    )

    for _ in feature_shape:
        serving_mask = (
            serving_mask.unsqueeze(-1)
        )

    return torch.where(
        serving_mask,
        serving_values,
        torch.zeros_like(
            serving_values
        ),
    )



def build_fixed_sionna_ppo_snapshot(
    *,
    config: (
        FixedSionnaPPOSnapshotConfig
        | None
    ) = None,
) -> FixedSionnaPPOSnapshot:
    """
    Construct one real paper-array Sionna snapshot
    and expose selected cells through the existing
    PPO-training interface.

    Pipeline:

        21-cell Sionna UMa topology
            ->
        paper-array MIMO channel
            ->
        cell association
            ->
        serving-cell padded layout
            ->
        ideal-SVD RI / direction surrogate
            ->
        SU-MIMO physical reports
            ->
        CQI surrogate
            ->
        OneLDSCellTTIObservation
            ->
        PPOPhysicalScoreInputs builder

    REPRODUCTION LABELS:

    PAPER-SPECIFIED:
        21-cell UMa geometry
        4 GHz
        30 kHz SCS
        18 RB / 18 RBG training bandwidth
        44 dBm
        12x8 dual-polarized gNB array
        max rank 2

    OPEN-REPRODUCTION:
        UMa O2I model
        ideal-SVD RI/PMI-direction surrogate
        CQI surrogate
        noise assumptions
        association implementation

    TEMPORARY SCAFFOLDING:
        reduced UE count
        selected subset of cells
        fixed radio realization across TTIs
    """

    if config is None:
        config = (
            FixedSionnaPPOSnapshotConfig()
        )

    device = config.device

    # ==========================================================
    # 1. REAL 21-CELL TOPOLOGY
    # ==========================================================

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,
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
        device=device,
    )

    topology = generate_topology(
        topology_config
    )

    # ==========================================================
    # 2. REAL PAPER-ARRAY MIMO CHANNEL
    # ==========================================================

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,
        subcarrier_spacing_hz=30.0e3,
        num_rbs=config.num_rbs,
        subcarriers_per_rb=(
            config.subcarriers_per_rb
        ),
        num_ofdm_symbols=1,
        antenna_mode="paper",
        direction="downlink",

        #
        # OPEN-REPRODUCTION.
        #
        o2i_model="low",

        enable_pathloss=True,
        enable_shadow_fading=True,
        precision="single",
        device=device,
        seed=config.channel_seed,
    )

    channel = generate_frequency_channel(
        topology=topology,
        topology_config=topology_config,
        channel_config=channel_config,
    )

    # ==========================================================
    # 3. CELL ASSOCIATION
    # ==========================================================

    association = build_cell_association(
        h_freq=channel.h_freq,
        precision=(
            channel_config.precision
        ),
    )

    # ==========================================================
    # 4. RBG + PADDED SERVING LAYOUT
    # ==========================================================

    rbg_config = RBGConfig(
        num_rbgs=config.num_rbgs,
        subcarriers_per_rb=(
            config.subcarriers_per_rb
        ),
        rbs_per_rbg=1,
    )

    rbg_data = compute_rbg_channel_power(
        channel=channel,
        config=rbg_config,
    )

    serving_data = build_serving_cell_data(
        rbg_data=rbg_data,
        association=association,
    )

    # ==========================================================
    # 5. TX POWER + THERMAL NOISE
    # ==========================================================

    sinr_config = SINRConfig(
        tx_power_dbm=44.0,
        subcarrier_spacing_hz=30.0e3,
        temperature_k=294.0,

        #
        # OPEN-REPRODUCTION:
        # paper does not publish exact receiver NF
        # used by the proprietary simulator.
        #
        receiver_noise_figure_db=None,

        precision="single",
        device=device,
    )

    tx_power_per_subcarrier_w = (
        compute_subcarrier_tx_power_w(
            config=sinr_config,
            num_subcarriers=(
                channel_config
                .num_subcarriers
            ),
        )
    )

    noise_power_per_subcarrier_w = (
        compute_noise_power_per_subcarrier_w(
            sinr_config
        )
    )

    # ==========================================================
    # 6. IDEAL-SVD RI / PMI-DIRECTION SURROGATE
    # ==========================================================

    rank_data = (
        compute_ideal_svd_rank_diagnostic(
            h_freq=channel.h_freq,
            serving_bs=(
                association.serving_bs
            ),
            num_rbgs=config.num_rbgs,
            tx_power_per_subcarrier_w=(
                tx_power_per_subcarrier_w
            ),
            noise_power_per_subcarrier_w=(
                noise_power_per_subcarrier_w
            ),
        )
    )

    serving_mimo = (
        extract_serving_mimo_rbg_channel(
            h_freq=channel.h_freq,
            serving_bs=(
                association.serving_bs
            ),
            num_rbgs=config.num_rbgs,
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
    # 7. SU-MIMO PHYSICAL REPORTS
    #
    # These feed PF-TDS instantaneous rate and CQI.
    # The final PPO allocation itself will later be
    # evaluated by the COMMON MU-MIMO RZF/MRC PHY.
    # ==========================================================

    link_adaptation_config = (
        LinkAdaptationConfig(
            num_rbgs=config.num_rbgs,
            subcarriers_per_rbg=(
                rbg_config
                .subcarriers_per_rbg
            ),
            precision="single",
            device=device,
        )
    )

    rate_config = RateConfig(
        subcarriers_per_rbg=(
            rbg_config
            .subcarriers_per_rbg
        ),
        device=device,
    )

    su_report = build_single_user_phy_reports(
        h_freq=channel.h_freq,
        serving_bs=(
            association.serving_bs
        ),
        recommended_rank=(
            csi_data.recommended_rank
        ),
        precoder_directions=(
            csi_data.precoder_directions
        ),
        num_rbgs=config.num_rbgs,
        subcarriers_per_rbg=(
            rbg_config
            .subcarriers_per_rbg
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

    global_rbg_rate_bps = (
        su_report
        .rate
        .target_compliant_rate_bps
    )

    global_mcs_index = (
        su_report
        .link_adaptation
        .mcs_index
    )

    global_meets_bler_target = (
        su_report
        .link_adaptation
        .meets_bler_target
    )

    # ==========================================================
    # 8. GLOBAL UE REPORTS -> SERVING-CELL LAYOUT
    # ==========================================================

    serving_rbg_rate_bps = (
        gather_serving_ue_rbg_values(
            global_ue_rbg_values=(
                global_rbg_rate_bps
            ),
            serving_data=serving_data,
        )
    )

    serving_mcs_index = (
        gather_serving_ue_rbg_values(
            global_ue_rbg_values=(
                global_mcs_index
            ),
            serving_data=serving_data,
        )
    )

    serving_meets_bler_target = (
        gather_serving_ue_rbg_values(
            global_ue_rbg_values=(
                global_meets_bler_target
            ),
            serving_data=serving_data,
        )
    )

    serving_rank = (
        _gather_global_ue_values_to_serving(
            global_ue_values=(
                csi_data.recommended_rank
            ),
            serving_data=serving_data,
        )
    )

    serving_precoder_directions = (
        _gather_global_ue_values_to_serving(
            global_ue_values=(
                csi_data
                .precoder_directions
            ),
            serving_data=serving_data,
        )
    )

    # ==========================================================
    # 9. CQI SURROGATE IN SERVING LAYOUT
    # ==========================================================

    cqi = build_cqi_surrogate(
        mcs_index=serving_mcs_index,
        meets_bler_target=(
            serving_meets_bler_target
        ),
        candidate_valid_mask=(
            serving_data.valid_ue_mask
        ),
        config=CQISurrogateConfig(),
    )

    # ==========================================================
    # 10. SELECT CELLS FOR FIRST INTEGRATION RUN
    #
    # TEMPORARY SCAFFOLDING:
    #
    # With only 21 total UEs, association may leave
    # several cells empty.
    #
    # For the smoke test, take the most populated
    # non-empty cells so both PPO streams exercise
    # actual scheduling.
    #
    # This selection MUST NOT become the final paper
    # training/evaluation cell-selection rule.
    # ==========================================================

    ue_counts = (
        serving_data
        .num_ues_per_cell[
            0,
            :,
        ]
    )

    descending_cells = torch.argsort(
        ue_counts,
        descending=True,
    )

    selected_cells: list[int] = []

    for cell_tensor in descending_cells:
        cell_index = int(
            cell_tensor.item()
        )

        if int(
            ue_counts[
                cell_index
            ].item()
        ) <= 0:
            continue

        selected_cells.append(
            cell_index
        )

        if (
            len(selected_cells)
            == config.num_training_cells
        ):
            break

    if (
        len(selected_cells)
        != config.num_training_cells
    ):
        raise RuntimeError(
            "Not enough non-empty cells exist for "
            "the requested real-Sionna PPO streams."
        )

    # ==========================================================
    # 11. BUILD SCHEDULER OBSERVATION FOR EACH STREAM
    # ==========================================================

    observations: list[
        OneLDSCellTTIObservation
    ] = []

    selected_num_ues: list[
        int
    ] = []

    for cell_index in selected_cells:
        global_ue_indices = (
            serving_data
            .global_ue_indices[
                0,
                cell_index,
                :,
            ]
        )

        valid_mask = (
            serving_data
            .valid_ue_mask[
                0,
                cell_index,
                :,
            ]
        )

        rbg_rate_bps = (
            serving_rbg_rate_bps[
                0,
                cell_index,
                :,
                :,
            ]
        )

        #
        # PF-TDS needs one instantaneous-rate number
        # per serving UE.
        #
        # Current physical-rate frontend convention:
        # sum the UE's target-compliant SU-MIMO rate
        # across all RBGs.
        #
        td_instantaneous_rate_bps = (
            rbg_rate_bps.sum(
                dim=-1
            )
        )

        rank = (
            serving_rank[
                0,
                cell_index,
                :,
            ]
        )

        wideband_cqi = (
            cqi
            .wideband_cqi[
                0,
                cell_index,
                :,
            ]
            .to(
                dtype=torch.float32
            )
        )

        subband_cqi = (
            cqi
            .subband_cqi[
                0,
                cell_index,
                :,
                :,
            ]
            .to(
                dtype=torch.float32
            )
        )

        precoder_directions = (
            serving_precoder_directions[
                0,
                cell_index,
                :,
                :,
                :,
                :,
            ]
        )

        #
        # Placeholder only.
        #
        # run_traffic_aware_ppo_cell_tti_step()
        # replaces this with the actual persistent
        # TrafficBufferManager queue state BEFORE
        # PF-TDS/state construction.
        #
        dl_buffer = torch.zeros_like(
            td_instantaneous_rate_bps,
            dtype=torch.float32,
        )

        observation = (
            OneLDSCellTTIObservation(
                serving_global_ue_indices=(
                    global_ue_indices
                ),
                serving_ue_valid_mask=(
                    valid_mask
                ),
                td_instantaneous_rate_bps=(
                    td_instantaneous_rate_bps
                ),
                rank=rank,
                dl_buffer=dl_buffer,
                wideband_cqi=(
                    wideband_cqi
                ),
                subband_cqi=(
                    subband_cqi
                ),
                precoder_directions=(
                    precoder_directions
                ),
            )
        )

        observations.append(
            observation
        )

        selected_num_ues.append(
            int(
                valid_mask
                .sum()
                .item()
            )
        )

    return FixedSionnaPPOSnapshot(
        config=config,
        topology_config=(
            topology_config
        ),

        topology=(
            topology
        ),

        frozen_serving_bs=(
            association
            .serving_bs
            .detach()
            .clone()
        ),

        channel_seed_used=(
            channel_config.seed
        ),
        selected_cell_indices=tuple(
            selected_cells
        ),
        selected_num_ues_per_cell=tuple(
            selected_num_ues
        ),
        all_num_ues_per_cell=(
            ue_counts
            .detach()
            .clone()
        ),
        observations=tuple(
            observations
        ),
        h_freq=channel.h_freq,
        recommended_rank=(
            csi_data.recommended_rank
        ),
        rx_combiners=(
            csi_data.rx_combiners
        ),
        csi_subcarrier_index=(
            csi_data.csi_subcarrier_index
        ),
        subcarriers_per_rbg=(
            rbg_config
            .subcarriers_per_rbg
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


def build_refreshed_sionna_ppo_snapshot(
    *,
    reference_snapshot: FixedSionnaPPOSnapshot,
    tti_index: int,
) -> FixedSionnaPPOSnapshot:
    """
    Generate a new Sionna radio realization for one
    later TTI while preserving the established UE
    identities and serving-cell association.

    What CHANGES each TTI:

        complex H
        RI surrogate
        CSI/PMI-direction surrogate
        SU-MIMO rate reports
        MCS/TBLER reports
        sub-band CQI
        wideband CQI

    What remains FIXED:

        BS positions
        UE positions
        UE global identities
        serving-BS association
        selected PPO stream/cell identities
        padded serving-UE layout

    IMPORTANT REPRODUCTION LABEL:

        TEMPORARY SCAFFOLDING

    This is an independently refreshed Sionna
    realization, NOT yet a physically time-correlated
    mobility trajectory.

    The next simulator milestone will handle actual
    temporal channel/mobility evolution.
    """

    if tti_index < 0:
        raise ValueError(
            "tti_index must be non-negative."
        )

    config = (
        reference_snapshot.config
    )

    #
    # Deterministic experiment convention:
    #
    #     TTI 0 -> base seed
    #     TTI 1 -> base + 1
    #     ...
    #
    channel_seed = (
        config.channel_seed
        + tti_index
    )

    # ==========================================================
    # 1. GENERATE A NEW CHANNEL OVER THE SAME TOPOLOGY
    # ==========================================================

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,
        subcarrier_spacing_hz=30.0e3,
        num_rbs=config.num_rbs,
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
        seed=channel_seed,
    )

    channel = generate_frequency_channel(
        topology=(
            reference_snapshot.topology
        ),
        topology_config=(
            reference_snapshot
            .topology_config
        ),
        channel_config=(
            channel_config
        ),
    )

    #
    # CRITICAL:
    #
    # Do NOT recompute cell association here.
    #
    # The initial association is deliberately frozen
    # for this temporal integration stage.
    #
    serving_bs = (
        reference_snapshot
        .frozen_serving_bs
    )

    # ==========================================================
    # 2. NEW RI + PRECODER-DIRECTION REPORTS
    # ==========================================================

    rank_data = (
        compute_ideal_svd_rank_diagnostic(
            h_freq=channel.h_freq,
            serving_bs=serving_bs,
            num_rbgs=config.num_rbgs,
            tx_power_per_subcarrier_w=(
                reference_snapshot
                .tx_power_per_subcarrier_w
            ),
            noise_power_per_subcarrier_w=(
                reference_snapshot
                .noise_power_per_subcarrier_w
            ),
        )
    )

    serving_mimo = (
        extract_serving_mimo_rbg_channel(
            h_freq=channel.h_freq,
            serving_bs=serving_bs,
            num_rbgs=config.num_rbgs,
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
    # 3. NEW SU-MIMO RADIO REPORTS
    #
    # These are scheduler-side reports used for:
    #
    #     PF-TDS instantaneous rate
    #     CQI features
    #
    # They are NOT the final MU-MIMO allocation
    # throughput.
    # ==========================================================

    su_report = build_single_user_phy_reports(
        h_freq=channel.h_freq,
        serving_bs=serving_bs,
        recommended_rank=(
            csi_data.recommended_rank
        ),
        precoder_directions=(
            csi_data.precoder_directions
        ),
        num_rbgs=config.num_rbgs,
        subcarriers_per_rbg=(
            reference_snapshot
            .subcarriers_per_rbg
        ),
        tx_power_per_subcarrier_w=(
            reference_snapshot
            .tx_power_per_subcarrier_w
        ),
        noise_power_per_subcarrier_w=(
            reference_snapshot
            .noise_power_per_subcarrier_w
        ),
        link_adaptation_config=(
            reference_snapshot
            .link_adaptation_config
        ),
        rate_config=(
            reference_snapshot
            .rate_config
        ),
    )

    global_rbg_rate_bps = (
        su_report
        .rate
        .target_compliant_rate_bps
    )

    global_mcs_index = (
        su_report
        .link_adaptation
        .mcs_index
    )

    global_meets_bler_target = (
        su_report
        .link_adaptation
        .meets_bler_target
    )

    # ==========================================================
    # 4. BUILD NEW GLOBAL CQI REPORTS
    # ==========================================================

    global_valid_ue_mask = torch.ones(
        global_mcs_index.shape[
            :-1
        ],
        dtype=torch.bool,
        device=(
            global_mcs_index.device
        ),
    )

    global_cqi = build_cqi_surrogate(
        mcs_index=(
            global_mcs_index
        ),
        meets_bler_target=(
            global_meets_bler_target
        ),
        candidate_valid_mask=(
            global_valid_ue_mask
        ),
        config=CQISurrogateConfig(),
    )

    # ==========================================================
    # 5. REBUILD ONLY THE ALREADY-SELECTED SERVING
    #    CELL OBSERVATIONS
    #
    # We use the REFERENCE global UE identity vector.
    #
    # We do not allow a new channel draw to reorder
    # the persistent serving layout.
    # ==========================================================

    observations: list[
        OneLDSCellTTIObservation
    ] = []

    for stream_index in range(
        reference_snapshot
        .num_training_cells
    ):
        reference_observation = (
            reference_snapshot
            .observations[
                stream_index
            ]
        )

        global_ue_indices = (
            reference_observation
            .serving_global_ue_indices
        )

        valid_mask = (
            reference_observation
            .serving_ue_valid_mask
        )

        #
        # Padding uses -1.
        #
        # Replace it temporarily with UE 0 so tensor
        # indexing remains legal. Padding is zeroed
        # immediately afterward.
        #
        safe_global_ue_indices = torch.where(
            valid_mask,
            global_ue_indices,
            torch.zeros_like(
                global_ue_indices
            ),
        )

        rbg_rate_bps = (
            global_rbg_rate_bps[
                0,
                safe_global_ue_indices,
                :,
            ]
        )

        rbg_rate_bps = torch.where(
            valid_mask.unsqueeze(
                -1
            ),
            rbg_rate_bps,
            torch.zeros_like(
                rbg_rate_bps
            ),
        )

        td_instantaneous_rate_bps = (
            rbg_rate_bps.sum(
                dim=-1
            )
        )

        rank = (
            csi_data
            .recommended_rank[
                0,
                safe_global_ue_indices,
            ]
        )

        rank = torch.where(
            valid_mask,
            rank,
            torch.zeros_like(
                rank
            ),
        )

        wideband_cqi = (
            global_cqi
            .wideband_cqi[
                0,
                safe_global_ue_indices,
            ]
            .to(
                dtype=torch.float32
            )
        )

        wideband_cqi = torch.where(
            valid_mask,
            wideband_cqi,
            torch.zeros_like(
                wideband_cqi
            ),
        )

        subband_cqi = (
            global_cqi
            .subband_cqi[
                0,
                safe_global_ue_indices,
                :,
            ]
            .to(
                dtype=torch.float32
            )
        )

        subband_cqi = torch.where(
            valid_mask.unsqueeze(
                -1
            ),
            subband_cqi,
            torch.zeros_like(
                subband_cqi
            ),
        )

        precoder_directions = (
            csi_data
            .precoder_directions[
                0,
                safe_global_ue_indices,
                :,
                :,
                :,
            ]
        )

        direction_mask = (
            valid_mask[
                :,
                None,
                None,
                None,
            ]
        )

        precoder_directions = torch.where(
            direction_mask,
            precoder_directions,
            torch.zeros_like(
                precoder_directions
            ),
        )

        #
        # This remains a placeholder because the
        # traffic-aware PPO step replaces it with the
        # persistent TrafficBufferManager state.
        #
        dl_buffer = torch.zeros_like(
            td_instantaneous_rate_bps,
            dtype=torch.float32,
        )

        observation = (
            OneLDSCellTTIObservation(
                serving_global_ue_indices=(
                    global_ue_indices
                    .detach()
                    .clone()
                ),
                serving_ue_valid_mask=(
                    valid_mask
                    .detach()
                    .clone()
                ),
                td_instantaneous_rate_bps=(
                    td_instantaneous_rate_bps
                ),
                rank=rank,
                dl_buffer=dl_buffer,
                wideband_cqi=(
                    wideband_cqi
                ),
                subband_cqi=(
                    subband_cqi
                ),
                precoder_directions=(
                    precoder_directions
                ),
            )
        )

        observations.append(
            observation
        )

    # ==========================================================
    # 6. NEW SNAPSHOT, SAME IDENTITIES
    # ==========================================================

    return FixedSionnaPPOSnapshot(
        config=config,

        topology_config=(
            reference_snapshot
            .topology_config
        ),

        topology=(
            reference_snapshot
            .topology
        ),

        frozen_serving_bs=(
            reference_snapshot
            .frozen_serving_bs
        ),

        channel_seed_used=(
            channel_seed
        ),

        selected_cell_indices=(
            reference_snapshot
            .selected_cell_indices
        ),

        selected_num_ues_per_cell=(
            reference_snapshot
            .selected_num_ues_per_cell
        ),

        all_num_ues_per_cell=(
            reference_snapshot
            .all_num_ues_per_cell
        ),

        observations=tuple(
            observations
        ),

        h_freq=channel.h_freq,

        recommended_rank=(
            csi_data.recommended_rank
        ),

        rx_combiners=(
            csi_data.rx_combiners
        ),

        csi_subcarrier_index=(
            csi_data.csi_subcarrier_index
        ),

        subcarriers_per_rbg=(
            reference_snapshot
            .subcarriers_per_rbg
        ),

        tx_power_per_subcarrier_w=(
            reference_snapshot
            .tx_power_per_subcarrier_w
        ),

        noise_power_per_subcarrier_w=(
            reference_snapshot
            .noise_power_per_subcarrier_w
        ),

        link_adaptation_config=(
            reference_snapshot
            .link_adaptation_config
        ),

        rate_config=(
            reference_snapshot
            .rate_config
        ),
    )


class IndependentSionnaPPOTTIInputProvider:
    """
    Provide one shared refreshed Sionna radio
    realization per TTI.

    Example:

        provider(0, cell 0)
            -> use initial snapshot

        provider(0, cell 1)
            -> SAME initial snapshot

        provider(1, cell 0)
            -> generate channel seed base+1

        provider(1, cell 1)
            -> reuse SAME TTI-1 snapshot

    Therefore all centralized PPO streams belonging
    to one TTI observe one coherent network channel
    realization.

    TEMPORARY SCAFFOLDING:

        Each TTI is an independently regenerated
        Sionna realization over fixed topology and
        frozen association.

        This is not yet Doppler/time-correlated
        mobility.
    """

    def __init__(
        self,
        *,
        initial_snapshot: (
            FixedSionnaPPOSnapshot
        ),
        initial_tti_index: int = 0,
    ) -> None:
        if initial_tti_index < 0:
            raise ValueError(
                "initial_tti_index must be "
                "non-negative."
            )

        self._current_snapshot = (
            initial_snapshot
        )

        self._current_tti_index = (
            initial_tti_index
        )

        self._num_channel_refreshes = 0


    @property
    def current_snapshot(
        self,
    ) -> FixedSionnaPPOSnapshot:
        return (
            self._current_snapshot
        )


    @property
    def current_tti_index(
        self,
    ) -> int:
        return (
            self._current_tti_index
        )


    @property
    def num_channel_refreshes(
        self,
    ) -> int:
        return (
            self._num_channel_refreshes
        )


    def _ensure_tti_snapshot(
        self,
        *,
        tti_index: int,
    ) -> None:
        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if (
            tti_index
            < self._current_tti_index
        ):
            raise ValueError(
                "Sionna PPO radio provider cannot "
                "move backward in TTI time."
            )

        if (
            tti_index
            == self._current_tti_index
        ):
            return

        self._current_snapshot = (
            build_refreshed_sionna_ppo_snapshot(
                reference_snapshot=(
                    self._current_snapshot
                ),
                tti_index=(
                    tti_index
                ),
            )
        )

        self._current_tti_index = (
            tti_index
        )

        self._num_channel_refreshes += 1


    def __call__(
        self,
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:
        self._ensure_tti_snapshot(
            tti_index=tti_index
        )

        return (
            self
            ._current_snapshot
            .training_inputs(
                tti_index=tti_index,
                stream_index=(
                    stream_index
                ),
            )
        )



def build_snapshot_state_managers(
    *,
    snapshot: FixedSionnaPPOSnapshot,
    initial_average_throughput_bps: float,
    throughput_forgetting_factor: float,
    num_candidates: int = 10,
) -> tuple[
    OneLDSCellTTIStateManager,
    ...,
]:
    """
    Create one persistent PF/state manager per
    selected real Sionna cell.

    PAPER-SPECIFIED:
        max PF-TDS candidates = 10.

    OPEN-REPRODUCTION:
        initial_average_throughput_bps
        throughput_forgetting_factor

    These values are therefore required explicitly
    rather than hidden inside this bridge.
    """

    if initial_average_throughput_bps <= 0.0:
        raise ValueError(
            "initial_average_throughput_bps must "
            "be positive."
        )

    if not (
        0.0
        <= throughput_forgetting_factor
        <= 1.0
    ):
        raise ValueError(
            "throughput_forgetting_factor must lie "
            "in [0, 1]."
        )

    if num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive."
        )

    managers: list[
        OneLDSCellTTIStateManager
    ] = []

    for observation in snapshot.observations:
        valid_mask = (
            observation
            .serving_ue_valid_mask
        )

        initial_history = torch.where(
            valid_mask,
            torch.full(
                valid_mask.shape,
                fill_value=(
                    initial_average_throughput_bps
                ),
                dtype=torch.float32,
                device=valid_mask.device,
            ),
            torch.zeros(
                valid_mask.shape,
                dtype=torch.float32,
                device=valid_mask.device,
            ),
        )

        manager = (
            OneLDSCellTTIStateManager(
                initial_average_throughput_bps=(
                    initial_history
                ),
                serving_global_ue_indices=(
                    observation
                    .serving_global_ue_indices
                ),
                serving_ue_valid_mask=(
                    valid_mask
                ),
                tds_config=(
                    PFTimeDomainConfig(
                        num_candidates=(
                            num_candidates
                        ),
                    )
                ),
                throughput_forgetting_factor=(
                    throughput_forgetting_factor
                ),
            )
        )

        managers.append(
            manager
        )

    return tuple(
        managers
    )


def build_snapshot_multicell_input_provider(
    *,
    snapshot: FixedSionnaPPOSnapshot,
) -> Callable[
    [
        int,
        int,
    ],
    PPOTrainingTTIInputs,
]:
    """
    Produce the exact two-argument provider expected
    by run_multicell_ppo_training():

        provider(
            tti_index,
            cell/stream_index,
        )
    """

    def input_provider(
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:
        return snapshot.training_inputs(
            tti_index=tti_index,
            stream_index=stream_index,
        )

    return input_provider

