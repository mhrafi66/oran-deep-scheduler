# Bell Labs Deep Scheduler — Open Simulator Architecture

## 1. Purpose

This document defines the simulator architecture for the open reproduction of:

**From Simulation to Practice: Generalizable Deep Reinforcement Learning for Cellular Schedulers**
Petteri Kela, Bryan Liu, and Alvaro Valcarce
arXiv:2411.08529v3

The original work uses a proprietary enterprise-grade, 3GPP-compliant 5G NR system-level simulator.

Our objective is therefore not a bit-identical recreation of Nokia's internal simulator.

Instead, the objective is to:

1. reproduce every publicly specified system parameter;
2. use established open wireless simulation components whenever possible;
3. implement the paper-specific scheduling algorithms ourselves;
4. explicitly document simulator substitutions;
5. validate each wireless layer before introducing reinforcement learning;
6. reproduce the published trends and performance as closely as possible;
7. use the validated reproduction for controlled stress testing.

---

# 2. Core Technology Decision

The primary implementation stack will be:

* **Python 3.11**
* **PyTorch**
* **Sionna PHY**
* **Sionna SYS**
* custom scheduler code
* custom traffic/buffer model where necessary
* custom RL implementation in PyTorch

We will not use Stable-Baselines3 for the final paper reproduction.

The earlier Stable-Baselines3 PPO implementation remains a toy proof-of-concept only.

---

# 3. Architectural Principle

The simulator and scheduler must remain separate modules.

Conceptually:

```text
              WIRELESS ENVIRONMENT
                     |
                     v
           channel / CSI / buffers
                     |
                     v
             TIME DOMAIN SCHEDULER
                     |
                     v
             10 candidate UEs
                     |
                     v
             SCHEDULING ALGORITHM
              /               \
       heuristic             deep RL
                     |
                     v
          RBG + MU-MIMO allocation
                     |
                     v
               RZF / PHY
                     |
                     v
              SINR / MCS / BLER
                     |
                     v
            realized throughput
                     |
                     v
            next TTI / new state
```

The RL algorithm must not contain wireless-system logic that properly belongs to the environment.

Likewise, the wireless simulator must not depend on one particular learning algorithm.

---

# 4. Major Component Ownership

Every component is classified as one of:

* `SIONNA`
* `CUSTOM`
* `SIONNA + CUSTOM`
* `REPRODUCTION ASSUMPTION`

---

## 4.1 Multi-Cell Topology

### Paper

Evaluation configuration:

* 21 cells
* 200 m inter-site distance
* 25 m gNB height
* 210 UEs
* uniformly distributed UEs
* 1.5 m UE height
* 3 km/h UE speed

Training:

* 420 UEs

### Implementation

**Owner: SIONNA**

Use:

```python
sionna.sys.gen_hexgrid_topology
```

with a one-ring topology.

Sionna's one-ring topology produces seven three-sector sites:

```text
7 sites × 3 sectors = 21 sector cells
```

Therefore the mapping is natural:

```text
paper 21 cells
        ↕
Sionna 21 sectors
```

Evaluation:

```text
10 UEs / sector × 21 sectors = 210 UEs
```

Training:

```text
20 UEs / sector × 21 sectors = 420 UEs
```

Explicit parameters:

```text
num_rings = 1
isd = 200 m
bs_height = 25 m
min_ut_height = 1.5 m
max_ut_height = 1.5 m
min_ut_velocity = 3 km/h
max_ut_velocity = 3 km/h
```

Convert velocity:

```text
3 km/h = 0.833333... m/s
```

### Validation

The first simulator test must assert:

```text
num_cells == 21
num_ues == 210
BS height == 25 m
UE height == 1.5 m
```

and visualize the topology.

---

# 5. Propagation / Channel Model

## Paper

Deployment:

```text
3GPP Dense Urban Macro
```

Carrier:

```text
4 GHz
```

## Open-Reproduction Implementation

**Owner: SIONNA + REPRODUCTION ASSUMPTION**

Primary Sionna model:

```python
sionna.phy.channel.tr38901.UMa
```

with:

```text
carrier_frequency = 4e9 Hz
```

The paper references a Dense Urban Macro evaluation configuration, whereas Sionna exposes TR 38.901 UMa rather than a simulator configuration explicitly named "Dense Urban Macro."

We therefore do not claim:

```text
Sionna UMa == proprietary Nokia DUMa simulator
```

Instead, we document:

```text
Open reproduction:
TR 38.901 UMa
+
paper-specified geometry and parameters
```

This is a reproduction assumption and must remain visible in all comparison discussions.

---

# 6. Pathloss, Shadowing, and Fast Fading

## Implementation

**Owner: SIONNA**

Use the TR 38.901 channel model with:

* pathloss enabled;
* shadow fading enabled;
* time-varying channel coefficients.

The simulator must expose actual channel-dependent quantities to the scheduler.

The reproduction must never use the old toy model:

```text
base_rate × Uniform(0.7, 1.3)
```

---

# 7. Mobility

## Paper

```text
UE speed = 3 km/h
```

## Implementation

**Owner: SIONNA**

Initial UE velocity is generated through the topology utility.

UE positions evolve between TTIs according to velocity.

Nominal reproduction:

```text
3 km/h = 0.833333 m/s
```

Later stress tests will vary this value.

---

# 8. gNB Antenna Array

## Paper

```text
12 × 8 antenna panel
2 polarizations
```

## Proposed Implementation

**Owner: SIONNA + VALIDATION**

Use a Sionna `PanelArray` corresponding to:

```text
12 rows
8 columns
dual polarization
38.901 antenna pattern
4 GHz
```

Candidate configuration:

```python
PanelArray(
    num_rows_per_panel=12,
    num_cols_per_panel=8,
    polarization="dual",
    polarization_type="VH",
    antenna_pattern="38.901",
    carrier_frequency=4e9,
)
```

Before accepting this mapping, explicitly verify:

* Sionna antenna-port count;
* polarization interpretation;
* whether the paper's `12 × 8 × 2` representation corresponds exactly to Sionna's dual-polarized panel convention.

Do not silently assume the two representations are identical.

---

# 9. UE Antenna

## Paper

```text
2 dual-polarized Rx antennas
```

Maximum reported UE rank:

```text
2
```

## Implementation

**Owner: SIONNA + REPRODUCTION ASSUMPTION**

This wording requires careful mapping to Sionna's `PanelArray`.

The exact number of physical elements versus polarization ports must be verified before fixing the final configuration.

Initial implementation should isolate the UE-array configuration so that it can be changed without modifying scheduler code.

---

# 10. Downlink Transmission

## Paper

Scheduling direction:

```text
downlink
```

gNB transmit power:

```text
44 dBm
```

## Implementation

**Owner: SIONNA**

Use:

```text
direction = downlink
bs_max_power_dbm = 44
```

Power allocation must ultimately respect the selected RBG/MU-MIMO allocation.

---

# 11. OFDM / NR Resource Configuration

## Paper Evaluation

```text
Bandwidth = 100 MHz
Subcarrier spacing = 30 kHz
273 RBs
18 RBGs
```

## Paper Training

```text
Bandwidth = 6.6 MHz
18 RBs
18 RBGs
```

## Implementation

**Owner: SIONNA + CUSTOM**

Sionna `ResourceGrid` provides the OFDM frequency-time representation.

However, the scheduler should operate on:

```text
RBGs
```

rather than directly on individual subcarriers.

Therefore introduce an explicit module:

```text
RBGMapper
```

whose responsibility is:

```text
subcarriers / RBs
        ↓
paper-compatible RBGs
```

The important scheduler-facing dimension remains:

```text
N_RBG = 18
```

for both training and evaluation.

This is crucial because the paper deliberately trains and evaluates with the same number of scheduler outputs despite a major bandwidth shift.

---

# 12. Frequency Selectivity

## Requirement

The paper's scheduler uses:

```text
sub-band CQI per RBG
```

and performs frequency-selective scheduling.

## Sionna Limitation

The Sionna system-level example estimates one effective achievable rate and broadcasts it across subcarriers.

That simplification cannot be used in this project.

## Implementation

**Owner: CUSTOM + SIONNA PHY**

We must obtain frequency-dependent channel/SINR information.

The scheduler-facing representation must contain:

```text
rate_or_CQI[cell, RBG, UE]
```

or equivalently:

```text
subband_CQI[cell, RBG, UE]
```

Candidate shape for the evaluation configuration:

```text
[21, 18, 10]
```

after TDS has selected ten candidate UEs per cell.

Before TDS, the UE dimension corresponds to all active UEs in the cell.

---

# 13. Time-Domain Scheduler

## Paper

```text
TD scheduler = Proportional Fair
maximum candidate UEs = 10
```

## Implementation

**Owner: CUSTOM**

Do not directly use `PFSchedulerSUMIMO` as the final scheduler.

Implement:

```text
PFTimeDomainScheduler
```

whose job is only:

```text
all active UEs
      ↓
PF priority
      ↓
top 10 candidates
```

The FDS/SDS scheduler then operates only over these ten candidates.

The built-in Sionna PF scheduler may be used as:

* a sanity reference;
* a unit-test oracle for simple SU-MIMO cases;
* a way to validate our discounted PF calculations.

---

# 14. Past Average Throughput

## Paper

For UE `u`:

```text
R_u(t)
=
(1 - epsilon) * p_u(t)
+
epsilon * R_u(t-1)
```

## Implementation

**Owner: CUSTOM**

Create explicit per-UE state:

```text
past_average_throughput
```

The paper does not publicly specify the numerical forgetting factor sufficiently for exact replication.

Therefore:

```text
epsilon
```

must be a configuration parameter.

It must never be buried as a hard-coded number inside the scheduler.

---

# 15. Frequency-Domain Scheduling

## Paper

The scheduler performs non-contiguous downlink resource allocation with RBG granularity.

## Implementation

**Owner: CUSTOM**

Create an explicit FDS interface.

Input:

```text
candidate UE set
per-RBG CQI/rate
UE history
buffers
rank
```

Output:

```text
initial RBG allocations
```

The FDS implementation must remain independent of RL.

---

# 16. Spatial-Domain Scheduling

## Paper

MU-MIMO user pairing is performed after / together with frequency allocation.

Evaluation:

```text
up to 8 co-scheduled UEs per RBG
```

Training:

```text
up to 4 co-scheduled UEs per RBG
```

## Implementation

**Owner: CUSTOM**

Create:

```text
SpatialScheduler
```

with implementations:

```text
BaselineSDS
PFGreedySDS
DeepSchedulerSDS
```

All three must use the same wireless evaluation functions.

This prevents differences in PHY implementation from contaminating scheduler comparisons.

---

# 17. RZF Precoding

## Paper

```text
MU-MIMO with regularized zero forcing
```

## Implementation

**Owner: SIONNA PHY + CUSTOM WRAPPER**

Sionna PHY provides RZF-related precoding functionality.

Create a thin project wrapper:

```text
RZFEngine
```

whose responsibility is:

```text
selected UE set
channel matrices
UE ranks
        ↓
RZF precoder
        ↓
post-precoding channel
        ↓
per-user SINR
```

The scheduler must not directly invoke low-level Sionna functions everywhere.

All RZF calculations go through this wrapper so that the implementation can be changed or validated centrally.

---

# 18. Precoder Cross-Correlation

## Paper State Feature

For each:

```text
RBG m
candidate UE u
```

the state includes the maximum cross-correlation between the candidate's precoder and those of already scheduled users.

## Implementation

**Owner: CUSTOM using SIONNA-derived precoders**

Create:

```text
PrecoderCorrelationCalculator
```

Inputs:

```text
candidate precoder P_m,u
scheduled-user precoders P_m,c
```

Outputs:

```text
max_correlation[m, u]
mean_correlation[m, u]
```

The maximum value is required by 1LDS.

The mean value is additionally required by the 2LDS feature set.

---

# 19. UE Rank / RI

## Paper

```text
maximum UE rank = 2
CSI includes RI
```

## Implementation

**Owner: CUSTOM + SIONNA PHY**

The simulator must expose:

```text
RI ∈ {1, 2}
```

to the scheduler.

Because exact Nokia RI feedback behavior is not publicly specified, the first implementation should use a clearly documented RI estimation policy.

RI logic must be isolated behind:

```text
RankIndicatorProvider
```

so that later:

* oracle RI;
* delayed RI;
* noisy RI;
* standard-inspired RI

can be tested independently.

---

# 20. CQI

## Paper State

* wideband CQI;
* sub-band CQI for every RBG.

## Implementation

**Owner: CUSTOM + SIONNA PHY**

Create:

```text
CQIProvider
```

Inputs:

```text
channel / SINR measurements
```

Outputs:

```text
wideband_CQI[UE]
subband_CQI[RBG, UE]
```

CQI generation must be separated from the scheduler.

This separation is important for later stress testing:

```text
perfect CQI
delayed CQI
quantized CQI
noisy CQI
missing CQI
```

---

# 21. PMI

## Paper

CSI includes:

```text
PMI
```

and precoder information is fundamental to the cross-correlation state.

## Implementation

**Owner: CUSTOM + SIONNA PHY**

Create:

```text
PMIProvider
```

The initial implementation may use channel-derived precoder information if exact standardized feedback behavior cannot be reproduced immediately.

Any such substitution must be explicitly labeled as a reproduction assumption.

Later PMI error/delay becomes a stress-test dimension.

---

# 22. Receiver Processing

## Paper

```text
UE receiver = MRC
```

## Implementation

**Owner: SIONNA PHY / CUSTOM VALIDATION**

The final reproduction must match an MRC-equivalent receiver.

Do not silently substitute the LMMSE receiver used in Sionna's example.

Before implementation, identify the exact Sionna primitive or construct the required MRC SINR computation explicitly.

Until then:

```text
receiver mapping = unresolved
```

This is a required reproduction-fidelity checkpoint.

---

# 23. SINR

## Implementation

**Owner: SIONNA PHY**

After:

```text
allocation
+
RZF
+
inter-cell interference
+
receiver processing
```

compute a per-user/per-resource post-equalization SINR.

The scheduler's instantaneous rate estimates and final BLER evaluation must be based on physically consistent SINR quantities.

---

# 24. Link Adaptation

## Paper

```text
QPSK through 256-QAM
10% BLER target
```

## Implementation

**Owner: SIONNA SYS**

Use Sionna link adaptation / PHY abstraction where compatible.

Nominal target:

```text
BLER = 0.10
```

Sionna's link-adaptation framework can align with TS 38.214 mappings when its default compatible transport-block and SINR functions are used.

Exact MCS-table choice must be explicitly recorded in configuration.

---

# 25. Physical-Layer Abstraction

## Implementation

**Owner: SIONNA SYS**

Use post-equalization SINR as input to Sionna's PHY abstraction.

Conceptually:

```text
per-resource SINR
      ↓
effective SINR
      ↓
BLER
      ↓
decoded / failed transport data
```

This gives the system simulator a substantially more realistic throughput model than Shannon-rate-only scheduling.

---

# 26. Traffic and Buffers

## Paper Evaluation

### Full Buffer

```text
100% FB
```

### FTP Model 3

```text
packet size = 0.5 MB
arrival rate = 20 packets/s
```

## Paper Training

```text
50% FB
50% FTP3
```

FTP3:

```text
packet size = 1.5 kB
arrival rate = 500 packets/s
```

## Implementation

**Owner: CUSTOM**

Create:

```text
TrafficGenerator
BufferManager
```

Traffic state must be independent of the scheduler.

Per UE maintain at least:

```text
buffer_bytes
packet arrivals
served bytes
```

The scheduler receives:

```text
normalized downlink buffer status
```

but cannot directly modify traffic-generation logic.

---

# 27. Realized Throughput

## Implementation

**Owner: CUSTOM + SIONNA PHY abstraction**

Delivered throughput is determined only after:

```text
allocation
MCS selection
PHY abstraction
BLER / decoding result
```

The realized throughput feeds:

* PF history;
* geometric mean throughput;
* PPO reward;
* evaluation metrics;
* FTP user-perceived throughput.

---

# 28. Simulation Time Step

The environment advances in:

```text
TTIs / slots
```

A single simulator step represents the scheduler decision cycle.

Conceptual sequence:

```text
START TTI t

1. Update traffic buffers
2. Obtain CSI
3. Update PF state from previous throughput
4. Run PF TDS
5. Build candidate state
6. Run FDS / SDS
7. Build RZF precoders
8. Select MCS
9. Compute PHY outcome
10. Calculate throughput
11. Record metrics
12. Move UEs
13. Advance channel
14. Enter TTI t+1
```

RL reward timing must follow the paper, not Gymnasium convenience.

---

# 29. Paper State Builder

Create:

```text
PaperStateBuilder
```

It is responsible only for transforming simulator state into the normalized input expected by the deep scheduler.

## 1LDS

Per candidate UE:

```text
past average throughput
rank
allocated RBG count
buffer status
wideband CQI
18 sub-band CQIs
18 maximum precoder correlations
```

With:

```text
10 candidates
18 RBGs
```

state dimension:

```text
10 × (5 + 2×18)
= 410
```

## 2LDS

State dimension:

```text
10 × 8 + 1
= 81
```

The state builder must be unit-tested independently of PPO/SACD.

---

# 30. Action Masking

Create:

```text
ActionMaskBuilder
```

For every RBG/layer, actions are:

```text
10 candidate UEs
+
1 no-allocation action
```

Total:

```text
11 actions
```

Mask at least:

* padded candidate entries;
* UEs already scheduled on the same RBG;
* infeasible allocations.

For 1LDS:

```text
mask shape = [18, 11]
```

---

# 31. Baseline Scheduler Order

Before any RL code is trusted, implement:

```text
1. PF TDS
2. SU-MIMO FDS
3. RZF evaluation
4. Baseline SDS
5. PF Greedy SDS
```

Each stage receives unit/integration tests.

Only after these produce plausible system behavior do we start PPO.

---

# 32. Deep Scheduler Interface

All learning algorithms must implement a common interface conceptually equivalent to:

```python
action = scheduler.select_action(
    state=state,
    action_mask=mask,
)
```

The simulator should not care whether `scheduler` is:

```text
BaselineSDS
PFGreedySDS
PPO1LDS
DSACD1LDS
SACD2LDS
```

This enables fair comparisons.

---

# 33. Training Versus Evaluation Configuration

Use separate configuration files.

Proposed:

```text
configs/
    training.yaml
    evaluation_full_buffer.yaml
    evaluation_ftp3.yaml
```

Never scatter train/evaluation differences throughout Python source.

Training:

```text
420 UEs
6.6 MHz
18 RBs
18 RBGs
4 MU-MIMO UE layers
50% FB + 50% FTP3
```

Evaluation:

```text
210 UEs
100 MHz
273 RBs
18 RBGs
8 MU-MIMO UE layers
100% FB
or
100% FTP3
```

---

# 34. Randomness and Reproducibility

At experiment startup set and record:

```text
Python seed
NumPy seed
PyTorch seed
Sionna seed
```

Every experiment result must store its configuration and seed.

The ten evaluation drops should therefore be reproducible.

---

# 35. GPU / CPU Strategy

The simulator should support:

```text
cpu
cuda:0
```

through configuration.

Do not hard-code the compute device.

Development/sanity tests:

```text
CPU whenever practical
```

Large channel simulation and training:

```text
CHPC GPU job
```

GPU allocation should only be requested when a real GPU workload is ready to run.

---

# 36. Expected Repository Modules

Target project structure:

```text
oran-deep-scheduler/
│
├── README.md
├── .gitignore
│
├── configs/
│   ├── training.yaml
│   ├── evaluation_full_buffer.yaml
│   └── evaluation_ftp3.yaml
│
├── docs/
│   ├── reproduction_spec.md
│   └── simulator_architecture.md
│
├── src/
│   └── oran_scheduler/
│       │
│       ├── simulator/
│       │   ├── topology.py
│       │   ├── channel.py
│       │   ├── resource_grid.py
│       │   ├── csi.py
│       │   ├── traffic.py
│       │   ├── buffers.py
│       │   ├── link_adaptation.py
│       │   └── system_simulator.py
│       │
│       ├── phy/
│       │   ├── rzf.py
│       │   ├── correlation.py
│       │   └── sinr.py
│       │
│       ├── schedulers/
│       │   ├── pf_tds.py
│       │   ├── baseline_sds.py
│       │   ├── pf_greedy_sds.py
│       │   └── interfaces.py
│       │
│       ├── state/
│       │   ├── builder.py
│       │   ├── normalization.py
│       │   └── action_mask.py
│       │
│       ├── rl/
│       │   ├── ppo/
│       │   ├── sacd/
│       │   └── dsacd/
│       │
│       ├── metrics/
│       │   ├── throughput.py
│       │   ├── fairness.py
│       │   └── latency.py
│       │
│       └── utils/
│
├── tests/
│   ├── simulator/
│   ├── schedulers/
│   ├── state/
│   └── rl/
│
├── experiments/
│
└── scripts/
    └── slurm/
```

This structure may evolve, but responsibilities should remain separated.

---

# 37. First Executable Milestones

## Milestone 1 — Topology

Produce:

```text
21 sector cells
210 UEs
200 m ISD
25 m BS height
1.5 m UE height
3 km/h UE velocity
```

Validate tensor dimensions and produce a topology plot.

No scheduler.

No PPO.

---

## Milestone 2 — Channel

Generate:

```text
4 GHz UMa channel matrices
```

and validate:

* pathloss;
* channel tensor dimensions;
* inter-cell links;
* finite values.

---

## Milestone 3 — Frequency-Selective Link State

Produce per:

```text
cell × UE × RBG
```

channel/SINR/rate information.

This is the first point at which the Bell Labs scheduler can receive meaningful frequency-selective input.

---

## Milestone 4 — PF TDS

Select:

```text
10 candidate UEs / cell
```

using PF.

---

## Milestone 5 — SU-MIMO Scheduling

Validate frequency-domain scheduling before spatial multiplexing.

---

## Milestone 6 — RZF + MU-MIMO

Allow multiple users on each RBG and evaluate the resulting SINR/throughput.

---

## Milestone 7 — Baseline SDS

Reproduce the paper's iterative baseline.

---

## Milestone 8 — PF Greedy SDS

Reproduce the more computationally expensive reference scheduler.

---

## Milestone 9 — Deep State Interface

Validate:

```text
1LDS input = 410
1LDS action logits = 198

2LDS input = 81
2LDS action logits = 11
```

---

## Milestone 10 — PPO

Only now implement paper PPO.

---

# 38. Reproduction-Fidelity Register

The following items must stay explicitly unresolved until validated:

| Component                                   | Current status      |
| ------------------------------------------- | ------------------- |
| Nokia proprietary system simulator          | unavailable         |
| DUMa-to-Sionna-UMa mapping                  | assumption          |
| exact UE antenna interpretation             | unresolved          |
| exact RI generation                         | unresolved          |
| exact PMI feedback generation               | unresolved          |
| exact CQI generation                        | unresolved          |
| exact MRC mapping in Sionna                 | unresolved          |
| exact RZF regularization parameter          | unresolved          |
| PF forgetting factor                        | unspecified         |
| some normalization constants                | unspecified         |
| exact traffic timing details                | partially specified |
| precise interference/product implementation | simulator-dependent |

These are not bugs in the reproduction.

They are differences between:

```text
publicly documented methodology
```

and:

```text
proprietary implementation details
```

Every final experiment must clearly separate them.

---

# 39. Stress-Testing Architecture Requirement

Although stress testing comes after reproduction, simulator components must be designed so that the following can later be changed independently:

```text
UE speed
number of UEs
bandwidth
traffic load
traffic distribution
CSI delay
CQI error
PMI error
RI error
number of candidate UEs
MU-MIMO layers
channel scenario
interference
buffer load
observation corruption
inference delay
```

No such parameter should be buried inside an RL algorithm.

---

# 40. Immediate Next Step

The first code written after this architecture document will not implement reinforcement learning.

It will implement:

```text
21-cell topology sanity test
```

Success criteria:

```text
21 cells
210 UEs
correct paper geometry
correct heights
correct nominal speed
valid Sionna UMa topology
visual inspection
automated tensor-shape assertions
```

Only after this test passes will channel simulation be added.


