# Bell Labs Deep Scheduler Reproduction Specification

## 1. Target Paper

This repository targets:

**From Simulation to Practice: Generalizable Deep Reinforcement Learning
for Cellular Schedulers**

Authors:
- Petteri Kela
- Bryan Liu
- Alvaro Valcarce

Target version:

- arXiv:2411.08529v3
- 9 October 2025

The original 2024 v1 paper is retained only for historical comparison.
Unless explicitly stated otherwise, implementation decisions should follow v3.

---

## 2. Reproduction Goal

The objective is to independently reproduce the deep scheduling framework
described in the paper using an open simulation and learning stack.

The original paper uses a proprietary enterprise-grade 3GPP-compliant
5G NR system-level simulator. Therefore, exact replication of the authors'
simulation engine is not possible from the public paper.

The goal of this project is instead to reproduce, as faithfully as possible:

1. Network configuration
2. Training and evaluation scenarios
3. Time-domain candidate selection
4. Frequency-domain scheduling
5. Spatial-domain MU-MIMO scheduling
6. Reference heuristic schedulers
7. 1LDS architecture
8. 2LDS architecture
9. PPO with expert guidance
10. SACD
11. DSACD
12. Training protocol
13. Evaluation metrics
14. Generalization experiments
15. Computational-complexity measurements

After reproduction, the implementation will be used for controlled
stress testing.

### PPO sample granularity: 84 vs. 1512 samples/TTI

The v3 paper contains an internal ambiguity in the definition of a
PPO training sample.

The 1LDS architecture performs one neural-network inference per
MU-MIMO user layer. The actor produces an action vector containing
one categorical action for every RBG:

    a = [a_1, ..., a_N_RBG]

and the PPO policy probability is defined as the product of the
individual RBG branch probabilities. Algorithm 1 likewise stores one
(state, action, reward, log-probability) item per user layer.

Therefore the primary reproduction treats one temporal PPO
transition as one complete joint all-RBG decision for one user layer.

For the training configuration:

    21 cells x 4 user layers = 84 joint PPO transitions / TTI.

Separately, the paper states that:

    21 cells x 4 user layers x 18 RBGs
        = 1512 samples / TTI.

We interpret this second number as the number of RBG-level scheduling
decision components contained within the 84 joint transitions:

    84 joint transitions x 18 RBG branches
        = 1512 RBG decision/sample equivalents.

This interpretation preserves the paper's reported sample-count
arithmetic (~450k RBG-level samples by TTI 400) without changing the
joint 1LDS PPO action and probability formulation.

This distinction is PAPER-AMBIGUOUS / OPEN-REPRODUCTION and should be
reported explicitly in experiments.

For PPO, M=128 is interpreted as the update threshold in joint
layer-level transitions. In synchronized multi-cell training, the
number of available transitions may cross M without equaling it
exactly. The primary reproduction therefore consumes the complete
current on-policy batch when the threshold is reached
(`use_all_when_reached`). This centralized batching rule is an
OPEN-REPRODUCTION choice.

---

## 3. Reproducibility Labels

Every implementation choice should be classified as one of:

### PAPER-SPECIFIED

The paper explicitly provides the value, equation, procedure, or architecture.

### PAPER-INFERRED

The paper strongly implies the implementation, but does not completely
specify it.

### UNSPECIFIED

The public paper does not provide enough information to uniquely reproduce
the implementation.

Unspecified parameters must not silently be invented. They should be
documented and investigated through sensitivity experiments where necessary.

---

## 4. Evaluation System Configuration

The following configuration is PAPER-SPECIFIED.

| Parameter | Paper value |
|---|---:|
| Deployment | 3GPP Dense Urban Macro |
| Number of cells | 21 |
| Inter-site distance | 200 m |
| gNB height | 25 m |
| gNB transmit power | 44 dBm |
| Number of UEs | 210 |
| UE placement | Uniformly distributed |
| Random drops | 10 |
| Bandwidth | 100 MHz |
| Number of RBs | 273 |
| Number of RBGs | 18 |
| Carrier frequency | 4 GHz |
| Subcarrier spacing | 30 kHz |
| MIMO scheme | MU-MIMO with RZF |
| Modulation | QPSK to 256-QAM |
| Link-adaptation BLER target | 10% |
| gNB antenna panel | 12 x 8, 2 polarizations |
| Maximum co-scheduled UEs | 8 |
| Maximum UE rank | 2 |
| Maximum MIMO layers | 16 |
| CSI | RI, sub-band CQI, PMI |
| Maximum scheduled candidate UEs | 10 |
| TD scheduler | Proportional Fair |
| UE speed | 3 km/h |
| UE height | 1.5 m |
| UE receiver | MRC |
| UE antennas | 2 dual-polarized Rx antennas |

### Evaluation Traffic

Two separate evaluation scenarios are used.

#### Full Buffer

All UEs use Full Buffer traffic.

#### FTP Model 3

All UEs use FTP Model 3 traffic with:

- Packet size: 0.5 MB
- Packet arrival rate: 20 packets/s

---

## 5. Training Configuration

Training deliberately differs from evaluation.

Only the following parameters are changed from the evaluation configuration.

| Parameter | Training value |
|---|---:|
| Number of UEs | 420 |
| Random drops | 1 |
| Bandwidth | 6.6 MHz |
| Number of RBs | 18 |
| Number of RBGs | 18 |
| Maximum co-scheduled UEs | 4 |

### Training Traffic

The training population contains:

- 50% Full Buffer UEs
- 50% FTP Model 3 UEs

Training FTP3 parameters:

- Packet size: 1.5 kB
- Packet arrival rate: 500 packets/s

The model is therefore intentionally evaluated under a substantial
distribution shift:

Training:
- 6.6 MHz
- 4 MU-MIMO UE layers
- heterogeneous FB + FTP3 traffic

Evaluation:
- 100 MHz
- 8 MU-MIMO UE layers
- homogeneous FB or homogeneous FTP3 traffic


---

## 6. Overall Scheduling Pipeline

The scheduling architecture is decomposed into:

1. Time Domain Scheduling (TDS)
2. Frequency Domain Scheduling (FDS)
3. Spatial Domain Scheduling (SDS)

### TDS

The TD scheduler shortlists candidate UEs.

Paper configuration:

- TD scheduler: Proportional Fair
- Maximum candidate UEs: |U| = 10

### FDS

Frequency resources are allocated with RBG granularity.

The paper focuses on downlink non-contiguous allocation.

### SDS

Additional UEs are spatially paired on each RBG using MU-MIMO.

The scheduling procedure conceptually operates as:

TDS candidate selection
        ->
SU-MIMO frequency allocation
        ->
MU-MIMO spatial-layer iterations
        ->
final RZF precoder calculation
        ->
MCS selection
        ->
throughput realization

Maximum evaluation user layers:

|L| = 8

Maximum UE rank:

2

Therefore maximum MIMO transmission layers:

8 x 2 = 16.

---

## 7. Reference Heuristic Schedulers

Two heuristic SDS schedulers must be reproduced before training RL models.

### 7.1 Baseline SDS

Candidate UEs are sorted according to their PF metric.

For each RBG and additional MU-MIMO layer:

1. Iterate through candidates in PF order.
2. Tentatively co-schedule a candidate with users already assigned.
3. Recompute the corresponding MU-MIMO throughput estimate.
4. Accept the first candidate that increases total RBG throughput.
5. Continue adding spatial users until:
   - no suitable candidate remains, or
   - no additional MU-MIMO gain is expected.

This scheduler is iterative but not exhaustive.

### 7.2 PF Greedy SDS

For each RBG and MU-MIMO layer:

1. Evaluate all valid candidate UEs.
2. Recompute beamforming and achievable throughput for each candidate.
3. Calculate the resulting PF sum metric.
4. Select the candidate producing the highest valid PF sum metric.
5. Require that the MU-MIMO throughput estimate improves.

This performs an exhaustive search within the current spatial layer,
but not an exhaustive search over every possible complete multi-layer
UE combination.

---

## 8. Deep Scheduler State

Candidate UE feature segments are concatenated to construct the scheduler state.

### 8.1 Per-UE Features

For candidate UE u:

#### Past Average Throughput

R_u = (1 - epsilon) * p_u + epsilon * R_u_previous

Normalized as:

R_hat_u = R_u / R_max

where:

- p_u = instantaneous UE throughput
- epsilon = forgetting factor
- R_max = normalization value

#### UE Rank

h_hat_u = h_u / 2

because the maximum UE rank is 2.

#### Already Allocated RBGs

d_hat_u = d_u / d_max

where d_u is the number of RBGs already assigned to UE u during
the previous MU-MIMO layer iteration.

#### Downlink Buffer Status

b_hat_u = b_u / b_max

#### Wideband CQI

o_hat_u =
    (o_u - o_min)
    /
    (o_max - o_min)

---

### 8.2 Per-RBG / Per-UE Features

For RBG m and candidate UE u:

#### Sub-band CQI

g_hat_m_u = g_m_u / g_bar_m_u

#### Maximum Precoder Cross-Correlation

For candidate u, cross-correlation is computed against users already
co-scheduled on RBG m.

The scheduler uses the maximum correlation against the currently
scheduled users as an input feature.

This feature is derived from UE precoder matrices.

---

## 9. 1LDS State

For every candidate UE, 1LDS receives:

Five general UE features:

1. Past average throughput
2. Rank
3. Already allocated RBG count
4. DL buffer status
5. Wideband CQI

and, for every RBG:

6. Sub-band CQI
7. Maximum precoder cross-correlation

Input size:

|U| * (5 + 2 * N_RBG)

With:

|U| = 10
N_RBG = 18

the input dimension is:

10 * (5 + 36) = 410

Therefore:

1LDS state shape = 410 values.

---

## 10. 2LDS State

2LDS operates on one RBG decision at a time.

Its input contains eight UE-specific features per candidate plus
one global value describing the number of already co-scheduled users.

Input size:

|U| * 8 + 1

Therefore:

10 * 8 + 1 = 81

2LDS state shape = 81 values.

---

## 11. Action Space

For every RBG / user-layer scheduling decision:

A = {1, ..., |U| + 1}

where:

|U| = 10 candidate UEs

and the additional action means:

NO ALLOCATION.

Thus there are:

11 possible actions per RBG.

Action masking prevents invalid decisions, including:

- selecting a UE already scheduled on the same RBG
- selecting an empty/padded candidate entry

---

## 12. 1LDS Action

1LDS processes all RBGs of one MU-MIMO user layer simultaneously.

Actor output size:

N_RBG * (|U| + 1)

Therefore:

18 * 11 = 198 logits.

The output is reshaped as:

(18, 11)

meaning:

18 RBGs
x
11 possible actions per RBG.

A separate categorical distribution is constructed for each RBG.

One neural-network inference therefore generates the complete RBG
allocation for one MU-MIMO user layer.

Evaluation inferences per TTI:

|L| = 8

Training inferences per TTI:

|L| = 4

---

## 13. 2LDS Action

2LDS makes a single RBG allocation decision during each inference.

Output size:

|U| + 1 = 11

Evaluation inferences per TTI:

8 * 18 = 144

Training inferences per TTI:

4 * 18 = 72

---

## 14. PPO Architecture and Hyperparameters

The v3 PPO configuration is:

| Parameter | Value |
|---|---:|
| Discount factor gamma | 0.95 |
| Epochs per sample | 1 |
| Mini-batch / update size | 128 |
| Actor learning rate | 0.0001 |
| Critic learning rate | 0.0002 |
| Hidden layers | 2 |
| Hidden units per layer | 32 |
| Activation | ReLU |
| Optimizer | Adam |
| Expert demonstration buffer | 4000 |
| Reward scaling factor k | 0.2 |
| PPO clipping epsilon | 0.2 |

This configuration supersedes the PPO parameters reported in v1.

---

## 15. PPO Actor

1LDS PPO actor input:

410 values

Hidden architecture:

410
 ->
32 ReLU
 ->
32 ReLU
 ->
198 logits

The 198 logits are reshaped to:

(18, 11)

and normalized separately across the 11 candidate actions for
each RBG.

---

## 16. PPO Critic

The PPO critic estimates the state value:

V(s)

It uses the scheduler state as input and produces one scalar value.

Exact separation/shared-layer implementation details not explicitly
specified by the paper must be documented when implemented.

---

## 17. PPO Reward

The v3 PPO reward differs substantially from the original v1 reward.

For RBG m and MU-MIMO layer l:

For the first user layer:

reward = P * v_m

For later user layers:

reward = k * v_m

where:

P = G / G_max

and G is the geometric mean of UE throughput.

The indicator v_m is:

- -1 if a better allocation choice exists according to the greedy PF search
- +1 otherwise

and:

k = 0.2

The reward corresponding to decisions in TTI t is based on throughput
realized during TTI t and becomes available for learning at the start
of TTI t+1.

---

## 18. PPO Expert Guidance

The PPO scheduler is not trained as vanilla PPO.

An expert PF scheduler generates expert scheduling decisions.

Two buffers are maintained:

### Agent PPO Buffer

Size:

128

Contains on-policy experience required for PPO updates.

### Expert Demonstration Buffer

Size:

4000

Contains:

(state, expert_action)

pairs.

Training alternates between:

1. PPO optimization on the agent's on-policy experiences.
2. Jensen-Shannon-divergence optimization using expert demonstrations.

The expert loss therefore directly modifies the actor in addition to
the regular PPO update.

---

## 19. PPO Data Augmentation

Candidate UE feature segments can be permuted.

When candidate order is permuted:

- the corresponding action indices must also be permuted
- reward must remain unchanged

This makes candidate identity/order less likely to become an unintended
feature learned by the model.

The number of permutations is represented by N_Pi.

---

## 20. Off-Policy Algorithms

The paper evaluates SACD and DSACD.

### Common Off-Policy Hyperparameters

| Parameter | Value |
|---|---:|
| Discount factor gamma | 0 |
| Epochs per sample | 1 |
| Mini-batch size | 32 |
| Actor learning rate | 0.0001 |
| Critic learning rate | 0.0001 |
| Hidden layers | 2 |
| Hidden units | 32 |
| Activation | ReLU |
| Optimizer | Adam |
| Replay buffer size | N_gNB * 1000 |
| Prioritized replay omega | 0.5 |
| Target smoothing coefficient | 0.001 |

### Architecture Selection

Paper's preferred off-policy configurations:

- 1LDS -> DSACD
- 2LDS -> SACD

### Target Entropy

For 1LDS:

beta = 0.999

For 2LDS:

beta = 0.4

---

## 21. DSACD

DSACD maintains:

- one actor
- two online critics
- two target critics

Distributional critics represent multiple quantiles for every action.

Number of quantiles:

N = 16

The critics use quantile Huber loss.

Huber threshold:

k = 1

The actor itself remains non-distributional.

Prioritized replay uses critic TD errors to assign transition priorities.

Action masks are stored with replay-buffer transitions because the valid
action set changes with scheduler state.

---

## 22. Off-Policy Reward

For candidate UE u, RBG m, and MU-MIMO layer l:

For layer 1:

reward = T_u_m_l / R_u

For later layers:

reward =
    T_u_m_l / R_u
    -
    T_u_m_(l-1) / R_u

where:

- T is achievable rate
- R_u is past average throughput

Rewards are subsequently normalized/clipped approximately to [-1, 1].

The off-policy discount factor is intentionally:

gamma = 0

because the learning objective emphasizes the immediate PF improvement
of each scheduling decision.

---

## 23. Centralized Training

Training is centralized.

A single shared model receives experiences collected across all simulated
gNBs.

The resulting actor is then intended to operate independently at individual
base stations during inference.

Training samples begin only after TTI 100 in order to allow the simulated
system to move beyond its initialization transient.

For the training setup:

4 user layers
x
18 RBGs
x
21 cells

gives:

1512 scheduling samples per TTI

for the per-RBG/layer collection procedure described by the paper.

---

## 24. Evaluation Metrics

At minimum reproduce:

1. User throughput distribution
2. 5th-percentile user throughput
3. Median user throughput
4. Geometric-mean user throughput
5. User Perceived Throughput for FTP3
6. Number of co-scheduled UEs per RBG
7. Neural-network inference latency

The main proportional-fairness KPI used by the paper is geometric-mean
user throughput.

Results should be collected across 10 independent random drops for the
main evaluation.

---

## 25. Published Result Targets

These values are validation targets, not values that our implementation
should be forced to match artificially.

### Full Buffer: Gain over Baseline

| Scheduler | 5th percentile | Median | Geomean |
|---|---:|---:|---:|
| PF Greedy | 11.1% | 1.3% | 9.5% |
| 1LDS PPO | -17.6% | 2.2% | 6.4% |
| 1LDS DSACD | 46.7% | 0.2% | 13.7% |
| 2LDS SACD | 20.6% | 3.1% | 15.6% |

### FTP Model 3: Gain over Baseline

| Scheduler | 5th percentile | Median | Geomean |
|---|---:|---:|---:|
| PF Greedy | 7.9% | 1.4% | 3.5% |
| 1LDS PPO | 85.5% | 3.5% | 18.5% |
| 1LDS DSACD | 97.5% | 4.5% | 18.3% |
| 2LDS SACD | 106.3% | 3.8% | 18.6% |

---

## 26. Known Public-Reproduction Gaps

The following quantities are currently considered UNSPECIFIED or
insufficiently specified in the public paper and must be resolved before
claiming exact reproduction.

### Wireless / Simulator

- Exact proprietary system-level simulator implementation
- Complete channel-model implementation details beyond referenced 3GPP setup
- Exact RZF regularization implementation
- Exact link-to-system mapping implementation
- Exact CQI generation procedure
- Exact PMI codebook / reporting implementation
- Exact RI reporting behavior
- Exact MCS table and selection implementation details
- Exact scheduler timing implementation
- Exact interference implementation details

### State Normalization

- PF throughput forgetting factor epsilon
- R_max
- b_max
- sub-band CQI normalization scalar g_bar
- initialization of past average throughput

### PPO

- Entropy coefficient xi
- G_max numerical value / estimation procedure
- number of UE permutations N_Pi
- exact expert-policy probability representation for JSD
- JSD optimizer learning-rate/weight interpretation
- GAE lambda
- detailed actor/critic parameter-sharing implementation
- exact initialization method
- random seeds

### SACD / DSACD

- initial entropy coefficient alpha
- entropy learning rate
- exact PER importance-sampling annealing schedule
- PER epsilon
- some initialization details

### Baselines

- complete PF TDS implementation details
- exact PF averaging constant
- some tie-breaking behavior
- exact RZF recomputation details

These gaps must be treated as experimental variables or documented
implementation assumptions.

---

## 27. Implementation Order

Implementation should proceed in the following order:

### Phase A — Simulator Validation

1. Open system-level simulation environment
2. 3GPP DUMa topology
3. Channel and interference
4. CQI / PMI / RI generation
5. Traffic and buffers
6. Link adaptation
7. Throughput validation

### Phase B — Classical Scheduler

1. PF TDS
2. SU-MIMO FDS
3. RZF
4. Baseline SDS
5. PF Greedy SDS

No reinforcement learning should be introduced until these components
produce plausible results.

### Phase C — Deep Scheduler Interface

1. State builder
2. State normalization
3. Action masking
4. 1LDS interface
5. 2LDS interface

### Phase D — PPO

1. Actor
2. Critic
3. PPO rollout/update
4. Paper reward
5. PF expert
6. Expert replay buffer
7. JSD update
8. UE permutation augmentation

### Phase E — Off-Policy

1. SACD
2. Prioritized replay
3. Adaptive entropy
4. DSACD distributional critics

### Phase F — Reproduction

Compare reproduced performance against the published results.

### Phase G — Stress Testing

Only after baseline reproduction:

- mobility shift
- CSI aging
- CQI errors
- PMI errors
- RI errors
- UE-load shift
- traffic-model shift
- bandwidth shift
- MU-MIMO-layer shift
- candidate-set shift
- interference shift
- topology shift
- observation corruption
- inference delay

Research questions should be derived from experimentally observed failure
modes rather than proposed before reproduction.

