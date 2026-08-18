# Deep Scheduler Reproduction

Independent implementation and stress-testing framework for the deep radio
resource schedulers described in:

**From Simulation to Practice: Generalizable Deep Reinforcement Learning
for Cellular Schedulers**

This repository focuses on reproducing the published scheduling framework
using an open simulation and learning stack, followed by controlled
stress-testing of the reproduced schedulers.

## Current Status

Reproduction infrastructure under development.

## Main Goals

1. Reproduce the paper's wireless scheduling setup as faithfully as possible.
2. Implement and validate the heuristic scheduling baselines.
3. Reproduce the 1LDS and 2LDS deep scheduler architectures.
4. Reproduce the PPO, SACD, and DSACD training procedures.
5. Compare reproduced results against the published results.
6. Stress-test the learned schedulers under controlled distribution shifts.