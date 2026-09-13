# Decision Log

## 2026-09-12 — Separate hypotheses

H1, H2, and H3 require independent evidence. Experiment 001 tests only H1.

## 2026-09-12 — Interactive evaluation

Recorded classification cannot establish control value because interventions
change trajectories. Every scored policy acts within the simulator.

## 2026-09-12 — Held-out mechanisms

The test set holds out complete workflow families and failure mechanisms,
including changed retry effectiveness. Random-seed-only splits are forbidden.

## 2026-09-12 — No aggregate utility headline

Metric components are reported separately to prevent a convenient weighting
from hiding severe failures or wasted work.

## 2026-09-12 — Minimal recurrent dependency

Experiment 001 uses a fully trainable NumPy GRU so the recurrent core receives
gradient updates without introducing a GPU-framework dependency.

## 2026-09-12 — Oracle-labeled causal training

Learned policies train on causal observations paired with privileged simulator
oracle labels. They no longer imitate the rules baseline. Exploratory behavior
generates retry-history coverage; hidden state never enters policy features.

## 2026-09-12 — Cluster uncertainty and memory ablation

Confidence intervals resample seed clusters. A no-memory GRU ablation separates
benefit from the learned nonlinear decision surface from benefit attributable
to persistent recurrent state.

## 2026-09-13 — Exact long-history counterfactual pair

Paired scenarios share random telemetry and an identical current observation at
failure onset. A signed cue visible only much earlier determines whether RETRY
or ESCALATE is correct. An automated test verifies both observational identity
and differing oracle actions.
