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

Experiment 001 uses a deterministic GRU feature map with a learned readout so
the baseline is reproducible without a GPU framework. A fully trainable GRU is
explicitly documented as the next strengthening step, not silently implied.

