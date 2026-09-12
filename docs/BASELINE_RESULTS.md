# Experiment 001 Reference Run

Command: `python -m neuroforge.benchmark --seeds 1 2 3 4 5 --episodes 120 --output results`

| Policy | Completion | Severe failure | Mean cost | Median time | p95 time |
|---|---:|---:|---:|---:|---:|
| oracle | 100.0% | 0.0% | 16.08 | 13.0 | 45.0 |
| rules | 93.7% | 6.3% | 16.20 | 19.0 | 55.0 |
| statistical | 70.0% | 30.0% | 16.01 | 47.0 | 55.0 |
| classifier | 61.8% | 38.2% | 42.24 | 51.0 | 55.0 |
| gru | 69.0% | 31.0% | 41.00 | 46.0 | 55.0 |

**Falsification signal:** simple deterministic rules outperformed every learned
controller in this reference run. H1 is not supported by this initial synthetic
benchmark. The environment and learned baselines must be strengthened before
drawing broader conclusions.

