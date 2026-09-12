# Experiment 001 Reference Run

Command: `python -m neuroforge.benchmark --seeds 1 2 3 4 5 --episodes 120 --output results`

| Policy | Completion [95% CI] | Severe failure | Mean cost [95% CI] | p95 time |
|---|---:|---:|---:|---:|
| oracle | 100.0% [100.0%, 100.0%] | 0.0% | 16.08 [15.90, 16.24] | 45 |
| rules | 93.7% [92.7%, 94.7%] | 6.3% | 16.20 [16.18, 16.22] | 55 |
| statistical | 70.0% [67.8%, 71.5%] | 30.0% | 16.01 [15.97, 16.06] | 55 |
| classifier | 99.3% [99.2%, 99.7%] | 0.7% | 14.96 [14.93, 14.99] | 50 |
| gru | 99.3% [99.2%, 99.7%] | 0.7% | 14.33 [14.30, 14.35] | 50 |
| gru without memory | 99.3% [99.2%, 99.7%] | 0.7% | **14.14 [14.12, 14.16]** | 50 |

**Primary finding:** learned nonlinear policies improve reliability, cost, and
latency relative to the implemented rules. However, resetting the GRU state at
every decision reduces mean cost by another 0.19 without changing reliability.
Therefore **recurrent persistence has not demonstrated value and H1 is not yet
supported as stated**. The current observation vector already contains explicit
history summaries, which may make recurrent memory redundant. The next benchmark
revision must isolate long-history dependence rather than adding new architectures.
