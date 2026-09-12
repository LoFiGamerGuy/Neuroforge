# Experiment 001 Reference Run

Command: `python -m neuroforge.benchmark --seeds 1 2 3 4 5 --episodes 120 --output results`

| Policy | Completion | Severe failure | Mean cost | Median time | p95 time |
|---|---:|---:|---:|---:|---:|
| oracle | 100.0% | 0.0% | 16.08 | 13.0 | 45.0 |
| rules | 93.7% | 6.3% | 16.20 | 19.0 | 55.0 |
| statistical | 70.0% | 30.0% | 16.01 | 47.0 | 55.0 |
| classifier | 99.3% | 0.7% | 14.96 | 8.0 | 50.0 |
| gru | 99.3% | 0.7% | 14.33 | 8.0 | 50.0 |

**Primary finding:** both learned controllers improved reliability, mean cost,
and latency relative to deterministic rules on this synthetic holdout. The GRU
matches the classifier's reliability and latency while reducing mean cost by
0.63, making it the deployable leader under the predefined ordering. This is
initial evidence that recurrent state may help, not a conclusive H1 result.
H1 remains provisional pending confidence intervals, comparable tuning, stronger
conventional baselines, and real-workflow validation.
