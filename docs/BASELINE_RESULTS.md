# Experiment 001 Reference Run

Command: `python -m neuroforge.benchmark --seeds 1 2 3 4 5 --episodes 140 --output results`

| Policy | Completion [95% CI] | Severe failure | Mean cost [95% CI] | p95 time |
|---|---:|---:|---:|---:|
| oracle | 100.0% [100.0%, 100.0%] | 0.0% | 15.88 [15.77, 15.99] | 45 |
| rules | 95.7% [93.9%, 97.7%] | 4.3% | 17.19 [17.15, 17.22] | 55 |
| statistical | 79.1% [77.6%, 81.0%] | 20.9% | 17.75 [17.74, 17.77] | 55 |
| classifier | 99.7% [99.4%, 100.0%] | 0.3% | 15.38 [15.36, 15.40] | 49 |
| gru | 99.7% [99.4%, 100.0%] | 0.3% | **15.10 [15.08, 15.12]** | 49 |
| gru without memory | 99.7% [99.4%, 100.0%] | 0.3% | 15.39 [15.37, 15.41] | 49 |

**Primary finding:** after adding an exact long-history pair, the persistent GRU
matches learned-policy reliability while using 0.29 less mean cost than its
no-memory ablation. On the paired scenarios, all three learned policies retry
the refreshable fault; only the persistent GRU avoids retrying the terminal
fault after its distinguishing cue has disappeared. This is controlled evidence
that recurrent memory affects useful behavior in the simulator. H1 remains
provisional because the effect comes from a constructed synthetic pair and has
not yet been replicated on real workflows.

| Policy | Retryable-fault retries | Terminal-fault retries |
|---|---:|---:|
| classifier | 1.00 | 1.00 |
| persistent GRU | 1.00 | **0.00** |
| GRU without memory | 1.00 | 1.00 |
