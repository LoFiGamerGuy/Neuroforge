# Experiment 001 — Interactive Workflow Supervision

## Question

Can temporal state improve WAIT/RETRY/ESCALATE decisions on partially observed
workflows without losing reliability or increasing total cost?

## Design fixed before optimization

The simulator contains hidden conditions but exposes only causal telemetry
available at decision time. Actions modify transition outcomes and costs.
Training scenarios and test scenarios differ by both workflow family and
failure mechanism. Test includes delayed dependency, spontaneous recovery,
novel deadlock, unseen slow computation, and a regime shift where retry
effectiveness falls from the training regime.

Near-identical inactivity telemetry is deliberately ambiguous: prior failures,
repetition, progress history, and interventions distinguish slow work,
dependency waits, loops, and deadlock. Policies receive sequences online and no
future-derived features. The oracle alone accesses hidden state and is reported
only as an upper bound.

The long-history pair emits a one-step signed context cue early in the episode.
Much later, the two workflows produce exactly identical current observation
vectors, but one fault is retryable and the other requires escalation. This
tests retained temporal context rather than classification from engineered
rolling summaries.

## Controllers and tuning

Policies are implemented in the required order. Learned readouts receive the
same observation vector and deterministic training data. Both learned methods
have one fixed configuration and no test-set tuning. Future comparisons must
publish equal tuning budgets.

## Metrics

Completion and severe failure are primary reliability outcomes. Total cost,
escalation/retry counts and costs, wasted compute, unnecessary and missed
interventions, detection delay, and median/p95 completion time are all emitted
without collapsing them into a convenient scalar.

## Reproduction

`python -m neuroforge.benchmark --seeds 1 2 3 --episodes 140 --output results`

The manifest records seeds, scenario split, runtime, timestamp, and a source
digest. JSON/CSV are machine readable; Markdown is the comparative report.

Confidence intervals bootstrap complete seed clusters rather than treating all
episodes as independent. `gru_no_memory` is a diagnostic ablation using the
trained GRU weights while resetting recurrent state before every action; it is
not an additional architecture competitor.

## Limits

This synthetic environment establishes apparatus validity, not external
validity for real agent workflows. Training labels use privileged simulator
state; that is appropriate for this controlled benchmark but such labels are
not assumed available in deployment. No claim about H2 or H3 is permitted from
this experiment.
