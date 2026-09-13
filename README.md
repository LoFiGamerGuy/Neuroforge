# NeuroForge

NeuroForge is a falsification-oriented research platform for asking whether a
compact persistent controller improves the reliability/cost tradeoff of
supervising long-running, partially observed workflows.

Experiment 001 is an **interactive** benchmark: `WAIT`, `RETRY`, and `ESCALATE`
change future workflow trajectories. Policies never receive hidden condition
labels (except the explicitly isolated oracle upper bound).

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
python -m neuroforge.benchmark --seeds 1 2 3 --episodes 140 --output results
```

The command trains where necessary, evaluates all five controllers on held-out
workflow families and failure mechanisms, and writes JSON, CSV, Markdown, and
an exact experiment manifest. For a fast smoke test use `--episodes 12`.

## Initial policies

1. Oracle (hidden-state upper bound; never a deployable result)
2. Deterministic rules/timers/retry budget
3. Statistical hazard/change detector
4. Rolling-feature logistic classifier
5. Small GRU with explicit elapsed-time inputs

The GRU is trained end-to-end with deterministic sequence BPTT and Adam
implemented in NumPy. This keeps Experiment 001 reproducible without a GPU
framework while ensuring that the recurrent state itself is learned.

See [the charter](docs/RESEARCH_CHARTER.md), [experiment specification](docs/EXPERIMENT_001.md),
and [decision log](docs/DECISIONS.md).

## Current finding

Learned policies outperform the initial rules on the synthetic holdout. After
adding an exact long-history counterfactual pair, the persistent GRU also beats
its no-memory ablation and uniquely avoids an unnecessary retry when the needed
cue occurred much earlier. This is controlled synthetic evidence—not real-world
validation—for useful recurrent persistence. See `docs/BASELINE_RESULTS.md`.
