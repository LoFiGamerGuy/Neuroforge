# NeuroForge Research Charter

NeuroForge is a falsification-oriented program. Results against weak baselines,
attractive demonstrations, or post-hoc metrics are not treated as evidence.

## H1 — Persistent controller

**Hypothesis:** A compact persistent controller can improve the reliability/cost
tradeoff of long-running workflow supervision relative to competent
ordinary-software baselines.

**Evidence:** a pre-specified, reproducible improvement over well-tuned rules,
timers, retry budgets, statistical detection, and rolling-feature baselines on
held-out workflow families and failure mechanisms, with reliability, cost,
severe failures, and latency reported separately across multiple seeds.

**Not evidence:** beating an always-on expensive model; fitting recorded traces;
lowering intervention count while missing failures; success on only familiar
workflow families; or outperforming deliberately weak rules.

## H2 — Biological structure

**Hypothesis:** Biological network structures or motifs may improve such
controllers relative to properly matched synthetic architectures.

**Evidence:** an advantage that survives controls for size, degree, sparsity,
sign/weight distributions, dynamics, encoders/decoders, tuning budget, and seeds,
and transfers beyond one narrowly selected task.

**Not evidence:** support for H1; a biological network merely solving a task;
beating a dense MLP; a compelling animal simulation; or uncontrolled topology
comparisons. H2 is not tested in Experiment 001.

## H3 — Spiking efficiency

**Hypothesis:** Spiking/event-driven execution may improve hardware efficiency.

**Evidence:** measured end-to-end latency and energy improvements on relevant
deployment hardware at matched outcome quality, including encoding and I/O.

**Not evidence:** sparse activation counts, neuron counts, simulator speed, or
energy estimates detached from target hardware. H3 is not tested in Experiment
001.

Success or failure of any hypothesis does not establish another.

