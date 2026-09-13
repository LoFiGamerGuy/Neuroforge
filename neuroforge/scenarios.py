from __future__ import annotations

from dataclasses import dataclass

from .types import HiddenCondition


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    mechanism: str
    condition: HiddenCondition
    duration: int
    onset: int
    retry_effectiveness: float
    recovery_at: int | None = None
    early_cue: float = 0.0


TRAIN_SCENARIOS = (
    Scenario("routine", "batch", "none", HiddenCondition.NORMAL, 18, 99, 0.0),
    Scenario("long_compile", "build", "slow", HiddenCondition.SLOW, 34, 99, 0.0),
    Scenario("flaky_api", "service", "transient", HiddenCondition.TRANSIENT, 30, 7, 0.80),
    Scenario("bad_config", "deploy", "permanent", HiddenCondition.PERMANENT, 45, 5, 0.0),
    Scenario("agent_loop", "coding", "repetition", HiddenCondition.LOOPING, 45, 8, 0.10),
    Scenario("noisy_output", "analysis", "false_progress", HiddenCondition.FALSE_PROGRESS, 45, 8, 0.15),
    Scenario("signed_retryable", "legacy_batch", "signed_fault", HiddenCondition.HISTORY_RETRY, 45, 10, 0.85, early_cue=1.0),
    Scenario("signed_terminal", "legacy_batch", "signed_fault", HiddenCondition.HISTORY_ESCALATE, 45, 10, 0.0, early_cue=-1.0),
)

# Entire families and mechanisms are held out, not merely random seeds.
TEST_SCENARIOS = (
    Scenario("queued_dataset", "data_pipeline", "dependency", HiddenCondition.DEPENDENCY, 38, 5, 0.05, 22),
    Scenario("self_healing_worker", "maintenance", "spontaneous_recovery", HiddenCondition.RECOVERING, 35, 6, 0.10, 19),
    Scenario("novel_deadlock", "simulation", "deadlock", HiddenCondition.PERMANENT, 48, 9, 0.0),
    Scenario("changed_retry_semantics", "migration", "regime_shift", HiddenCondition.TRANSIENT, 40, 7, 0.15),
    Scenario("unseen_slow_solver", "optimization", "slow_solver", HiddenCondition.SLOW, 39, 99, 0.0),
    Scenario("credential_refreshable", "secure_transfer", "early_context_fault", HiddenCondition.HISTORY_RETRY, 45, 10, 0.85, early_cue=1.0),
    Scenario("credential_revoked", "secure_transfer", "early_context_fault", HiddenCondition.HISTORY_ESCALATE, 45, 10, 0.0, early_cue=-1.0),
)
