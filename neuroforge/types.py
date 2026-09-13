from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Protocol

import numpy as np


class Action(IntEnum):
    WAIT = 0
    RETRY = 1
    ESCALATE = 2


class HiddenCondition(StrEnum):
    NORMAL = "normal_progress"
    SLOW = "legitimately_slow"
    DEPENDENCY = "delayed_dependency"
    TRANSIENT = "transient_failure"
    PERMANENT = "permanent_failure"
    LOOPING = "looping"
    FALSE_PROGRESS = "false_apparent_progress"
    RECOVERING = "spontaneous_recovery"
    HISTORY_RETRY = "history_dependent_retry"
    HISTORY_ESCALATE = "history_dependent_escalate"


@dataclass(frozen=True)
class Observation:
    timestamp: float
    elapsed: float
    since_progress: float
    activity_rate: float
    output_rate: float
    tool_success_rate: float
    recent_failures: int
    retries: int
    repetition: float
    progress_signal: float
    accumulated_cost: float
    last_action: Action
    context_signal: float = 0.0

    def vector(self) -> np.ndarray:
        return np.asarray([
            np.log1p(self.elapsed) / 5.0,
            np.log1p(self.since_progress) / 5.0,
            self.activity_rate,
            self.output_rate,
            self.tool_success_rate,
            min(self.recent_failures, 5) / 5.0,
            min(self.retries, 4) / 4.0,
            self.repetition,
            self.progress_signal,
            np.log1p(self.accumulated_cost) / 5.0,
            float(self.last_action == Action.RETRY),
            float(self.last_action == Action.ESCALATE),
            self.context_signal,
        ], dtype=np.float64)


@dataclass(frozen=True)
class EpisodeResult:
    completed: bool
    severe_failure: bool
    total_cost: float
    escalation_count: int
    escalation_cost: float
    retry_count: int
    retry_cost: float
    wasted_compute: float
    unnecessary_interventions: int
    missed_interventions: int
    detection_delay: float | None
    completion_time: float
    steps: int


class Policy(Protocol):
    name: str
    def reset(self) -> None: ...
    def act(self, observation: Observation) -> Action: ...
