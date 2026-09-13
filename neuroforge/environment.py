from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenarios import Scenario
from .types import Action, EpisodeResult, HiddenCondition, Observation


@dataclass
class WorkflowEnv:
    scenario: Scenario
    seed: int
    max_steps: int = 55

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> Observation:
        self.t = self.progress = self.last_progress = 0
        self.retries = self.escalations = self.recent_failures = 0
        self.cost = self.retry_cost = self.escalation_cost = self.wasted = 0.0
        self.unnecessary = self.missed = 0
        self.failure_seen_at: int | None = None
        self.intervened_at: int | None = None
        self.done = self.completed = self.severe = False
        self.last_action = Action.WAIT
        return self._observe()

    def _active_condition(self) -> HiddenCondition:
        if self.t < self.scenario.onset:
            return HiddenCondition.NORMAL
        if self.scenario.recovery_at is not None and self.t >= self.scenario.recovery_at:
            return HiddenCondition.NORMAL
        return self.scenario.condition

    def _observe(self) -> Observation:
        c = self._active_condition()
        normal = c == HiddenCondition.NORMAL
        slow = c == HiddenCondition.SLOW
        looping = c == HiddenCondition.LOOPING
        false = c == HiddenCondition.FALSE_PROGRESS
        failed = c in (HiddenCondition.TRANSIENT, HiddenCondition.PERMANENT, HiddenCondition.HISTORY_RETRY, HiddenCondition.HISTORY_ESCALATE)
        activity = np.clip(self.rng.normal(.72 if normal else (.50 if slow else .67 if looping or false else .10), .08), 0, 1)
        output = np.clip(self.rng.normal(.68 if normal else (.32 if slow else .72 if false else .58 if looping else .08), .09), 0, 1)
        success = np.clip(self.rng.normal(.92 if normal or slow else .25 if failed else .62, .08), 0, 1)
        repetition = np.clip(self.rng.normal(.82 if looping else .48 if false else .12, .08), 0, 1)
        signal = np.clip(self.rng.normal(.72 if normal else .08 if false or looping else .18, .12), 0, 1)
        cue=self.scenario.early_cue if self.t==1 else 0.0
        return Observation(float(self.t), float(self.t), float(self.t-self.last_progress), float(activity), float(output), float(success), self.recent_failures, self.retries, float(repetition), float(signal), self.cost, self.last_action, cue)

    def step(self, action: Action) -> tuple[Observation, bool]:
        if self.done:
            raise RuntimeError("episode is complete")
        c = self._active_condition()
        needs = c in (HiddenCondition.TRANSIENT, HiddenCondition.PERMANENT, HiddenCondition.LOOPING, HiddenCondition.FALSE_PROGRESS, HiddenCondition.HISTORY_RETRY, HiddenCondition.HISTORY_ESCALATE)
        if needs and self.failure_seen_at is None:
            self.failure_seen_at = self.t
        self.last_action = action
        if action == Action.RETRY:
            self.retries += 1; self.retry_cost += 2.0; self.cost += 2.0
            if not needs: self.unnecessary += 1
            if self.rng.random() < self.scenario.retry_effectiveness:
                self.progress += 5; self.last_progress = self.t; self.recent_failures = 0
                c = HiddenCondition.NORMAL
            else:
                self.recent_failures += 1; self.wasted += 2.0
            if needs and self.intervened_at is None: self.intervened_at = self.t
        elif action == Action.ESCALATE:
            self.escalations += 1; self.escalation_cost += 12.0; self.cost += 12.0
            if not needs: self.unnecessary += 1
            if needs and self.intervened_at is None: self.intervened_at = self.t
            self.progress = self.scenario.duration; self.completed = self.done = True
        else:
            self.cost += 0.25
            if needs: self.missed += 1; self.wasted += 1.0

        if not self.done:
            p = .82 if c == HiddenCondition.NORMAL else .30 if c == HiddenCondition.SLOW else .0
            if self.rng.random() < p:
                self.progress += 1; self.last_progress = self.t
            if c in (HiddenCondition.TRANSIENT, HiddenCondition.PERMANENT, HiddenCondition.HISTORY_RETRY, HiddenCondition.HISTORY_ESCALATE): self.recent_failures += 1
            self.t += 1
            if self.progress >= self.scenario.duration: self.completed = self.done = True
            if self.t >= self.max_steps:
                self.done = True; self.severe = not self.completed
        return self._observe(), self.done

    def result(self) -> EpisodeResult:
        delay = None if self.failure_seen_at is None or self.intervened_at is None else float(self.intervened_at-self.failure_seen_at)
        return EpisodeResult(self.completed, self.severe, self.cost, self.escalations, self.escalation_cost, self.retries, self.retry_cost, self.wasted, self.unnecessary, self.missed, delay, float(self.t), self.t)

    @property
    def hidden_condition(self) -> HiddenCondition:
        return self._active_condition()
