from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .types import Action, HiddenCondition, Observation


class OraclePolicy:
    name = "oracle"
    def __init__(self, hidden_state): self.hidden_state = hidden_state
    def reset(self): pass
    def act(self, o):
        c = self.hidden_state()
        # The privileged ceiling knows which silence is benign, while still
        # escalating before a benign wait can consume the episode horizon.
        if c in (HiddenCondition.NORMAL, HiddenCondition.SLOW):
            return Action.ESCALATE if o.elapsed >= 45 else Action.WAIT
        if c in (HiddenCondition.DEPENDENCY, HiddenCondition.RECOVERING):
            return Action.ESCALATE if o.since_progress >= 9 else Action.WAIT
        if c == HiddenCondition.TRANSIENT and o.retries < 1: return Action.RETRY
        return Action.ESCALATE


class RulesPolicy:
    name = "rules"
    def reset(self): pass
    def act(self, o):
        if o.repetition > .68 and o.since_progress > 3: return Action.ESCALATE
        if o.recent_failures >= 3: return Action.RETRY if o.retries < 2 else Action.ESCALATE
        if o.since_progress > 14: return Action.ESCALATE
        return Action.WAIT


class StatisticalPolicy:
    name = "statistical"
    def reset(self): self.ema = 0.0; self.n = 0
    def act(self, o):
        anomaly = .45*(1-o.tool_success_rate)+.30*o.repetition+.25*min(o.since_progress/15,1)
        self.ema = .72*self.ema+.28*anomaly; self.n += 1
        if self.n > 3 and self.ema > .62: return Action.RETRY if o.retries < 1 else Action.ESCALATE
        if self.n > 5 and self.ema > .48: return Action.RETRY if o.retries < 2 else Action.ESCALATE
        return Action.WAIT


def heuristic_label(o: Observation) -> int:
    if o.repetition > .65 and o.since_progress > 4: return int(Action.ESCALATE)
    if o.recent_failures >= 2: return int(Action.RETRY if o.retries < 1 else Action.ESCALATE)
    if o.since_progress > 13: return int(Action.ESCALATE)
    return int(Action.WAIT)


class ClassifierPolicy:
    name = "classifier"
    def __init__(self): self.model = LogisticRegression(max_iter=500, random_state=0, class_weight="balanced")
    def fit(self, sequences):
        X=[]; y=[]
        for seq in sequences:
            for o in seq: X.append(o.vector()); y.append(heuristic_label(o))
        self.model.fit(np.asarray(X), np.asarray(y)); return self
    def reset(self): pass
    def act(self, o): return Action(int(self.model.predict(o.vector()[None])[0]))


class GRUPolicy:
    """Small deterministic GRU feature map plus learned multinomial readout."""
    name = "gru"
    def __init__(self, seed=41, hidden=16):
        rng=np.random.default_rng(seed); d=12
        self.Wz=rng.normal(0,.28,(hidden,d)); self.Uz=rng.normal(0,.20,(hidden,hidden))
        self.Wr=rng.normal(0,.28,(hidden,d)); self.Ur=rng.normal(0,.20,(hidden,hidden))
        self.Wh=rng.normal(0,.28,(hidden,d)); self.Uh=rng.normal(0,.20,(hidden,hidden))
        self.model=LogisticRegression(max_iter=600, random_state=seed, class_weight="balanced")
        self.hidden=hidden; self.reset()
    @staticmethod
    def _sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
    def _advance(self,x):
        z=self._sigmoid(self.Wz@x+self.Uz@self.h); r=self._sigmoid(self.Wr@x+self.Ur@self.h)
        candidate=np.tanh(self.Wh@x+self.Uh@(r*self.h)); self.h=(1-z)*self.h+z*candidate
        return np.r_[x,self.h]
    def fit(self,sequences):
        X=[]; y=[]
        for seq in sequences:
            self.reset()
            for o in seq: X.append(self._advance(o.vector())); y.append(heuristic_label(o))
        self.model.fit(np.asarray(X),np.asarray(y)); self.reset(); return self
    def reset(self): self.h=np.zeros(self.hidden)
    def act(self,o): return Action(int(self.model.predict(self._advance(o.vector())[None])[0]))
