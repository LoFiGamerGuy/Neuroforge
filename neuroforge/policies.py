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


class ClassifierPolicy:
    name = "classifier"
    def __init__(self): self.model = LogisticRegression(max_iter=500, random_state=0, class_weight="balanced")
    def fit(self, sequences):
        X=[]; y=[]
        for seq in sequences:
            for o,label in seq: X.append(o.vector()); y.append(int(label))
        self.model.fit(np.asarray(X), np.asarray(y)); return self
    def reset(self): pass
    def act(self, o): return Action(int(self.model.predict(o.vector()[None])[0]))


class GRUPolicy:
    """Small GRU trained end-to-end with deterministic NumPy BPTT/Adam."""
    name = "gru"
    def __init__(self, seed=41, hidden=16):
        rng=np.random.default_rng(seed); d=12; self.seed=seed; scale=.16
        self.params={
            "Wz":rng.normal(0,scale,(hidden,d)), "Uz":rng.normal(0,scale,(hidden,hidden)), "bz":np.zeros(hidden),
            "Wr":rng.normal(0,scale,(hidden,d)), "Ur":rng.normal(0,scale,(hidden,hidden)), "br":np.zeros(hidden),
            "Wh":rng.normal(0,scale,(hidden,d)), "Uh":rng.normal(0,scale,(hidden,hidden)), "bh":np.zeros(hidden),
            "Wo":rng.normal(0,scale,(3,hidden)), "bo":np.zeros(3),
        }
        self.hidden=hidden; self.reset()
    @staticmethod
    def _sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
    def _advance(self,x,cache=False):
        p=self.params; hp=self.h.copy()
        z=self._sigmoid(p["Wz"]@x+p["Uz"]@hp+p["bz"])
        r=self._sigmoid(p["Wr"]@x+p["Ur"]@hp+p["br"])
        n=np.tanh(p["Wh"]@x+p["Uh"]@(r*hp)+p["bh"])
        self.h=(1-z)*hp+z*n; logits=p["Wo"]@self.h+p["bo"]
        return (logits,(x,hp,z,r,n,self.h.copy())) if cache else logits
    def _loss_grads(self,seq,class_weights):
        self.reset(); caches=[]; labels=[]; loss=0.0
        for o,label in seq:
            logits,c=self._advance(o.vector(),True); logits-=logits.max()
            prob=np.exp(logits); prob/=prob.sum(); w=class_weights[int(label)]
            loss-=w*np.log(prob[int(label)]+1e-12); caches.append((c,prob,w)); labels.append(int(label))
        grads={k:np.zeros_like(v) for k,v in self.params.items()}; dh_next=np.zeros(self.hidden); p=self.params
        for ((x,hp,z,r,n,h),prob,w),label in zip(reversed(caches),reversed(labels)):
            dl=prob.copy(); dl[label]-=1; dl*=w
            grads["Wo"]+=np.outer(dl,h); grads["bo"]+=dl; dh=p["Wo"].T@dl+dh_next
            dz=dh*(n-hp); dn=dh*z; dhp=dh*(1-z)
            dan=dn*(1-n*n); grads["Wh"]+=np.outer(dan,x); grads["Uh"]+=np.outer(dan,r*hp); grads["bh"]+=dan
            drhp=p["Uh"].T@dan; dr=drhp*hp; dhp+=drhp*r
            dar=dr*r*(1-r); grads["Wr"]+=np.outer(dar,x); grads["Ur"]+=np.outer(dar,hp); grads["br"]+=dar; dhp+=p["Ur"].T@dar
            daz=dz*z*(1-z); grads["Wz"]+=np.outer(daz,x); grads["Uz"]+=np.outer(daz,hp); grads["bz"]+=daz; dhp+=p["Uz"].T@daz
            dh_next=dhp
        denom=max(len(seq),1)
        return loss/denom,{k:np.clip(v/denom,-3,3) for k,v in grads.items()}
    def fit(self,sequences,epochs=24,lr=.012):
        counts=np.bincount([int(y) for s in sequences for _,y in s],minlength=3).astype(float)
        weights=counts.sum()/np.maximum(3*counts,1); weights/=weights.mean()
        m={k:np.zeros_like(v) for k,v in self.params.items()}; v={k:np.zeros_like(x) for k,x in self.params.items()}
        rng=np.random.default_rng(self.seed); step=0
        for _ in range(epochs):
            for idx in rng.permutation(len(sequences)):
                _,g=self._loss_grads(sequences[idx],weights); step+=1
                for k in self.params:
                    m[k]=.9*m[k]+.1*g[k]; v[k]=.999*v[k]+.001*g[k]*g[k]
                    self.params[k]-=lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        self.reset(); return self
    def reset(self): self.h=np.zeros(self.hidden)
    def act(self,o): return Action(int(np.argmax(self._advance(o.vector()))))
