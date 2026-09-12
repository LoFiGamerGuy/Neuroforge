from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import numpy as np

from .environment import WorkflowEnv
from .scenarios import Scenario
from .types import Action, EpisodeResult, Policy


def run_episode(scenario: Scenario, seed: int, factory: Callable[[WorkflowEnv], Policy]) -> tuple[EpisodeResult, list]:
    env=WorkflowEnv(scenario,seed); policy=factory(env); policy.reset(); obs=env.reset(); history=[]
    while not env.done:
        history.append(obs); obs,_=env.step(policy.act(obs))
    return env.result(),history


def aggregate(results: list[EpisodeResult]) -> dict[str,float]:
    arr=lambda field: np.asarray([getattr(r,field) for r in results],dtype=float)
    delays=np.asarray([r.detection_delay for r in results if r.detection_delay is not None],dtype=float)
    times=arr("completion_time")
    return {
        "episodes": len(results), "completion_rate": arr("completed").mean(),
        "severe_failure_rate": arr("severe_failure").mean(), "total_cost_mean": arr("total_cost").mean(),
        "escalation_count_mean": arr("escalation_count").mean(), "escalation_cost_mean": arr("escalation_cost").mean(),
        "retry_count_mean": arr("retry_count").mean(), "retry_cost_mean": arr("retry_cost").mean(),
        "wasted_compute_mean": arr("wasted_compute").mean(),
        "unnecessary_interventions_mean": arr("unnecessary_interventions").mean(),
        "missed_interventions_mean": arr("missed_interventions").mean(),
        "detection_delay_mean": float(delays.mean()) if len(delays) else 0.0,
        "completion_time_median": float(np.median(times)), "completion_time_p95": float(np.quantile(times,.95)),
    }


def seed_cluster_intervals(seed_results: list[list[EpisodeResult]], bootstrap_seed: int=20260912) -> dict[str,list[float]]:
    """Bootstrap whole seed clusters; episodes within a seed stay together."""
    metrics=("completion_rate","severe_failure_rate","total_cost_mean","completion_time_p95")
    per_seed=[aggregate(x) for x in seed_results]
    values={m:np.asarray([s[m] for s in per_seed]) for m in metrics}
    rng=np.random.default_rng(bootstrap_seed); n=len(per_seed)
    indices=rng.integers(0,n,size=(4000,n))
    return {m:[float(x) for x in np.quantile(v[indices].mean(axis=1),[.025,.975])] for m,v in values.items()}


def episode_dict(r: EpisodeResult) -> dict: return asdict(r)
