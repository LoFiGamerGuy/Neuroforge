from __future__ import annotations

import argparse, csv, hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path

from .environment import WorkflowEnv
from .evaluation import aggregate, episode_dict, run_episode
from .policies import ClassifierPolicy, GRUPolicy, OraclePolicy, RulesPolicy, StatisticalPolicy
from .scenarios import TEST_SCENARIOS, TRAIN_SCENARIOS


def collect_training(seeds, episodes):
    from .types import Action
    import numpy as np
    sequences=[]
    for seed in seeds:
        for i in range(max(18,episodes//2)):
            sc=TRAIN_SCENARIOS[i%len(TRAIN_SCENARIOS)]
            env=WorkflowEnv(sc,seed*10000+i); obs=env.reset(); seq=[]
            oracle=OraclePolicy(lambda:env.hidden_condition)
            rng=np.random.default_rng(seed*10000+i+97)
            while not env.done:
                seq.append((obs,oracle.act(obs)))
                action=Action.RETRY if obs.recent_failures>0 and rng.random()<.35 else Action.WAIT
                obs,_=env.step(action)
            sequences.append(seq)
    return sequences


def benchmark(seeds,episodes):
    train=collect_training(seeds,episodes)
    classifier=ClassifierPolicy().fit(train); gru=GRUPolicy().fit(train)
    factories={
      "oracle":lambda e:OraclePolicy(lambda:e.hidden_condition), "rules":lambda e:RulesPolicy(),
      "statistical":lambda e:StatisticalPolicy(), "classifier":lambda e:classifier, "gru":lambda e:gru,
    }
    raw=[]; summary={}
    for name,factory in factories.items():
        results=[]
        for seed in seeds:
            for i in range(episodes):
                sc=TEST_SCENARIOS[i%len(TEST_SCENARIOS)]
                r,_=run_episode(sc,seed*100000+i,factory); results.append(r)
                raw.append({"policy":name,"seed":seed,"scenario":sc.name,"family":sc.family,"mechanism":sc.mechanism,**episode_dict(r)})
        summary[name]=aggregate(results)
    return summary,raw


def report(summary):
    lines=["# Experiment 001 Comparative Report","", "Held-out workflow families and failure mechanisms; lower cost/failure/latency is better.","", "| Policy | Completion | Severe failure | Mean cost | Median time | p95 time |", "|---|---:|---:|---:|---:|---:|"]
    for n,m in summary.items(): lines.append(f"| {n} | {m['completion_rate']:.1%} | {m['severe_failure_rate']:.1%} | {m['total_cost_mean']:.2f} | {m['completion_time_median']:.1f} | {m['completion_time_p95']:.1f} |")
    nonoracle={n:m for n,m in summary.items() if n!='oracle'}
    best=max(nonoracle,key=lambda n:(nonoracle[n]['completion_rate'],-nonoracle[n]['severe_failure_rate'],-nonoracle[n]['total_cost_mean']))
    lines += ["",f"**Primary finding:** `{best}` leads the deployable policies by the predefined lexicographic reliability/cost ordering."]
    if best=="rules": lines.append("**Falsification signal:** simple deterministic rules outperformed every learned controller; this result must not be hidden.")
    lines.append("The oracle is an inaccessible upper bound and is never considered deployable evidence.")
    return "\n".join(lines)+"\n"


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]); p.add_argument("--episodes",type=int,default=120); p.add_argument("--output",type=Path,default=Path("results")); a=p.parse_args(argv)
    if a.episodes<5: p.error("--episodes must be at least 5 to cover every held-out scenario")
    summary,raw=benchmark(a.seeds,a.episodes); a.output.mkdir(parents=True,exist_ok=True)
    manifest={"experiment":"001","created_utc":datetime.now(timezone.utc).isoformat(),"seeds":a.seeds,"episodes_per_seed":a.episodes,"train_scenarios":[s.name for s in TRAIN_SCENARIOS],"test_scenarios":[s.name for s in TEST_SCENARIOS],"python":platform.python_version(),"platform":platform.platform(),"source_digest":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (a.output/"report.md").write_text(report(summary))
    with (a.output/"episodes.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=raw[0]); w.writeheader(); w.writerows(raw)
    print(report(summary)); print(f"Artifacts: {a.output.resolve()}")


if __name__=="__main__": main()
