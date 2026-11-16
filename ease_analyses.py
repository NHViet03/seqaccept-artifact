"""Analyses for the EASE 2027 mentor review (offline: no LLM, no Docker). Writes results/ease.json.

  uv run python ease_analyses.py

1. Refund ablation: no refund, the largest refund Theorem 4.2 allows (alpha_j min(E, 1)), the implemented capped
   refund and the uncapped refund, on recorded loops, late streams, refund stress and the worst case.
2. The 40-block shortcut: clearly broken candidates whose 40 blocks serve as truth and pool; re-split 20/20.
3. Truth vs pool rates of every distinct candidate (natural and real-flaky).
4. Units (repositories, bugs, loops, episodes, candidates, blocks) and bug-level cluster intervals.
5. Cost: blocks and recorded block time per episode, p0 calibration, full-suite B1-B3 runs.
6. Operating curve near eps at p0 = 0 and p0 = 0.2 (Kelly and mixture wallets, KL floor).
7. Persistence (FAR after the first k candidates) with bug-level cluster intervals.
"""
import json
import math
from functools import partial
from pathlib import Path

import replay
from extra_analyses import STRESS, STRESS_REPS, fullsuite_episodes, replay_rules_fixed
from rules import all_rules, seq_accept
from sim import experiment

REFUNDS = {"none": False, "max": "max", "capped": True, "uncapped": "uncapped"}
OP_D = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)
OP_REPS = 2000


def bug_of(ep):
    return ep["name"].split("-")[0].split("@")[0]


def is_shortcut(ep, v):
    """A clearly broken candidate stopped at the probe: its probe blocks are both its truth and its pool."""
    return len(v["truth"]) == ep["params"]["probe"] and v["truth"] == v["pool"]


def kl(p, q):
    a = p * math.log(p / q) if p > 0 else 0.0
    return a + (1 - p) * math.log((1 - p) / (1 - q))


def run_rule(make_rule, episodes, reps=replay.REPS):
    return [replay.replay(make_rule(ep["p0"] + replay.MARGIN), ep, reps, seed=replay.episode_seed(ep, i))
            for i, ep in enumerate(episodes)]


def resplit(ep):
    """Clearly broken candidates: their 40 blocks served as truth and pool. Split them 20/20 into separate halves."""
    fixes = {}
    for f, v in ep["fixes"].items():
        if is_shortcut(ep, v):
            b = v["truth"]
            fixes[f] = {**v, "truth": b[:len(b) // 2], "pool": b[len(b) // 2:]}
        else:
            fixes[f] = v
    return {**ep, "fixes": fixes}


def shortcut(ep):
    return [f for f, v in ep["fixes"].items() if is_shortcut(ep, v)]


def rates(episodes):
    """Distinct (bug, candidate): truth rate, pool rate, eps and truth label."""
    seen, out = set(), []
    for ep in episodes:
        eps = ep["p0"] + replay.MARGIN
        for f, lab in zip(ep["stream"], replay.labels(ep)):
            key = (ep["name"].split("-")[0], f)
            if key in seen:
                continue
            seen.add(key)
            v = ep["fixes"][f]
            out.append({"truth": sum(v["truth"]) / len(v["truth"]), "pool": sum(v["pool"]) / len(v["pool"]),
                        "n_truth": len(v["truth"]), "eps": eps, "label": lab, "shortcut": is_shortcut(ep, v)})
    return out


def single(rule, p):
    r = experiment(rule, [p], reps=OP_REPS, seed=71)
    return {"accept": r["accepts"] / r["reps"], "blocks_per_accept": r["blocks_per_accept"]}


if __name__ == "__main__":
    paths = sorted(Path("episodes").glob("*.json"))
    episodes = [replay.load_episode(p) for p in paths if not replay.excluded(p)]
    episodes = [ep for ep in episodes if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS]
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = [replay.load_episode(p) for p in sorted(Path("episodes_flaky").glob("*.json"))]
    out = {}

    print("1. refund ablation")
    late = {k: replay.late_streams(episodes, k) for k in (10, 40)}
    out["refund"] = {}
    for tag, r in REFUNDS.items():
        mk = lambda eps, r=r: partial(seq_accept, eps=eps, refund=r)
        rec = replay.summarize(run_rule(mk, episodes), episodes)
        row = {"recorded": {k: rec[k] for k in ("far", "frr", "blocks_per_accept")},
               "late": {k: {x: s[x] for x in ("far", "late_accept", "blocks_per_accept")}
                        for k, v in late.items() for s in [replay.summarize(run_rule(mk, v), v)]},
               "stress": {d: experiment(partial(seq_accept, refund=r), ps, reps=STRESS_REPS, seed=41)["far"] for d, ps in STRESS.items()},
               "worst": {n: experiment(partial(seq_accept, refund=r), [0.051] * n, reps=1000 if n < 50 else 400, seed=31)["far"]
                         for n in (15, 50)}}
        out["refund"][tag] = row
        print(f"   {tag}: {row}")

    print("2. 40-block shortcut")
    n_short = sum(len(shortcut(ep)) for ep in episodes)
    distinct = {(ep["name"].split("-")[0], f) for ep in episodes for f in shortcut(ep)}
    split_eps = [resplit(ep) for ep in episodes]
    res = replay.evaluate(split_eps)
    out["shortcut"] = {"candidate_slots": n_short, "distinct": len(distinct),
                       "broken_distinct": sum(1 for r in rates(episodes) if r["label"] == "broken"),
                       "table": {n: replay.summarize(rs, split_eps)["far"] for n, rs in res.items()},
                       "natural": {n: replay.summarize([r for r, ep in zip(rs, split_eps) if ep.get("flakiness", "natural") == "natural"],
                                                       [ep for ep in split_eps if ep.get("flakiness", "natural") == "natural"])["far"]
                                   for n, rs in res.items()}}
    print(f"   {out['shortcut']}")

    print("3. truth vs pool")
    out["rates"] = {"natural": rates(natural), "realflaky": rates(flaky)}
    for arm, rs in out["rates"].items():
        dis = sum((r["pool"] > r["eps"]) != (r["truth"] > r["eps"]) for r in rs)
        print(f"   {arm}: {len(rs)} candidates, {dis} on different sides of eps by truth and pool")

    print("4. units and bug-level intervals (natural)")
    per = {Path(e["file"]).stem: e for e in json.loads(Path("results/results.json").read_text())["per_episode"]}
    units = {"repositories": len({ep.get("repo") for ep in episodes}), "bugs": len({bug_of(ep) for ep in episodes}),
             "loops": len(natural), "episodes": len(episodes),
             "episodes_by_flakiness": {fl: sum(ep.get("flakiness", "natural") == fl for ep in episodes) for fl in replay.MAIN_FLAKINESS},
             "distinct_candidates": len({(bug_of(ep), f) for ep in natural for f in ep["fixes"]}),
             "candidate_positions": sum(len(ep["stream"]) for ep in natural),
             "blocks": sum(len(v["truth"]) + (0 if is_shortcut(ep, v) else len(v["pool"])) for ep in episodes for v in ep["fixes"].values())}
    nat_far = {}
    for name in all_rules():
        vals = [per[ep["name"]]["rules"][name]["far"] for ep in natural]
        nat_far[name] = {"far": sum(vals) / len(vals),
                         "ci_bug": replay.cluster_ci(vals, [bug_of(ep) for ep in natural]),
                         "ci_repo": replay.cluster_ci(vals, [ep.get("repo") for ep in natural])}
    out["units"], out["natural_far"] = units, nat_far
    print(f"   {units}")

    print("5. cost")
    cost = {}
    for name in all_rules():
        rs = [per[ep["name"]]["rules"][name] for ep in natural]
        reps = sum(r["reps"] for r in rs)
        with_c = [r["frr"] for r, ep in zip(rs, natural) if "correct" in replay.labels(ep)]
        cost[name] = {"blocks_per_episode": sum(r["blocks"] for r in rs) / reps, "seconds_per_episode": sum(r["seconds"] for r in rs) / reps,
                      "far": nat_far[name]["far"], "frr": sum(with_c) / len(with_c)}
    full = [ep for ep in (fullsuite_episodes(episodes) or []) if ep.get("flakiness", "natural") == "natural"]
    for name in replay_rules_fixed:
        rule = all_rules()[name]
        rs = [replay.replay(rule, ep, replay.REPS, seed=replay.episode_seed(ep, i)) for i, ep in enumerate(full)]
        s = replay.summarize(rs, full)
        cost[f"{name} (full suite)"] = {"runs_per_episode": sum(r["blocks"] for r in rs) / sum(r["reps"] for r in rs),
                                        "far": s["far"], "frr": s["frr"]}
    tasks = {bug_of(ep) for ep in natural}
    out["cost"] = {"rules": cost, "p0_blocks_per_task": 400, "loops_per_task": len(natural) / len(tasks)}
    print(f"   {json.dumps(cost)[:600]}")

    print("6. operating curve at p0 = 0 and 0.2")
    out["operating"] = []
    for p0 in (0.0, 0.2):
        eps = p0 + replay.MARGIN
        for d in OP_D:
            p = p0 + d
            row = {"p0": p0, "p": p, "eps": eps, "kl_floor": math.log(2 / 0.05) / kl(p, eps) if p < eps else None,
                   "kelly": single(partial(seq_accept, eps=eps), p), "mixture": single(partial(seq_accept, eps=eps, bet="mixture"), p)}
            out["operating"].append(row)
            print(f"   p0 {p0} p {p:.2f}: kelly {row['kelly']} mixture {row['mixture']}")

    print("7. persistence with bug-level intervals")
    pers = {}
    rules7 = {n: r for n, r in all_rules().items() if n in replay_rules_fixed or n == "SeqAccept"}
    for model in sorted({ep["model"] for ep in episodes}):
        mine = [ep for ep in episodes if ep["model"] == model]
        pers[model] = {}
        for k in replay.PREFIXES:
            cut = [{**ep, "stream": ep["stream"][:k]} for ep in mine]
            for name in rules7:
                vals = [replay.replay(all_rules(eps=ep["p0"] + replay.MARGIN)[name], ep, replay.REPS,
                                      seed=replay.episode_seed(ep, i))["far"] for i, ep in enumerate(cut)]
                pers[model].setdefault(name, []).append(
                    {"k": k, "far": sum(vals) / len(vals), "ci": replay.cluster_ci(vals, [bug_of(ep) for ep in cut])})
    out["persistence"] = pers

    Path("results/ease.json").write_text(json.dumps(out, indent=1, default=str))
    print("saved results/ease.json")
