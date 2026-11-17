"""Sensitivity and cost analyses (offline: no LLM, no Docker). Writes results/sensitivity.json.

  uv run python sensitivity.py

1. Cost table: per natural loop and per accept, without calibration, with p0 calibration shared per task or repeated
   per loop, and minutes of recorded block time per accept, for the main rules and the attempt-cap comparators.
2. Block-fraction sweep: each candidate's failing tests are inferred from its recorded blocks (a test in any passing
   block cannot be a deterministic failure); when every failing block contains one, the candidate's failure rate at
   another block size follows from the hypergeometric law. B1-B3, B4 and SeqAccept are replayed at half, the
   recorded and twice the recorded block size; candidates with failures no fixed test set explains (order bugs)
   keep their recorded rate.
3. Regime map: the smallest block that makes a regression of r always-failing tests in a suite of N fail a block
   with probability above eps, and whether SeqAccept on that block costs less per accept than full-suite B1/B2
   (run time proportional to tests, the break-even of ease_round4.py).
4. FAR scored against the pool rate (what replay draws from) beside FAR against truth labels.
5. Repository-clustered intervals for natural loops (bug-clustered ones are in ease2.json).
6. Matched-cost comparators on the real-flaky arm.
7. Margin against acceptance: one candidate as good as the original code (p = p0), eps = p0 + m; and the real-flaky
   arm at margins 0.05, 0.10, 0.15.
"""
import json
import math
import random
from functools import partial
from pathlib import Path
from statistics import mean

import replay
from extra_analyses import evaluate_rules, extra_rules, margin_labels, pool_labels
from rules import all_rules, bonferroni, fixed_spending, seq_accept
from sim import experiment, label, run_episodes

MAIN = ("B1 stop at first pass", "B2 rerun-3 all green", "B3 majority of 5", "B4 peeking Clopper-Pearson",
        "B5 naive confidence sequence", "B6 SeqAccept, no refund", "SeqAccept")
FIXED = MAIN[:3]
SWEEP_RULES = ("B1 stop at first pass", "B2 rerun-3 all green", "B3 majority of 5", "B4 peeking Clopper-Pearson", "SeqAccept")
FACTORS = (0.5, 1, 2)
SUITES = (640, 1000, 3000, 10000)
REGRESSIONS = (1, 2, 3, 5, 10)
P0S = (0.0, 0.1, 0.2, 0.3)
MARGINS = (0.03, 0.05, 0.10, 0.15)


def task_of(ep):
    return ep["name"].split("-")[0]


def subset(ids, b, seed):
    return set(random.Random(f"subset-{seed}").sample(ids, b))  # collect.subset: the block's tests


def failing_set(fix, ids, b):
    """The tests a candidate fails deterministically, or None when no fixed set explains its failing blocks."""
    outs = fix["truth"] + fix["pool"]
    if len(fix["seeds"]) != len(outs):  # shortcut candidates: the same blocks serve as truth and pool
        outs = fix["truth"][:len(fix["seeds"])]
    if not any(outs):
        return set()
    subs = [subset(ids, b, s) for s in fix["seeds"]]
    passed = set().union(*[s for s, y in zip(subs, outs) if not y])
    F = set(ids) - passed
    return F if all(s & F for s, y in zip(subs, outs) if y) else None


def hit(N, k, b):
    """Pr[a uniformly random b-subset of N tests contains one of k given tests]."""
    return 1.0 if b > N - k else 1 - math.comb(N - k, b) / math.comb(N, b)


def cost_rows(natural, per, rules_extra):
    by_task = {}
    for ep in natural:
        by_task.setdefault(task_of(ep), []).append(ep)
    p0b, secs = {}, {}
    for t, eps_ in by_task.items():
        p0b[t] = json.loads(Path(f"tasks/{t}.json").read_text())["p0_blocks"]
        secs[t] = mean(min(s, replay.CAP_SECONDS) for ep in eps_ for f in ep["fixes"].values() for s in f.get("seconds", []))
    results = {n: [per[ep["name"]]["rules"][n] for ep in natural] for n in MAIN}
    results.update(evaluate_rules(natural, rules_extra))
    out = {}
    for n, rs in results.items():
        reps, acc = sum(r["reps"] for r in rs), sum(r["accepts"] for r in rs)
        blocks, seconds = sum(r["blocks"] for r in rs), sum(r["seconds"] for r in rs)
        calib = n not in FIXED and not n.startswith("Rerun")
        ct = sum(p0b[task_of(ep)] / len(by_task[task_of(ep)]) * r["reps"] for ep, r in zip(natural, rs)) if calib else 0
        cl = sum(p0b[task_of(ep)] * r["reps"] for ep, r in zip(natural, rs)) if calib else 0
        cs = sum(p0b[task_of(ep)] / len(by_task[task_of(ep)]) * secs[task_of(ep)] * r["reps"]
                 for ep, r in zip(natural, rs)) if calib else 0
        out[n] = {"far": mean(r["far"] for r in rs), "frr": replay.summarize(rs, natural)["frr"],
                  "blocks_per_loop": blocks / reps, "blocks_per_accept": blocks / acc,
                  "bpa_task": (blocks + ct) / acc, "bpa_loop": (blocks + cl) / acc,
                  "minutes_per_accept": (seconds + cs) / acc / 60}
    return out


def sweep(natural, reps=replay.REPS):
    out = {"candidates": 0, "explained": 0, "model_gap": [], "rules": {}}
    tasks = {}
    for ep in natural:
        t = task_of(ep)
        if t not in tasks:
            tk = json.loads(Path(f"tasks/{t}.json").read_text())
            tasks[t] = (tk["test_ids"], tk["block_size"])
    fsets = {}
    for ep in natural:
        ids, b = tasks[task_of(ep)]
        for f, fx in ep["fixes"].items():
            key = (task_of(ep), f)
            if key not in fsets:
                fsets[key] = failing_set(fx, ids, b)
                out["candidates"] += 1
                out["explained"] += fsets[key] is not None
                if fsets[key]:
                    out["model_gap"].append(abs(hit(len(ids), len(fsets[key]), b) - mean(fx["truth"])))
    for factor in FACTORS:
        res = {n: [] for n in SWEEP_RULES}
        for i, ep in enumerate(natural):
            ids, b = tasks[task_of(ep)]
            bb = max(1, min(len(ids), round(b * factor)))
            ps = []
            for f in ep["stream"]:
                F = fsets[(task_of(ep), f)]
                ps.append(mean(ep["fixes"][f]["truth"]) if F is None else hit(len(ids), len(F), bb) if F else 0.0)
            labs = [label(p, ep["p0"]) for p in ps]
            for n in SWEEP_RULES:
                rng = random.Random(replay.episode_seed(ep, i))
                rule = all_rules(eps=ep["p0"] + replay.MARGIN)[n]
                res[n].append(run_episodes(rule, labs, lambda: (lambda j: int(rng.random() < ps[j])), reps))
        out["rules"][factor] = {n: {"far": mean(r["far"] for r in rs),
                                    "far_ci_repo": list(replay.cluster_ci([r["far"] for r in rs], [ep.get("repo") for ep in natural])),
                                    "blocks_per_loop": sum(r["blocks"] for r in rs) / sum(r["reps"] for r in rs)}
                                for n, rs in res.items()}
    out["model_gap"] = {"mean": mean(out["model_gap"]), "n": len(out["model_gap"])}
    return out


def regime(breakeven, eps=0.05):
    out = {}
    ratios = {k: breakeven[k]["ratio"] for k in ("B1", "B2")}
    for N in SUITES:
        for r in REGRESSIONS:
            b = next((b for b in range(1, N + 1) if hit(N, r, b) > eps + 1e-12), N)  # strictly above eps, float-safe
            # SeqAccept tests per accept: bpa_sa * b; full suite: runs_per_accept * N; ratio = bpa_sa / runs_per_accept
            out[f"{N}-{r}"] = {"block": b, "frac": b / N, **{f"cheaper_{k}": b * v < N for k, v in ratios.items()},
                               **{f"cost_ratio_{k}": b * v / N for k, v in ratios.items()}}
    return out


if __name__ == "__main__":
    paths = sorted(Path("episodes").glob("*.json"))
    episodes = [replay.load_episode(p) for p in paths if not replay.excluded(p)]
    episodes = [ep for ep in episodes if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS]
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = [replay.load_episode(p) for p in sorted(Path("episodes_flaky").glob("*.json"))]
    per = {Path(e["file"]).stem: e for e in json.loads(Path("results/results.json").read_text())["per_episode"]}
    e4 = json.loads(Path("results/ease4.json").read_text())
    k_match = json.loads(Path("results/extra.json").read_text())["k_match"]
    out = {}

    print("1. cost table")
    extra = lambda ep: {**extra_rules(ep["p0"] + replay.MARGIN, k_match),
                        "Bonferroni-100 spending": partial(fixed_spending, slices=bonferroni(100), eps=ep["p0"] + replay.MARGIN)}
    out["cost"] = cost_rows(natural, per, extra)
    print("   " + json.dumps({n: [round(v["blocks_per_accept"], 1), round(v["bpa_task"], 1), round(v["minutes_per_accept"], 2)]
                              for n, v in out["cost"].items()}))

    print("2. block-fraction sweep")
    out["block_fraction"] = sweep(natural)
    print(f"   {out['block_fraction']['explained']} of {out['block_fraction']['candidates']} explained; gap {out['block_fraction']['model_gap']}")
    print("   " + json.dumps({f: {n[:3]: round(100 * v["far"], 1) for n, v in d.items()} for f, d in out["block_fraction"]["rules"].items()}))

    print("3. regime map")
    out["regime"] = regime(e4["breakeven"])
    print("   " + json.dumps({k: (v["block"], v["cheaper_B1"], round(v["cost_ratio_B1"], 2)) for k, v in out["regime"].items()}))

    print("4. FAR against pool labels")
    out["pool_scored"] = {}
    for arm, eps_ in (("natural", natural), ("realflaky", flaky)):
        rs = evaluate_rules(eps_, lambda ep: all_rules(eps=ep["p0"] + replay.MARGIN), labels_of=pool_labels)
        out["pool_scored"][arm] = {n: mean(r["far"] for r in v) for n, v in rs.items()}
    print("   " + json.dumps({a: {n[:3]: round(100 * v, 2) for n, v in d.items()} for a, d in out["pool_scored"].items()}))

    print("5. repository-clustered intervals, natural loops")
    out["natural_repo_ci"] = {}
    for n in MAIN:
        fars = [per[ep["name"]]["rules"][n]["far"] for ep in natural]
        out["natural_repo_ci"][n] = {"far": mean(fars), "ci": list(replay.cluster_ci(fars, [ep.get("repo") for ep in natural])),
                                     "repos": len({ep.get("repo") for ep in natural})}
    print("   " + json.dumps({n[:3]: [round(100 * x, 1) for x in v["ci"]] for n, v in out["natural_repo_ci"].items()}))

    print("6. matched-cost comparators, real-flaky arm")
    rules_rf = lambda ep: {**{n: all_rules(eps=ep["p0"] + replay.MARGIN)[n] for n in ("B4 peeking Clopper-Pearson", "SeqAccept")},
                           **extra_rules(ep["p0"] + replay.MARGIN, k_match),
                           "Bonferroni-100 spending": partial(fixed_spending, slices=bonferroni(100), eps=ep["p0"] + replay.MARGIN)}
    res = evaluate_rules(flaky, rules_rf)
    hi = [i for i, ep in enumerate(flaky) if ep["p0"] >= 0.2]
    out["realflaky_matched"] = {}
    for n, rs in res.items():
        s = replay.summarize(rs, flaky)
        out["realflaky_matched"][n] = {"far": s["far"], "frr": s["frr"], "blocks_per_accept": s["blocks_per_accept"],
                                       "frr_high": replay.summarize([rs[i] for i in hi], [flaky[i] for i in hi])["frr"]}
    print("   " + json.dumps({n: [round(100 * v["far"], 2), round(100 * v["frr"], 1), round(v["blocks_per_accept"] or 0, 1)]
                              for n, v in out["realflaky_matched"].items()}))

    print("7. margin against acceptance")
    out["margin_curve"] = {}
    for p0 in P0S:
        for m in MARGINS:
            r = experiment(partial(seq_accept, eps=p0 + m), [p0], reps=2000, seed=11, p0=p0)
            out["margin_curve"][f"{p0}-{m}"] = {"accept": r["correct_accept"], "blocks": r["blocks"] / r["reps"]}
    out["realflaky_margin"] = {}
    for m in (0.05, 0.10, 0.15):
        rs = evaluate_rules(flaky, lambda ep: {"SeqAccept": all_rules(eps=ep["p0"] + m)["SeqAccept"]}, labels_of=margin_labels(m))["SeqAccept"]
        out["realflaky_margin"][m] = {"far": mean(r["far"] for r in rs),
                                      "frr": mean(r["frr"] for r, ep in zip(rs, flaky) if "correct" in margin_labels(m)(ep)),
                                      "frr_high": mean(rs[i]["frr"] for i in hi if "correct" in margin_labels(m)(flaky[i]))}
    print("   " + json.dumps({k: round(100 * v["accept"], 1) for k, v in out["margin_curve"].items()}))
    print("   " + json.dumps({m: [round(100 * v["far"], 2), round(100 * v["frr"], 1), round(100 * v["frr_high"], 1)]
                              for m, v in out["realflaky_margin"].items()}))

    Path("results/sensitivity.json").write_text(json.dumps(out, indent=1, default=str))
    print("saved results/sensitivity.json")
