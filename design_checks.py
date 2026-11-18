"""Design checks (EASE round 8, offline, no LLM or Docker). Writes results/design_checks.json.

1. Wallet and spending fraction: SeqAccept with the mixture accept wallet and with gamma = 3/4, beside the default, on
   natural, near-eps and real-flaky loops (both are covered by Theorem 4.2).
2. Refund against schedule: harmonic spending with and without the maximal refund (gamma_j = 1/(j + 2) of the pool,
   which without refunds gives harmonic's slices alpha / ((j+1)(j+2))), beside SeqAccept and B6, under the shared
   and the slice-scaled caps: natural cost, near-eps cost and FAR, late acceptance.
3. Fixed-sample Bonferroni sized for power: each candidate runs the fewest blocks m at which a candidate failing at
   p0 + 0.01 (the Correct boundary) is accepted with probability 0.9, at attempt caps 15, 100 and 1,000.
4. Corollary 4.3 within total level 0.05: alpha = 0.04 for SeqAccept and alpha_0 = 0.01 for the lower bound on p0,
   delta = 0.05 + sup(p0hat - L_0.99) at 400 blocks; on natural, near-eps and real-flaky loops, scored against
   p0hat + delta. The near-eps arm also gets the level-0.10 configuration of the paper's tables.
5. Near-eps exceedances by lower bound: per loop and per task, a one-sided 95% Clopper-Pearson lower bound on FAR
   above alpha, for B4, B5, B6 and SeqAccept.

  uv run python design_checks.py
"""
import json
import sys
from functools import partial
from pathlib import Path
from statistics import mean

import replay
import rules
from extra_analyses import evaluate_rules, margin_labels
from intervals import cp_lower
from rules import (all_rules, binom_cdf, bonferroni, critical, fixed_n, fixed_spending, harmonic, seq_accept, Result)
from near_eps_checks import calib

ALPHA = 0.05
eps_of = lambda ep: ep["p0"] + replay.MARGIN
task_of = lambda ep: ep["name"].split("-")[0]
load = lambda d, keep=lambda p: True: [replay.load_episode(p) for p in sorted(Path(d).glob("*.json")) if keep(p)]
HARM_REFUND = lambda j: 1 / (j + 2)  # fraction of the pool; without refunds this is harmonic spending exactly


def summary(rs, eps_list):
    s = replay.summarize(rs, eps_list)
    hi = [i for i, ep in enumerate(eps_list) if ep["p0"] >= 0.2]
    by = {}
    for r, ep in zip(rs, eps_list):
        by.setdefault(task_of(ep), []).append(r["far"])
    return {"far": s["far"], "frr": s["frr"], "blocks_per_accept": s["blocks_per_accept"],
            "blocks_per_loop": sum(r["blocks"] for r in rs) / sum(r["reps"] for r in rs),
            "frr_high": replay.summarize([rs[i] for i in hi], [eps_list[i] for i in hi])["frr"] if hi else None,
            "worst_task": max((mean(v), t) for t, v in by.items()),
            "correct_accept": mean(r["correct_accept"] for r in rs)}


def wallets(ep):
    e = eps_of(ep)
    return {"SeqAccept": partial(seq_accept, eps=e), "SeqAccept, mixture wallet": partial(seq_accept, eps=e, bet="mixture"),
            "SeqAccept, gamma 3/4": partial(seq_accept, eps=e, gamma=0.75)}


def schedules(ep):
    e = eps_of(ep)
    return {"SeqAccept": partial(seq_accept, eps=e), "B6": partial(seq_accept, eps=e, refund=False),
            "Harmonic": partial(fixed_spending, slices=harmonic(), eps=e),
            "Harmonic + refund": partial(seq_accept, eps=e, gamma=HARM_REFUND)}


def power_m(alpha_j, eps, p, power=0.9, limit=50_000):
    """Fewest blocks m at which the exact test at level alpha_j accepts a candidate failing at rate p w.p. >= power."""
    m = math.ceil(math.log(alpha_j) / math.log(1 - eps))
    while m <= limit:
        k = critical(m, eps, alpha_j)
        if k >= 0 and binom_cdf(k, m, p) >= power:
            return m
        m += max(1, m // 50)
    return limit


def fixed_power(n, block, cap, eps, p):
    """Fixed-sample Bonferroni over `cap` attempts, every candidate run for power_m blocks (stops early once rejected)."""
    alpha_j = ALPHA / cap
    m = power_m(alpha_j, eps, p)
    crit = critical(m, eps, alpha_j)
    blocks = 0
    for j in range(min(n, cap)):
        fails = 0
        for _ in range(m):
            fails += block(j)
            blocks += 1
            if fails > crit:
                break
        if fails <= crit:
            return Result(j, blocks)
    return Result(None, blocks)


def corollary_rule(nb, alpha, alpha0):
    gap = max(k / nb - cp_lower(k, nb, alpha0) for k in range(nb + 1))
    delta = replay.MARGIN + round(gap, 3)
    return delta, lambda ep: {"cor": partial(seq_accept, alpha=alpha,
                                             eps=cp_lower(round(ep["p0"] * nb), nb, alpha0) + delta)}


def save(out):
    Path("results/design_checks.json").write_text(json.dumps(out, indent=1, default=str))


import math  # noqa: E402  (used by power_m)

if __name__ == "__main__":
    episodes = load("episodes", lambda p: not replay.excluded(p))
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = load("episodes_flaky", lambda p: not replay.excluded(p))
    near = load("episodes_neareps")
    R = replay.NEAR_REPS
    arms = (("natural", natural, replay.REPS), ("neareps", near, R), ("realflaky", flaky, replay.REPS))
    out = json.loads(Path("results/design_checks.json").read_text()) if "--resume" in sys.argv and Path("results/design_checks.json").exists() else {}

    print("1. mixture wallet and gamma 3/4")
    if "wallets" not in out:
        out["wallets"] = {arm: {n: summary(rs, eps_) for n, rs in evaluate_rules(eps_, wallets, reps=reps).items()}
                          for arm, eps_, reps in arms}
        save(out)
    print("   " + json.dumps({a: {n: [round(100 * v["far"], 2), round(100 * (v["frr"] or 0), 2), round(v["blocks_per_loop"])]
                                  for n, v in d.items()} for a, d in out["wallets"].items()}))

    print("2. refund against schedule, shared and slice-scaled caps")
    if "schedules" not in out:
        out["schedules"] = {}
        for flag, name in ((False, "shared"), (True, "slice")):
            rules.SLICE_CAPS = flag
            d = {arm: {n: summary(rs, eps_) for n, rs in evaluate_rules(eps_, schedules, reps=reps).items()}
                 for arm, eps_, reps in arms[:2]}
            for k in (10, 40):
                late = replay.late_streams(episodes, k)
                d[f"late{k}"] = {n: mean(r["correct_accept"] for r in rs) for n, rs in evaluate_rules(late, schedules).items()}
            out["schedules"][name] = d
        rules.SLICE_CAPS = False
        save(out)
    print("   " + json.dumps({c: {a: {n: (round(v["blocks_per_loop"]) if isinstance(v, dict) else round(100 * v, 1))
                                       for n, v in d.items()} for a, d in dd.items()} for c, dd in out["schedules"].items()}))

    print("3. fixed-sample Bonferroni sized for power at p0 + 0.01")
    if "fixed_power" not in out:
        out["fixed_power"] = {}
        for cap in (15, 100, 1000):
            rf = lambda ep, cap=cap: {"fixed": partial(fixed_power, cap=cap, eps=eps_of(ep), p=ep["p0"] + 0.01),
                                      "fixed n0": partial(fixed_n, slices=bonferroni(cap), eps=eps_of(ep), scale=1)}
            d = {n: summary(rs, natural) for n, rs in evaluate_rules(natural, rf).items()}
            for k in (10, 40):
                late = replay.late_streams(episodes, k)
                for n, rs in evaluate_rules(late, rf).items():
                    d[n][f"late{k}"] = mean(r["correct_accept"] for r in rs)
            d["m"] = power_m(ALPHA / cap, 0.05, 0.01)
            out["fixed_power"][str(cap)] = d
        save(out)
    print("   " + json.dumps({c: {n: ([round(v["blocks_per_accept"] or 0), round(100 * v["late10"], 1), round(100 * v["late40"], 1)]
                                     if isinstance(v, dict) else v) for n, v in d.items()} for c, d in out["fixed_power"].items()}))

    print("4. Corollary 4.3 at total level 0.05 (alpha 0.04, alpha_0 0.01) and at 0.10 near eps")
    if "split" not in out:
        nb = json.loads(Path(f"tasks/{task_of(natural[0])}.json").read_text())["p0_blocks"]
        out["split"] = {}
        for tag, a, a0 in (("total05", 0.04, 0.01), ("total10", 0.05, 0.05)):
            delta, rf = corollary_rule(nb, a, a0)
            lab = margin_labels(delta)
            d = {"delta": delta}
            for arm, eps_, reps in arms:
                rs = evaluate_rules(eps_, rf, reps=reps, labels_of=lab)["cor"]
                s = replay.summarize(rs, eps_)
                hi = [i for i, ep in enumerate(eps_) if ep["p0"] >= 0.2]
                acc = sum(r["accepts"] for r in rs)
                cb = calib(eps_)
                d[arm] = {"far": s["far"], "frr": s["frr"], "blocks_per_accept": s["blocks_per_accept"],
                          "bpa_calib": (sum(r["blocks"] for r in rs) + sum(c[0] * r["reps"] for c, r in zip(cb, rs))) / acc if acc else None,
                          "blocks_per_loop": sum(r["blocks"] for r in rs) / sum(r["reps"] for r in rs),
                          "worst_loop": max(r["far"] for r in rs),
                          "frr_high": replay.summarize([rs[i] for i in hi], [eps_[i] for i in hi])["frr"] if hi else None}
            out["split"][tag] = d
        save(out)
    print("   " + json.dumps({t: {a: ([round(100 * v["far"], 2), round(100 * (v["frr"] or 0), 2), round(v["blocks_per_accept"] or 0)]
                                     if isinstance(v, dict) else v) for a, v in d.items()} for t, d in out["split"].items()}))

    print("5. near-eps exceedances by lower bound")
    if "near_lower" not in out:
        four = lambda ep: {n: all_rules(eps=eps_of(ep))[n] for n in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence",
                                                                     "B6 SeqAccept, no refund", "SeqAccept")}
        rr = evaluate_rules(near, four, reps=R)
        d = {}
        for n, rs in rr.items():
            ks = [round(r["far"] * r["reps"]) for r in rs]
            by = {}
            for k, ep in zip(ks, near):
                by.setdefault(task_of(ep), []).append(k)
            d[n] = {"loops_lower_above": sum(cp_lower(k, R) > ALPHA for k in ks),
                    "tasks": {t: {"far": sum(v) / (R * len(v)), "lower": cp_lower(sum(v), R * len(v))} for t, v in by.items()}}
            d[n]["tasks_lower_above"] = sum(v["lower"] > ALPHA for v in d[n]["tasks"].values())
        out["near_lower"] = d
        save(out)
    print("   " + json.dumps({n[:3]: [v["loops_lower_above"], v["tasks_lower_above"],
                                     {t.split("@")[0]: round(100 * x["far"], 2) for t, x in v["tasks"].items()}]
                                for n, v in out["near_lower"].items()}))
    print("6. latency: SeqAccept's blocks per accept times the mean block seconds, blocks run 8 at a time")
    secs = lambda eps_: mean(min(x, replay.CAP_SECONDS) for ep in eps_ for f in ep["fixes"].values() for x in f.get("seconds", []))
    out["latency"] = {arm: {"block_s": secs(eps_), "blocks_per_accept": out["wallets"][arm]["SeqAccept"]["blocks_per_accept"],
                            "minutes_at_8": out["wallets"][arm]["SeqAccept"]["blocks_per_accept"] * secs(eps_) / 8 / 60}
                      for arm, eps_, _ in arms}
    print("   " + json.dumps({a: round(v["minutes_at_8"], 1) for a, v in out["latency"].items()}))
    save(out)
    print("saved results/design_checks.json")
