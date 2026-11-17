"""EASE review round 3 follow-ups (offline: no LLM, no Docker). Writes results/caps_adversary.json.

  uv run python caps_adversary.py

1. Bonferroni with an attempt cap n_max = 15, 40, 100 (spending and fixed-n) against SeqAccept: natural loops and
   late streams of k = 10, 20, 40 broken candidates before a correct one.
2. White-box adaptive adversary: it sees every outcome, tracks the rule's pool and picks the next candidate's failure
   rate; run against no, maximal, capped and uncapped refunds.
3. Dependence to FAR: the observed lag-1 correlation of near-eps real-flaky candidates mapped through the
   correlated-block simulation.
4. Full suite against sampled blocks: recorded run times, and the run-time ratio (and suite size, if run time is
   proportional to tests) at which SeqAccept on 32-test blocks costs as much per accept as full-suite B1 or B2.
5. Calibration amortised per task (shared by its loops) or per loop.
6. Real-flaky arm per task, with and without labels whose posterior chance of being wrong exceeds 5%.
7. Failure rates of the planted bugs before the bug filter, and of the recorded Broken candidates.
8. Worst case at p = 0.10 for n = 1, 5, 15, 50; FAR on natural loops that contain a Broken candidate.
"""
import json
import random
from functools import partial
from pathlib import Path
from statistics import mean

import replay
from comparators import bug_of
from extra_analyses import CORR_P, correlated, evaluate_rules, lag1_corr, summary as extra_summary
from label_risk import beta_cdf
from rules import REJECT_AT, _refund, all_rules, bets, bonferroni, derived_cap, fixed_n, fixed_spending, seq_accept
from sim import experiment, label

NMAX = (15, 40, 100)
LATE_K = (10, 20, 40)
REFUNDS = {"none": (False, False), "max": ("max", True), "capped": (True, True), "uncapped": ("uncapped", True),
           "online": ("uncapped", False)}  # (refund, refund at the cap); "online" is the rule the live runs used
EPS = 0.05
ALPHA, GAMMA, RHO_REJECT, CAP = 0.05, 0.5, REJECT_AT, derived_cap(EPS)  # the rule's defaults


def cap_rules(eps):
    out = {"SeqAccept": partial(seq_accept, eps=eps)}
    for n in NMAX:
        out[f"Bonferroni-{n} spending"] = partial(fixed_spending, slices=bonferroni(n), eps=eps)
        out[f"Fixed-n Bonferroni-{n}"] = partial(fixed_n, slices=bonferroni(n), eps=eps, scale=1)
    return out


def adversary(refund, policy, n, reps, seed, near=0.051, far_p=0.9, at_cap=True, trace=None):
    """FAR of SeqAccept with `refund` against a white-box adversary. Every candidate is broken. The adversary hands out
    the block outcomes, so it can replay the rule's arithmetic and knows the pool W before choosing the next candidate:
    policy "near" always proposes p = near; policy "switch" proposes p = near while W >= alpha / 2 and otherwise
    alternates a quickly rejected candidate (p = far_p) with a near one."""
    rng, hits = random.Random(seed), 0
    for _ in range(reps):
        st = {"W": ALPHA, "j": -1, "outs": [], "p": None, "k": 0}

        def close():  # replay the finished candidate's wallets to update the adversary's copy of W
            a_j = GAMMA * st["W"]
            E = R = peak = 1.0
            fails, ended = 0, False
            for t, y in enumerate(st["outs"]):
                lam, mu = bets(fails, t, EPS)
                fails += y
                E *= 1 + lam * (EPS - y)
                R *= 1 + mu * (y - EPS)
                peak = max(peak, E)
                if R >= RHO_REJECT:
                    ended = True
                    break
            st["W"] = st["W"] - a_j + (_refund(refund, a_j, E, peak) if (ended or (at_cap and len(st["outs"]) >= CAP)) else 0.0)

        def block(j):
            if j != st["j"]:
                if st["j"] >= 0:
                    close()
                st["j"], st["outs"] = j, []
                if policy == "near" or st["W"] >= ALPHA / 2:
                    st["p"] = near
                else:
                    st["k"] += 1
                    st["p"] = far_p if st["k"] % 2 else near
            y = int(rng.random() < st["p"])
            st["outs"].append(y)
            return y

        if trace:
            trace(None, None, None)  # a new run starts
        res = seq_accept(n, block, eps=EPS, refund=refund, setaside_refund=at_cap, trace=trace)
        hits += res.accepted is not None
    return hits / reps


def ambiguous_labels(ep):
    """Labels with a Broken or Correct label whose Beta posterior gives more than 5% to the other side set Marginal."""
    eps, out = ep["p0"] + replay.MARGIN, []
    for f, lab in zip(ep["stream"], replay.labels(ep)):
        t = ep["fixes"][f]["truth"]
        below = beta_cdf(eps, 1 + sum(t), 1 + len(t) - sum(t))
        risky = (lab == "broken" and below > 0.05) or (lab == "correct" and 1 - below > 0.05)
        out.append("marginal" if risky else lab)
    return out


def ols(xs, ys):
    mx, my = mean(xs), mean(ys)
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - c * mx, c


if __name__ == "__main__":
    paths = sorted(Path("episodes").glob("*.json"))
    episodes = [replay.load_episode(p) for p in paths if not replay.excluded(p)]
    episodes = [ep for ep in episodes if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS]
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = [replay.load_episode(p) for p in sorted(Path("episodes_flaky").glob("*.json"))]
    per = {Path(e["file"]).stem: e for e in json.loads(Path("results/results.json").read_text())["per_episode"]}
    e3 = json.loads(Path("results/intervals.json").read_text())
    out = {}

    print("1. attempt caps")
    res = evaluate_rules(natural, lambda ep: cap_rules(ep["p0"] + replay.MARGIN))
    out["caps"] = {"natural": {n: {k: v[k] for k in ("far", "frr", "blocks_per_accept")} for n, v in extra_summary(res, natural).items()},
                   "late": {}}
    for k in LATE_K:
        late = replay.late_streams(episodes, k)
        res = evaluate_rules(late, lambda ep: cap_rules(ep["p0"] + replay.MARGIN))
        out["caps"]["late"][k] = {n: {"accept": mean(r["correct_accept"] for r in rs),
                                      "blocks_per_accept": (sum(r["blocks"] for r in rs) / sum(r["accepts"] for r in rs)
                                                            if sum(r["accepts"] for r in rs) else None)}
                                  for n, rs in res.items()}
    print(f"   {json.dumps({n: round(v['blocks_per_accept'] or 0, 1) for n, v in out['caps']['natural'].items()})}")
    print(f"   {json.dumps({k: {n: round(100 * v['accept'], 1) for n, v in d.items()} for k, d in out['caps']['late'].items()})}")

    print("2. white-box adaptive adversary")
    out["adversary"] = {}
    for n, reps in ((15, 1000), (50, 300)):
        for policy in ("near", "switch"):
            for name, r in REFUNDS.items():
                out["adversary"][f"{policy}-{n}-{name}"] = {"far": adversary(r[0], policy, n, reps, seed=97, at_cap=r[1]), "reps": reps}
        print(f"   n = {n}: {json.dumps({k: round(100 * v['far'], 2) for k, v in out['adversary'].items() if f'-{n}-' in k})}")

    print("3. dependence to FAR")
    lags = []
    for ep in flaky:
        for f in ep["fixes"].values():
            if f["pool"] and len(f["truth"]) + len(f["pool"]) == len(f["seeds"]) and ep["p0"] < mean(f["pool"]) <= 0.5:
                seq = f["truth"] + f["pool"]
                if 0 < sum(seq) < len(seq):
                    lags.append(lag1_corr(seq))
    lags.sort()
    q90 = lags[int(0.9 * len(lags))]
    upper_mean = e3["near_eps_dependence"]["realflaky"]["lag1_ci_task"][1]
    out["dependence"] = {"candidates": len(lags), "q90": q90, "upper_mean": upper_mean, "far": {}}
    for tag, phi in (("zero", 0.0), ("upper_mean", max(0.0, upper_mean)), ("q90", max(0.0, q90))):
        out["dependence"]["far"][tag] = {n: correlated(seq_accept, CORR_P, n, phi, 2000, 53) for n in (1, 15)}
    print(f"   {out['dependence']}")

    print("4. full suite against sampled blocks")
    block_s, full_s = [], []
    for ep in natural:
        block_s += [min(x, replay.CAP_SECONDS) for f in ep["fixes"].values() for x in f.get("seconds", [])]
        variant = ep["name"].split("-")[0]
        for f in ep["fixes"]:
            fp = Path(f"fullsuite/{variant}__{f}.json")
            if fp.exists():
                full_s += json.loads(fp.read_text())["seconds"]
    cc = e3["cost_calibrated"]
    bpa_sa = cc["SeqAccept"]["blocks_per_accept"]
    out["breakeven"] = {"block_s": mean(block_s), "full_s": mean(full_s), "sa_blocks_per_accept": bpa_sa}
    for name, key in (("B1 stop at first pass", "B1"), ("B2 rerun-3 all green", "B2")):
        rpa = cc[f"{name} (full suite)"]["runs_per_accept"]
        # SeqAccept is cheaper per accept once a full-suite run costs more than bpa_sa / rpa blocks; with run time
        # proportional to tests and no start-up cost, that is a suite of 32 * bpa_sa / rpa tests (a lower bound:
        # start-up cost, which dominates our runs, favours the full suite)
        out["breakeven"][key] = {"runs_per_accept": rpa, "ratio": bpa_sa / rpa, "suite_tests": 32 * bpa_sa / rpa}
    print(f"   {out['breakeven']}")

    print("5. calibration amortised per task or per loop")
    loops_of = {}
    for ep in natural:
        loops_of.setdefault(ep["name"].split("-")[0], []).append(ep)
    out["amortised"] = {}
    for name in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "SeqAccept"):
        rs = [per[ep["name"]]["rules"][name] for ep in natural]
        acc = sum(r["accepts"] for r in rs)
        blocks = sum(r["blocks"] for r in rs)
        p0b = {ep["name"]: json.loads(Path(f"tasks/{ep['name'].split('-')[0]}.json").read_text())["p0_blocks"] for ep in natural}
        task = sum(p0b[ep["name"]] / len(loops_of[ep["name"].split("-")[0]]) * r["reps"] for ep, r in zip(natural, rs))
        loop = sum(p0b[ep["name"]] * r["reps"] for ep, r in zip(natural, rs))
        out["amortised"][name] = {"none": blocks / acc, "per_task": (blocks + task) / acc, "per_loop": (blocks + loop) / acc}
    print(f"   {out['amortised']}")

    print("6. real-flaky arm per task, with and without ambiguous labels")
    tasks = {}
    for ep in flaky:
        tasks.setdefault(ep["name"].split("-")[0], []).append(ep)
    rules_of = lambda ep: {n: all_rules(eps=ep["p0"] + replay.MARGIN)[n] for n in ("B4 peeking Clopper-Pearson", "SeqAccept")}
    plain = evaluate_rules(flaky, rules_of)
    amb = evaluate_rules(flaky, rules_of, labels_of=ambiguous_labels)
    out["realflaky_tasks"] = {}
    for t, eps_ in sorted(tasks.items()):
        idx = [i for i, ep in enumerate(flaky) if ep["name"].split("-")[0] == t]
        eps = eps_[0]["p0"] + replay.MARGIN
        near = sum(1 for ep in eps_ for f, lab in zip(ep["stream"], replay.labels(ep))
                   if lab == "broken" and mean(ep["fixes"][f]["truth"]) <= eps + 0.03)
        sa, b4 = "SeqAccept", "B4 peeking Clopper-Pearson"
        out["realflaky_tasks"][t] = {"p0": eps_[0]["p0"], "loops": len(eps_), "near_broken": near,
                                     "far_sa": mean(plain[sa][i]["far"] for i in idx), "far_b4": mean(plain[b4][i]["far"] for i in idx),
                                     "frr_sa": mean(plain[sa][i]["frr"] for i in idx),
                                     "far_sa_unamb": mean(amb[sa][i]["far"] for i in idx),
                                     "far_b4_unamb": mean(amb[b4][i]["far"] for i in idx)}
    out["realflaky_unamb"] = {n: {"far": mean(r["far"] for r in amb[n]), "far_plain": mean(r["far"] for r in plain[n])}
                              for n in amb}
    print(f"   {out['realflaky_unamb']}")

    print("7. failure rates before the bug filter")
    bugs = {}
    for fp in sorted(p for p in Path("runs/blocks").glob("bug-*.jsonl") if "@" not in p.name):
        rows = [json.loads(x) for x in fp.read_text().splitlines() if x.strip()]
        name = fp.stem[4:]
        task = Path(f"tasks/{name}.json")
        p0 = json.loads(task.read_text())["p0"] if task.exists() else None
        bugs[name] = {"rate": mean(r["failed"] for r in rows), "blocks": len(rows), "p0": p0}
    seen, rates = set(), []
    for ep in natural:
        eps = ep["p0"] + replay.MARGIN
        for f, lab in zip(ep["stream"], replay.labels(ep)):
            key = (ep["name"].split("-")[0], f)
            if lab == "broken" and key not in seen:
                seen.add(key)
                rates.append(mean(ep["fixes"][f]["truth"]) - eps)
    out["prefilter_rates"] = {"bugs": bugs, "broken_minus_eps": {
        "n": len(rates), "le_005": sum(r <= 0.05 for r in rates), "le_025": sum(r <= 0.25 for r in rates),
        "median": sorted(rates)[len(rates) // 2]}}
    print(f"   {len(bugs)} bugs; broken candidates {out['prefilter_rates']['broken_minus_eps']}")

    print("8. worst case at p = 0.10; conditional FAR")
    rules = {n: r for n, r in all_rules(eps=0.05).items() if n in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence",
                                                                     "B6 SeqAccept, no refund", "SeqAccept")}
    out["worst010"] = {n: {name: experiment(r, [0.10] * n, reps=1000 if n < 50 else 400, seed=31)["far"] for name, r in rules.items()}
                       for n in (1, 5, 15, 50)}
    with_broken = [ep for ep in natural if "broken" in replay.labels(ep)]
    out["conditional_far"] = {"loops": len(with_broken),
                              **{n: mean(per[ep["name"]]["rules"][n]["far"] for ep in with_broken) for n in all_rules()}}
    print(f"   {out['worst010']}")

    Path("results/caps_adversary.json").write_text(json.dumps(out, indent=1, default=str))
    print("saved results/caps_adversary.json")
