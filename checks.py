"""Validity, interval and selection checks (offline: no LLM, no Docker). Writes results/checks.json.

  uv run python checks.py

1. The proof's invariant: M_t = W_t + alpha_j E_j for broken candidates (Theorem 4.2) logged in the adaptive adversary
   and in constructed near-eps streams; its mean at the stop must stay at or below alpha (optional stopping), a far
   lower-variance check than FAR. The uncapped refund, outside the proof, is logged for contrast.
2. Power of the runs test against lag-1 dependence phi = 0.2 at the observed length and failure rate of each
   near-eps real-flaky candidate (and its size at phi = 0).
3. Paired, cluster-bootstrapped differences: real-flaky FRR of SeqAccept minus each valid rule; natural cost ratio
   (blocks per accept with calibration) of SeqAccept to Bonferroni-100 spending, fixed-n Bonferroni-15 and B4.
4. Fixed-sample Bonferroni over caps up to 10,000: cost with calibration, and late acceptance at a cap of 1,000.
5. Replay without replacement (each candidate reads its 200-block pool in a random order, block cap 200).
6. Real-flaky FRR with labels whose posterior chance of being wrong exceeds 5% set to Marginal.
7. Abstention with the block cap removed (B = 10,000) for one candidate as good as the original code, and on the
   real-flaky arm.
8. The two tasks left out because grok-4.7 could not record them: rules on natural loops with the other models' loops.
9. Drift over the recording window: B1 FAR on each model's run 1 and run 2 of the same natural tasks.
10. Corollary 4.3 as run: eps = L + delta from a one-sided 95% Clopper-Pearson bound L on p0, delta = 0.05 + the
    largest p0hat - L at 400 blocks, scored against p0 + delta (natural, injected, real-flaky).
11. Real-flaky FAR over the loops that contain a Broken candidate, and the worst task with a bound.
12. Refund break-even: blocks to accept a candidate that never fails at stream position j under SeqAccept (earlier
    wallets unmoved), harmonic spending and no refund; the first j at which the no-refund slice falls below harmonic.
"""
import json
import random
from functools import partial
from pathlib import Path
from statistics import mean

import replay
from caps_adversary import adversary, ambiguous_labels
from comparators import bug_of
from extra_analyses import evaluate_rules, extra_rules, runs_test
from rules import all_rules, bonferroni, fixed_n, fixed_spending, harmonic, seq_accept
from sensitivity import cost_rows, task_of
from sim import experiment, run_episodes

ALPHA = 0.05
VALID = ("B6 SeqAccept, no refund", "Harmonic spending", "Bonferroni-15 spending", "Bonferroni-100 spending",
         "Fixed-n Bonferroni-15 (1n0)")
CAPS = (15, 40, 100, 300, 1000, 3000, 10000)


class Tracker:
    """Collects M = pool + alpha_j E per run from seq_accept's trace hook (every candidate here is broken)."""

    def __init__(self):
        self.stops, self.maxes, self.cur = [], [], None

    def __call__(self, pool, alpha_j, E):
        if pool is None:  # adversary: a new run starts
            self.close()
            return
        m = pool + alpha_j * E
        if self.cur is None:
            self.cur = [m, m]
        self.cur = [m, max(self.cur[1], m)]

    def close(self):
        if self.cur is not None:
            self.stops.append(self.cur[0])
            self.maxes.append(self.cur[1])
        self.cur = None

    def summary(self):
        self.close()
        n = len(self.stops)
        mu = mean(self.stops)
        se = (sum((x - mu) ** 2 for x in self.stops) / (n - 1) / n) ** 0.5
        return {"runs": n, "mean_stop": mu, "se": se, "upper": mu + 1.96 * se, "far": mean(x >= 1 for x in self.maxes)}


def constructed(p, n, reps, seed, refund="max"):
    rng, tr = random.Random(seed), Tracker()
    for _ in range(reps):
        seq_accept(n, lambda j: int(rng.random() < p), eps=0.05, refund=refund, trace=tr)
        tr.close()
    return tr.summary()


def markov(rng, n, p, phi):
    """Stationary 0/1 Markov chain with rate p and lag-1 correlation phi."""
    y, out = rng.random() < p, []
    for _ in range(n):
        out.append(int(y))
        y = rng.random() < (p + phi * (1 - p) if y else p * (1 - phi))
    return out


def ratio_ci(groups, n_boot=2000, seed=0):
    """95% bootstrap interval of sum(a)/sum(b) over sum(c)/sum(d), resampling clusters; groups: {cluster: (a, b, c, d)}."""
    keys, rng, stats = list(groups), random.Random(seed), []
    for _ in range(n_boot):
        pick = [groups[rng.choice(keys)] for _ in keys]
        a, b, c, d = (sum(x[i] for x in pick) for i in range(4))
        if b and d and c:
            stats.append((a / b) / (c / d))
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]


def without_replacement(rule, ep, reps, seed):
    fixes, rng = ep["fixes"], random.Random(seed)
    pools = [fixes[f]["pool"] for f in ep["stream"]]

    def make_block():
        order = [rng.sample(p, len(p)) for p in pools]
        pos = [0] * len(pools)

        def draw(j):
            y = order[j][pos[j]]
            pos[j] += 1
            return y
        return draw
    return run_episodes(rule, replay.labels(ep), make_block, reps)


if __name__ == "__main__":
    paths = sorted(Path("episodes").glob("*.json"))
    episodes = [replay.load_episode(p) for p in paths if not replay.excluded(p)]
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = [replay.load_episode(p) for p in sorted(Path("episodes_flaky").glob("*.json"))]
    per = {Path(e["file"]).stem: e for e in json.loads(Path("results/results.json").read_text())["per_episode"]}
    k_match = json.loads(Path("results/extra.json").read_text())["k_match"]
    out = {}

    print("1. invariant M")
    out["invariant"] = {}
    for policy in ("near", "switch"):
        for refund, at_cap in (("max", True), ("uncapped", True)):
            tr = Tracker()
            adversary(refund, policy, 15, 1000, seed=97, at_cap=at_cap, trace=tr)
            out["invariant"][f"adversary-{policy}-15-{refund}"] = tr.summary()
    for n, reps in ((15, 1000), (50, 400)):
        out["invariant"][f"constructed-0.051-{n}-max"] = constructed(0.051, n, reps, seed=5)
    print("   " + json.dumps({k: [round(v["mean_stop"], 4), round(v["upper"], 4), round(100 * v["far"], 1)] for k, v in out["invariant"].items()}))

    print("2. runs-test power")
    seqs = []
    for ep in flaky:
        for f in ep["fixes"].values():
            if f["pool"] and len(f["truth"]) + len(f["pool"]) == len(f["seeds"]) and ep["p0"] < mean(f["pool"]) <= 0.5:
                s = f["truth"] + f["pool"]
                if 0 < sum(s) < len(s):
                    seqs.append(s)
    rng = random.Random(3)
    out["runs_power"] = {"candidates": len(seqs)}
    for phi in (0.0, 0.2, 0.5):
        rej = [runs_test(markov(rng, len(s), mean(s), phi)) for s in seqs for _ in range(20)]
        out["runs_power"][str(phi)] = mean(bool(r) for r in rej if r is not None)
    print(f"   {out['runs_power']}")

    print("3. paired intervals")
    eps_of = lambda ep: ep["p0"] + replay.MARGIN
    rules_rf = lambda ep: {"SeqAccept": all_rules(eps=eps_of(ep))["SeqAccept"],
                           "B6 SeqAccept, no refund": all_rules(eps=eps_of(ep))["B6 SeqAccept, no refund"],
                           **{k: v for k, v in extra_rules(eps_of(ep), k_match).items() if k in VALID},
                           "Bonferroni-100 spending": partial(fixed_spending, slices=bonferroni(100), eps=eps_of(ep))}
    res = evaluate_rules(flaky, rules_rf)
    has_c = [i for i, ep in enumerate(flaky) if "correct" in replay.labels(ep)]
    out["frr_diff"] = {}
    for n in VALID:
        d = [res["SeqAccept"][i]["frr"] - res[n][i]["frr"] for i in has_c]
        out["frr_diff"][n] = {"diff": mean(d), "ci_task": list(replay.cluster_ci(d, [task_of(flaky[i]) for i in has_c])),
                              "ci_repo": list(replay.cluster_ci(d, [flaky[i].get("repo") for i in has_c]))}
    print("   " + json.dumps({n: [round(100 * v["diff"], 1)] + [round(100 * x, 1) for x in v["ci_task"]] for n, v in out["frr_diff"].items()}))
    by_task = {}
    for ep in natural:
        by_task.setdefault(task_of(ep), []).append(ep)
    p0b = {t: json.loads(Path(f"tasks/{t}.json").read_text())["p0_blocks"] for t in by_task}
    comp = lambda ep: {"Bonferroni-100 spending": partial(fixed_spending, slices=bonferroni(100), eps=eps_of(ep)),
                       "Fixed-n Bonferroni-15 (1n0)": partial(fixed_n, slices=bonferroni(15), eps=eps_of(ep), scale=1)}
    cres = evaluate_rules(natural, comp)
    cres["SeqAccept"] = [per[ep["name"]]["rules"]["SeqAccept"] for ep in natural]
    cres["B4 peeking Clopper-Pearson"] = [per[ep["name"]]["rules"]["B4 peeking Clopper-Pearson"] for ep in natural]

    def task_sums(name):
        g = {}
        for ep, r in zip(natural, cres[name]):
            t = task_of(ep)
            cal = p0b[t] / len(by_task[t]) * r["reps"]
            b, a = g.get(t, (0.0, 0))
            g[t] = (b + r["blocks"] + cal, a + r["accepts"])
        return g
    sa = task_sums("SeqAccept")
    out["cost_ratio"] = {}
    for n in ("Bonferroni-100 spending", "Fixed-n Bonferroni-15 (1n0)", "B4 peeking Clopper-Pearson"):
        o = task_sums(n)
        g = {t: (*sa[t], *o[t]) for t in sa}
        point = (sum(v[0] for v in g.values()) / sum(v[1] for v in g.values())) / (sum(v[2] for v in g.values()) / sum(v[3] for v in g.values()))
        out["cost_ratio"][n] = {"ratio": point, "ci_task": list(ratio_ci(g))}
    print("   " + json.dumps({n: [round(v["ratio"], 2)] + [round(x, 2) for x in v["ci_task"]] for n, v in out["cost_ratio"].items()}))

    print("4. Bonferroni caps up to 10,000")
    caps = lambda ep: {**{f"Fixed-n Bonferroni-{n}": partial(fixed_n, slices=bonferroni(n), eps=eps_of(ep), scale=1) for n in CAPS},
                       "Bonferroni-1000 spending": partial(fixed_spending, slices=bonferroni(1000), eps=eps_of(ep))}
    c = cost_rows(natural, per, caps)
    out["caps"] = {n: {k: c[n][k] for k in ("bpa_task", "blocks_per_accept", "frr")} for n in c if "Bonferroni" in n}
    out["caps"]["SeqAccept"] = {k: c["SeqAccept"][k] for k in ("bpa_task", "blocks_per_accept", "frr")}
    out["caps_late"] = {}
    for k in (10, 40):
        late = replay.late_streams(episodes, k)
        lr = evaluate_rules(late, lambda ep: {"Fixed-n Bonferroni-1000": partial(fixed_n, slices=bonferroni(1000), eps=eps_of(ep), scale=1),
                                              "Bonferroni-1000 spending": partial(fixed_spending, slices=bonferroni(1000), eps=eps_of(ep))})
        out["caps_late"][k] = {n: mean(r["correct_accept"] for r in rs) for n, rs in lr.items()}
    print("   " + json.dumps({n: round(v["bpa_task"], 1) for n, v in out["caps"].items()}) + " " + json.dumps(out["caps_late"]))

    print("5. without replacement")
    out["no_replacement"] = {}
    for arm, eps_ in (("natural", natural), ("realflaky", flaky)):
        out["no_replacement"][arm] = {}
        for n in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "B6 SeqAccept, no refund", "SeqAccept"):
            rs = [without_replacement(partial(all_rules(eps=eps_of(ep))[n].func, *all_rules(eps=eps_of(ep))[n].args,
                                              **{**all_rules(eps=eps_of(ep))[n].keywords, "max_blocks": 200}),
                                      ep, replay.REPS, replay.episode_seed(ep, i)) for i, ep in enumerate(eps_)]
            s = replay.summarize(rs, eps_)
            out["no_replacement"][arm][n] = {"far": s["far"], "frr": s["frr"], "blocks_per_accept": s["blocks_per_accept"]}
    print("   " + json.dumps({a: {n[:3]: [round(100 * v["far"], 2), round(100 * v["frr"], 1)] for n, v in d.items()} for a, d in out["no_replacement"].items()}))

    print("6. real-flaky FRR with ambiguous labels relabelled")
    r6 = evaluate_rules(flaky, lambda ep: {n: all_rules(eps=eps_of(ep))[n] for n in ("B4 peeking Clopper-Pearson", "SeqAccept")},
                        labels_of=ambiguous_labels)
    keep = [i for i, ep in enumerate(flaky) if "correct" in ambiguous_labels(ep)]
    out["frr_unamb"] = {n: {"frr": mean(rs[i]["frr"] for i in keep), "loops": len(keep)} for n, rs in r6.items()}
    print(f"   {out['frr_unamb']}")

    print("7. abstention without the block cap")
    out["uncapped_B"] = {}
    for p0 in (0.1, 0.2, 0.3):
        for B in (800, 10000):
            r = experiment(partial(seq_accept, eps=p0 + replay.MARGIN, max_blocks=B), [p0], reps=500, seed=13, p0=p0)
            out["uncapped_B"][f"{p0}-{B}"] = {"accept": r["correct_accept"], "blocks": r["blocks"] / r["reps"]}
    print("   " + json.dumps({k: [round(100 * v["accept"], 1), round(v["blocks"])] for k, v in out["uncapped_B"].items()}))

    hi = [i for i, ep in enumerate(flaky) if ep["p0"] >= 0.2]
    r7 = evaluate_rules(flaky, lambda ep: {"SeqAccept, B=10000": partial(seq_accept, eps=eps_of(ep), max_blocks=10000)})["SeqAccept, B=10000"]
    s7 = replay.summarize(r7, flaky)
    out["realflaky_uncapped_B"] = {"far": s7["far"], "frr": s7["frr"], "blocks_per_accept": s7["blocks_per_accept"],
                                   "frr_high": replay.summarize([r7[i] for i in hi], [flaky[i] for i in hi])["frr"]}
    print(f"   real flaky, B = 10000: {out['realflaky_uncapped_B']}")

    print("8. tasks left out for grok-4.7")
    dropped = [replay.load_episode(p) for p in paths if replay.excluded(p) and "@b7" not in p.stem]
    dropped = [ep for ep in dropped if ep.get("flakiness", "natural") == "natural"]
    both = natural + dropped
    r8 = evaluate_rules(both, lambda ep: all_rules(eps=eps_of(ep)))
    out["dropped_tasks"] = {"loops_added": len(dropped), "tasks": sorted({task_of(ep) for ep in dropped}),
                            "with": {n: mean(r["far"] for r in rs) for n, rs in r8.items()},
                            "without": {n: mean(per[ep["name"]]["rules"][n]["far"] for ep in natural) for n in r8}}
    print(f"   +{len(dropped)} loops: B1 {out['dropped_tasks']['without']['B1 stop at first pass']:.3f} -> {out['dropped_tasks']['with']['B1 stop at first pass']:.3f}")

    print("9. drift over the recording window: run 1 against run 2 of the same tasks")
    out["drift"] = {}
    for model in sorted({ep["model"] for ep in natural}):
        runs = {r: [e for e in natural if e["model"] == model and e["name"].endswith(f"-{r}")] for r in (1, 2)}
        f = lambda xs: mean(per[e["name"]]["rules"]["B1 stop at first pass"]["far"] for e in xs)
        out["drift"][model] = {"run1": f(runs[1]), "run2": f(runs[2]), "loops": len(runs[1]),
                               "run1_start": min(e["agent"]["started"] for e in runs[1])[:10],
                               "run2_end": max(e["agent"]["started"] for e in runs[2])[:10]}
    print(f"   {json.dumps({m: [round(100 * v['run1'], 1), round(100 * v['run2'], 1), v['loops']] for m, v in out['drift'].items()})}")

    print("10. Corollary 4.3: eps = L + delta, delta = 0.05 + sup(p0hat - L) at 400 blocks, scored against p0 + delta")
    from extra_analyses import margin_labels
    from intervals import cp_lower
    nb = json.loads(Path(f"tasks/{task_of(natural[0])}.json").read_text())["p0_blocks"]
    gap = max(k / nb - cp_lower(k, nb) for k in range(nb + 1))
    delta = replay.MARGIN + round(gap, 3)
    cor = "SeqAccept, eps = L + delta"
    lab = margin_labels(delta)
    rule_cor = lambda ep: {cor: all_rules(eps=cp_lower(round(ep["p0"] * nb), nb) + delta)["SeqAccept"]}
    c10 = cost_rows(natural, per, rule_cor)[cor]
    main_eps = [ep for ep in episodes if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS]
    r10 = evaluate_rules(main_eps, rule_cor, labels_of=lab)[cor]
    far_of = lambda fl: mean(r["far"] for r, ep in zip(r10, main_eps) if ep.get("flakiness", "natural") == fl)
    nat_r = [r for r, ep in zip(r10, main_eps) if ep.get("flakiness", "natural") == "natural"]
    nat_e = [ep for ep in main_eps if ep.get("flakiness", "natural") == "natural"]
    hit = {bug_of(ep) for ep, r in zip(nat_e, nat_r) if r["far"] > 0}
    with_b = {bug_of(ep) for ep in nat_e if "broken" in lab(ep)}
    rf10 = evaluate_rules(flaky, rule_cor, labels_of=lab)[cor]
    s_rf = replay.summarize(rf10, flaky)
    out["corollary"] = {"delta": delta, "natural": {**c10, "far": mean(r["far"] for r in nat_r),
                                                     "bugs_hit": len(hit), "bugs_with_broken": len(with_b),
                                                     "bugs_upper": replay.cp_upper(0, len(with_b)) if not hit else None},
                        "bernoulli": far_of("injected-bernoulli"), "markov": far_of("injected-markov"),
                        "realflaky": {"far": s_rf["far"], "frr": s_rf["frr"], "blocks_per_accept": s_rf["blocks_per_accept"],
                                      "far_ci_repo": list(replay.cluster_ci([r["far"] for r in rf10], [ep.get("repo") for ep in flaky])),
                                      "frr_high": replay.summarize([rf10[i] for i in hi], [flaky[i] for i in hi])["frr"]}}
    print(f"   delta {delta}: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out['corollary'].items() if k != 'natural'})}")

    print("11. real-flaky FAR over loops with a Broken candidate; worst task")
    main4 = ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "B6 SeqAccept, no refund", "SeqAccept")
    r11 = evaluate_rules(flaky, lambda ep: {n: all_rules(eps=eps_of(ep))[n] for n in main4})
    wb = [i for i, ep in enumerate(flaky) if "broken" in replay.labels(ep)]
    out["realflaky_broken_loops"] = {"loops": len(wb), "of": len(flaky)}
    for n in main4:
        by = {}
        for i, ep in enumerate(flaky):
            by.setdefault(task_of(ep), []).append(r11[n][i])
        worst = max(by, key=lambda t: mean(r["far"] for r in by[t]))
        k = sum(round(r["far"] * r["reps"]) for r in by[worst])
        m_ = sum(r["reps"] for r in by[worst])
        out["realflaky_broken_loops"][n] = {"far": mean(r11[n][i]["far"] for i in wb), "worst_task": worst,
                                            "worst_far": k / m_, "worst_upper": replay.cp_upper(k, m_)}
    print(f"   {json.dumps(out['realflaky_broken_loops'], default=str)}")

    print("12. refund break-even against harmonic spending: blocks to accept a candidate that never fails")
    pass_all = lambda a: fixed_spending(1, lambda j: 0, slices=lambda j: a, eps=0.05, max_blocks=10 ** 6).blocks
    g = 0.5
    worst_j = next(j for j in range(1, 100) if g * (1 - g) ** (j - 1) < 1 / (j * (j + 1)))
    out["refund_breakeven"] = {"refund": pass_all(g * ALPHA), "worst_from": worst_j,
                               **{f"harmonic_{j}": pass_all(ALPHA / (j * (j + 1))) for j in (5, 15, 40)},
                               **{f"norefund_{j}": pass_all(ALPHA * g * (1 - g) ** (j - 1)) for j in (5, 15)}}
    print(f"   {out['refund_breakeven']}")

    Path("results/checks.json").write_text(json.dumps(out, indent=1, default=str))
    print("saved results/checks.json")
