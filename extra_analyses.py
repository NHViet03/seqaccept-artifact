"""Offline analyses asked for in review round 1 (no LLM calls). Writes results/extra.json.

1. Valid no-refund spending schedules (harmonic, Bonferroni over the 15-attempt budget) and a rerun rule with the
   block cost of one SeqAccept accept, on the main episodes, the late streams and the worst-case simulation.
2. Refund stress: simulated streams that alternate quickly rejected and near-eps candidates, so refunds happen and the
   wallet cap binds; SeqAccept with and without the cap.
3. The recorded loops' own first green: the label of the first candidate whose feedback block passed, and every rule
   replayed on streams cut at that candidate.
4. Margin 0.10 with labels at eps (Broken = p > p0 + 0.10).
5. Boundary stress scored against the pool's failure rate, the quantity Theorem 6 is about in replay.
6. Correlated blocks: block outcomes of a candidate follow a two-state Markov chain with stationary failure rate p
   and lag-1 correlation rho (Assumption 1 violated on purpose).
7. Serial dependence in the recorded blocks: a runs test per candidate over its blocks in seed order.
8. Late streams that mix clearly broken and near-eps candidates (k <= 15), then one correct candidate.
9. Full-suite arm (fullsuite.py), when its results exist: B1-B3 on whole-suite runs instead of sampled blocks.

Usage: uv run python extra_analyses.py
"""
import json
import random
import zlib
from statistics import median
from functools import partial
from pathlib import Path

import replay
from rules import all_rules, bonferroni, discard_spending, fixed_n, fixed_spending, harmonic, rerun_k, seq_accept
from sim import experiment, label, run_episodes

REPS = replay.REPS
WORST_P, WORST_N = (0.051, 0.06, 0.08), (1, 5, 15, 50)  # as run_sim.py
STRESS = {"alt-0.10": [0.10, 0.051] * 25, "alt-0.08": [0.08, 0.051] * 25}  # quickly rejected, then near eps
STRESS_REPS = 1000
replay_rules_fixed = ("B1 stop at first pass", "B2 rerun-3 all green", "B3 majority of 5")
RHOS = (0.0, 0.2, 0.5, 0.8)  # lag-1 correlation between consecutive blocks of one candidate
CORR_P, CORR_N = 0.06, (1, 15)
MIXED_K = (5, 10, 15)


FIXN_SCALES = (1, 2, 4)  # fixed-n sample sizes, in multiples of the fewest blocks at which a perfect candidate passes


def extra_rules(eps, k_match):
    return {"Harmonic spending": partial(fixed_spending, slices=harmonic(), eps=eps),
            "Discard spending": partial(discard_spending, eps=eps),
            "Bonferroni-15 spending": partial(fixed_spending, slices=bonferroni(15), eps=eps),
            **{f"Fixed-n Bonferroni-15 ({k}n0)": partial(fixed_n, slices=bonferroni(15), eps=eps, scale=k) for k in FIXN_SCALES},
            f"Rerun-{k_match} (all green)": lambda n, block: rerun_k(n, k_match, block)}


def evaluate_rules(episodes, rules_of, reps=REPS, labels_of=replay.labels):
    out = {}
    for i, ep in enumerate(episodes):
        seed = replay.episode_seed(ep, i)
        for name, rule in rules_of(ep).items():
            out.setdefault(name, []).append(replay_with(rule, ep, reps, seed, labels_of))
    return out


def replay_with(rule, ep, reps, seed, labels_of):
    """replay.replay with a choice of labels (same draws, same costs)."""
    if labels_of is replay.labels:
        return replay.replay(rule, ep, reps, seed=seed)
    fixes = ep["fixes"]
    rng = random.Random(seed)
    pools = [fixes[f]["pool"] for f in ep["stream"]]
    costs = [sum(min(s, replay.CAP_SECONDS) for s in fixes[f]["seconds"]) / len(fixes[f]["seconds"])
             for f in ep["stream"]]
    mask = ep.get("mask", 0.0)

    def draw(j):
        y = rng.choice(pools[j])
        return 0 if y and mask and rng.random() < mask else y

    return run_episodes(rule, labels_of(ep), lambda: draw, reps, cost=lambda j: costs[j])


def near_eps(eps_list):
    """Distinct Broken candidates per episode: how many lie within 0.03 and 0.05 of eps by truth rate, and how many
    have a replay pool rate below eps (such a candidate looks Correct to every rule)."""
    near = {"broken": 0, "within_003": 0, "within_005": 0, "pool_below_eps": 0}
    for ep in eps_list:
        eps, seen = ep["p0"] + replay.MARGIN, set()
        for f, lab in zip(ep["stream"], replay.labels(ep)):
            if lab == "broken" and f not in seen:
                seen.add(f)
                fx = ep["fixes"][f]
                rate = sum(fx["truth"]) / len(fx["truth"])
                near["broken"] += 1
                near["within_003"] += rate <= eps + 0.03
                near["within_005"] += rate <= eps + 0.05
                near["pool_below_eps"] += sum(fx["pool"]) / len(fx["pool"]) < eps
    return near


def summary(results, episodes):
    return {name: replay.summarize(rs, episodes) for name, rs in results.items()}


def first_green(episodes):
    """Per episode: index of the first candidate whose feedback block passed in the recorded loop (None if none)."""
    out = []
    for ep in episodes:
        task, model, run = ep["name"].split("-")[0].split("@")[0], ep["name"].split("-", 1)[1].rsplit("-", 1)[0], \
            ep["name"].rsplit("-", 1)[1]
        stream = json.loads(Path(f"streams/{task}-{model}-{run}.json").read_text())["stream"]
        assert [c["fix_id"] for c in stream] == ep["stream"], ep["name"]
        out.append(next((i for i, c in enumerate(stream) if not c["agent_failed"]), None))
    return out


def margin_labels(margin):
    def labels_of(ep):
        fixes = ep["fixes"]
        rates = [sum(fixes[f]["truth"]) / len(fixes[f]["truth"]) for f in ep["stream"]]
        return ["correct" if p < ep["p0"] + 0.01 else "broken" if p > ep["p0"] + margin else "marginal" for p in rates]
    return labels_of


def pool_labels(ep):
    """Labels from the (masked) pool rate: what a rule's draws actually estimate in replay."""
    keep = 1 - ep.get("mask", 0.0)
    return [label(keep * sum(ep["fixes"][f]["pool"]) / len(ep["fixes"][f]["pool"]), ep["p0"]) for f in ep["stream"]]


def correlated(rule, p, n, rho, reps, seed):
    """FAR of `rule` on n broken candidates whose blocks follow a stationary Markov chain (rate p, lag-1 corr rho)."""
    rng = random.Random(seed)

    def make_block():
        last = {}

        def block(j):
            if j not in last:
                y = rng.random() < p
            else:
                y = rng.random() < (p + rho * (1 - p) if last[j] else p * (1 - rho))
            last[j] = y
            return int(y)
        return block

    return run_episodes(rule, [label(p)] * n, make_block, reps)["far"]


def runs_test(ys):
    """Two-sided Wald-Wolfowitz runs test p < 0.05 for a 0/1 sequence in run order (None if constant)."""
    n1, n0 = sum(ys), len(ys) - sum(ys)
    if not n1 or not n0:
        return None
    runs = 1 + sum(a != b for a, b in zip(ys, ys[1:]))
    mu = 2 * n1 * n0 / (n1 + n0) + 1
    var = (mu - 1) * (mu - 2) / (n1 + n0 - 1)
    return var > 0 and abs(runs - mu) / var ** 0.5 > 1.96


def lag1_corr(ys):
    """Lag-1 autocorrelation of a non-constant 0/1 sequence."""
    m = sum(ys) / len(ys)
    return sum((a - m) * (b - m) for a, b in zip(ys, ys[1:])) / sum((a - m) ** 2 for a in ys)


def fullsuite_episodes(episodes):
    """Episodes whose pools are full-suite random-order outcomes (fullsuite.py), and default-order outcomes."""
    out = []
    for ep in episodes:
        variant = ep["name"].split("-")[0]
        files = {f: Path(f"fullsuite/{variant}__{f}.json") for f in ep["fixes"]}
        if not all(x.exists() for x in files.values()):
            return None
        res = {f: json.loads(x.read_text()) for f, x in files.items()}
        out.append({**ep, "default": {f: r["default"] for f, r in res.items()},
                    "fixes": {f: {**v, "pool": res[f]["random"], "seconds": res[f]["seconds"]} for f, v in ep["fixes"].items()}})
    return out


if __name__ == "__main__":
    paths = sorted(Path("episodes").glob("*.json"))
    episodes = [replay.load_episode(p) for p in paths if not replay.excluded(p)]
    episodes = [ep for ep in episodes if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS]
    assert len(episodes) == json.loads(Path("results/results.json").read_text())["episodes"]
    oc = json.loads(Path("results/operating_curve.json").read_text())
    k_match = round(next(r["blocks"] for r in oc["single"] if r["p"] == 0.0))  # blocks SeqAccept needs on a perfect fix
    report = {"k_match": k_match}

    print("1. spending baselines and cost-matched rerun: main episodes")
    main = evaluate_rules(episodes, lambda ep: extra_rules(ep["p0"] + replay.MARGIN, k_match))
    report["main"] = summary(main, episodes)
    report["late"] = {}
    for k in replay.LATE_K:
        mine = replay.late_streams(episodes, k)
        print(f"   late streams k = {k}")
        report["late"][k] = summary(evaluate_rules(mine, lambda ep: extra_rules(ep["p0"] + replay.MARGIN, k_match)), mine)
    report["worst"] = []
    for p in WORST_P:
        for n in WORST_N:
            reps = 1000 if n < 50 else 400
            cells = {name: experiment(rule, [p] * n, reps=reps, seed=31) for name, rule in extra_rules(0.05, k_match).items()}
            report["worst"].append({"p": p, "n": n, "reps": reps, "far": {k_: c["far"] for k_, c in cells.items()}})
            print(f"   worst p = {p}, n = {n}")

    print("2. refund stress")
    report["stress"] = {}
    stress_rules = {"SeqAccept": seq_accept, "SeqAccept, uncapped refund": partial(seq_accept, refund="uncapped"),
                    "B6 SeqAccept, no refund": partial(seq_accept, refund=False),
                    "Harmonic spending": partial(fixed_spending, slices=harmonic()),
                    "Discard spending": discard_spending,
                    "B5 naive confidence sequence": all_rules()["B5 naive confidence sequence"],
                    "B4 peeking Clopper-Pearson": all_rules()["B4 peeking Clopper-Pearson"]}
    for design, ps in STRESS.items():
        report["stress"][design] = {"ps": ps[:2], "n": len(ps), "reps": STRESS_REPS,
                                    "far": {name: experiment(rule, ps, reps=STRESS_REPS, seed=41)["far"]
                                            for name, rule in stress_rules.items()}}
        print(f"   {design}: {report['stress'][design]['far']}")

    print("3. first green of the recorded loops")
    greens = first_green(episodes)
    natural = [(ep, g) for ep, g in zip(episodes, greens) if ep.get("flakiness", "natural") == "natural"]
    by_model = {}
    for ep, g in natural:
        lab = None if g is None else replay.labels(ep)[g]
        d = by_model.setdefault(ep["model"], {"episodes": 0, "broken": 0, "correct": 0, "marginal": 0, "none": 0})
        d["episodes"] += 1
        d[lab or "none"] += 1
    report["first_green"] = {"natural": by_model}
    cut = [{**ep, "stream": ep["stream"][:g + 1]} for ep, g in zip(episodes, greens) if g is not None]
    res = replay.evaluate(cut)
    res.update(evaluate_rules(cut, lambda ep: extra_rules(ep["p0"] + replay.MARGIN, k_match)))
    report["truncated"] = {"episodes": len(cut), "dropped_no_green": greens.count(None), "table": summary(res, cut)}

    print("4. margin 0.10 with labels at eps")
    res = evaluate_rules(episodes, lambda ep: all_rules(eps=ep["p0"] + 0.10), labels_of=margin_labels(0.10))
    report["margin_relabel"] = {"margin": 0.10, "table": {n: {"far": s["far"]} for n, s in summary(res, episodes).items()}}

    print("5. boundary stress against pool labels")
    natural_eps = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    report["boundary_pool"] = {}
    for m in replay.MASKS:
        mine = [replay.masked(ep, m) for ep in natural_eps]
        res = evaluate_rules(mine, lambda ep: all_rules(eps=ep["p0"] + replay.MARGIN), reps=replay.NEAR_REPS,
                             labels_of=pool_labels)
        report["boundary_pool"][m] = {n: {"far": s["far"], "false_accepts": round(sum(r["far"] * r["reps"] for r in res[n]))}
                                      for n, s in summary(res, mine).items()}
        print(f"   m = {m}: " + " ".join(f"{n[:10]} {100 * v['far']:.2f}" for n, v in report["boundary_pool"][m].items()))


    print("6. correlated blocks")
    corr_rules = {k: v for k, v in stress_rules.items() if k not in ("SeqAccept, uncapped refund", "Discard spending")}
    report["correlated"] = {"p": CORR_P, "rows": []}
    for rho in RHOS:
        for n in CORR_N:
            reps = 1000
            report["correlated"]["rows"].append({"rho": rho, "n": n, "reps": reps,
                                                 "far": {k: correlated(r, CORR_P, n, rho, reps, 51) for k, r in corr_rules.items()}})
            print(f"   rho = {rho}, n = {n}: {report['correlated']['rows'][-1]['far']}")

    print("7. serial dependence of recorded blocks (seed order)")
    tested = rejected = 0
    lag1 = []
    for ep in episodes:
        for f in ep["fixes"].values():
            # all blocks in seed order: truth then pool, or the shared 40 of a clearly broken candidate
            seq = f["truth"] + f["pool"] if len(f["truth"]) + len(f["pool"]) == len(f["seeds"]) else f["truth"]
            r = runs_test(seq)
            if r is not None:
                tested += 1
                rejected += r
                lag1.append(lag1_corr(seq))
    lag1.sort()
    report["serial"] = {"tested": tested, "rejected": rejected, "lag1_median": lag1[len(lag1) // 2],
                        "lag1_mean": sum(lag1) / len(lag1), "lag1_abs_over_02": sum(abs(r) > 0.2 for r in lag1)}

    print("7b. near-eps candidates in the recorded (natural) loops")
    near = near_eps([ep for ep in episodes if ep.get("flakiness", "natural") == "natural"])
    report["near_eps"] = near
    print(f"   {near}")
    print(f"   {rejected} of {tested} candidates reject independence at 5%")

    print("8. mixed late streams")
    report["mixed_late"] = {}
    mixed_rules = {"SeqAccept": seq_accept, "B6 SeqAccept, no refund": partial(seq_accept, refund=False),
                   "Harmonic spending": partial(fixed_spending, slices=harmonic()),
                   "Discard spending": discard_spending,
                   "Bonferroni-15 spending": partial(fixed_spending, slices=bonferroni(15)),
                   **{f"Fixed-n Bonferroni-15 ({k}n0)": partial(fixed_n, slices=bonferroni(15), scale=k) for k in FIXN_SCALES}}
    for k in MIXED_K:
        ps = [0.5 if i % 2 == 0 else CORR_P for i in range(k)] + [0.0]
        report["mixed_late"][k] = {name: experiment(rule, ps, reps=1000, seed=61) for name, rule in mixed_rules.items()}
        print(f"   k = {k}: " + " ".join(f"{n[:10]} {100 * v['correct_accept']:.1f}/{100 * v['far']:.1f}"
                                        for n, v in report["mixed_late"][k].items()))

    full = fullsuite_episodes(episodes)
    if full:
        print("9. full-suite arm")
        res = evaluate_rules(full, lambda ep: {n: r for n, r in all_rules().items() if n in replay_rules_fixed})
        report["fullsuite"] = {"table": {n: {"far": s["far"], "far_ci": s["far_ci"]} for n, s in summary(res, full).items()}}
        for fl in replay.MAIN_FLAKINESS:
            idx = [i for i, ep in enumerate(full) if ep.get("flakiness", "natural") == fl]
            report["fullsuite"][fl] = {n: sum(res[n][i]["far"] for i in idx) / len(idx) for n in res}
        nat = [ep for ep in full if ep.get("flakiness", "natural") == "natural"]
        first = []
        for ep in nat:
            labs = replay.labels(ep)
            j = next((i for i, f in enumerate(ep["stream"]) if ep["default"][f] == 0), None)
            first.append(None if j is None else labs[j])
        report["fullsuite"]["default_first_pass"] = {"episodes": len(nat), "broken": first.count("broken"),
                                                     "correct": first.count("correct"), "marginal": first.count("marginal"),
                                                     "none": first.count(None)}
        passing = {}  # natural bugs with a Broken candidate that passed a random-order full-suite run: bug -> bug kind
        for ep in nat:
            v = ep["name"].split("-")[0]
            if any(lab == "broken" and 0 in ep["fixes"][f]["pool"] for f, lab in zip(ep["stream"], replay.labels(ep))):
                passing[v] = json.loads(Path(f"tasks/{v}.json").read_text())["bug_kind"]
        report["fullsuite"]["natural_false_pass_bugs"] = passing
        passes = {}  # distinct Broken natural candidates by how many of their full-suite runs passed
        for ep in nat:
            for f, lab in zip(ep["stream"], replay.labels(ep)):
                if lab == "broken":
                    passes[(ep["name"].split("-")[0], f)] = ep["fixes"][f]["pool"].count(0)
        report["fullsuite"]["broken_passes"] = {k: list(passes.values()).count(k) for k in range(6)}
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        report["fullsuite"]["cost"] = {  # per candidate: mean seconds of a full-suite run and of a recorded block
            "full_seconds": mean([mean(v["seconds"]) for ep in full for v in ep["fixes"].values()]),
            "block_seconds": mean([mean(v["seconds"]) for ep in episodes for v in ep["fixes"].values()]),
            "full_median": median([x for p in Path("fullsuite").glob("*.json") for x in json.loads(p.read_text())["seconds"]]),
            "block_median": median([x for ep in episodes for v in ep["fixes"].values() for x in v["seconds"]]),
            "tests_per_block": mean([len(t["test_ids"]) / t["block_size"] for t in
                                     (json.loads(Path(f"tasks/{ep['name'].split('-')[0]}.json").read_text()) for ep in full)])}
        print(f"   {report['fullsuite']}")
    else:
        print("9. full-suite arm: results not complete, skipped")

    import flaky_arm
    expected = [n for b, _, n in flaky_arm.jobs() if Path(f"tasks/{b}@realflaky.json").exists()]
    done = [Path(f"{flaky_arm.OUT}/{n}.json") for n in expected]
    if expected and all(p.exists() for p in done) and all(
            json.loads(Path(f"tasks/{b}@realflaky.json").read_text())["p0"] is not None for b, _, n in flaky_arm.jobs()
            if n in expected):
        print("10. real-flaky arm")
        flaky = [replay.load_episode(p) for p in done]
        same = {n.replace("@realflaky", "") for n in expected}  # the same loops without the flaky tests
        base = [ep for ep in episodes if ep["name"] in same]
        report["realflaky"] = {"episodes": len(flaky), "bugs": len({ep["name"].split("-")[0] for ep in flaky}),
                               "p0": {b: json.loads(Path(f"tasks/{b}@realflaky.json").read_text())["p0"]
                                      for b in sorted({ep["name"].split("-")[0].split("@")[0] for ep in flaky})},
                               "table": summary(res := replay.evaluate(flaky), flaky),
                               "same_loops_without": summary(replay.evaluate(base), base),
                               "near_eps": near_eps(flaky), "near_eps_without": near_eps(base),
                               "seqaccept_false_accept_bugs": sorted({ep["name"].split("-")[0].split("@")[0]
                                                                      for r, ep in zip(res["SeqAccept"], flaky) if r["far"] > 0})}
        for part, keep in (("p0_high", lambda ep: ep["p0"] >= 0.2), ("p0_low", lambda ep: ep["p0"] < 0.2)):
            idx = [i for i, ep in enumerate(flaky) if keep(ep)]
            report["realflaky"][part] = {"episodes": len(idx), "table": summary({n: [rs[i] for i in idx] for n, rs in res.items()},
                                                                                [flaky[i] for i in idx])}
        rf = report["realflaky"]
        rf["repos"] = len({ep.get("repo") for ep in flaky})
        labs = [replay.labels(ep) for ep in flaky]
        idx = [i for i, lab in enumerate(labs) if "correct" in lab]
        rf["frr_ci_seqaccept"] = replay.cluster_ci([res["SeqAccept"][i]["frr"] for i in idx], [flaky[i].get("repo") for i in idx])
        rf["p0_se"] = {b_: (p * (1 - p) / 400) ** 0.5 for b_, p in rf["p0"].items()}  # 400 blocks of the original code
        rf["margin_010"] = summary(evaluate_rules(flaky, lambda ep: all_rules(eps=ep["p0"] + 0.10), labels_of=margin_labels(0.10)), flaky)
        print(f"   {len(flaky)} episodes; p0 {rf['p0']}; margin 0.10: " + " ".join(
            f"{n[:10]} far {100 * v['far']:.2f} frr {100 * (v['frr'] or 0):.1f}" for n, v in rf["margin_010"].items()))
    else:
        print(f"10. real-flaky arm: {sum(p.exists() for p in done)}/{len(expected)} episodes, skipped")

    Path("results/extra.json").write_text(json.dumps(report, indent=2, default=str))
    print("saved results/extra.json")
