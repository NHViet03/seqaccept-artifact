"""Deployment checks (EASE round 9, offline, no LLM or Docker). Writes results/deploy_checks.json.

1. Refund share under dependent blocks: SeqAccept with a share e in {0, 0.5, 1} of the maximal refund, and with
   gamma = 1/4, on 1 and 15 broken candidates at p = 0.06 whose blocks follow a Markov chain with lag-1 correlation
   phi in {0, 0.2, 0.5} (1,000 runs, seed 51, as extra_analyses.py step 6); and the same variants' cost on natural,
   near-eps and real-flaky loops (blocks per accept and FRR). Every variant is covered by Theorem 4.2.
2. Deployment rows in matched conditions: the default, Corollary 4.3 at total level 0.05 with margin 0.05
   (eps = L_0.99 + 0.05, alpha = 0.04, alpha_0 = 0.01), B4 and fixed-n Bonferroni-15, all scored against labels at
   p0hat + 0.05: FAR, FRR, FRR at p0 >= 0.2 and blocks per accept with the task's p0 blocks repeated per loop.
3. On-policy prefix: each recorded stream cut after its first candidate whose feedback block passed (a whole stream
   if none did). Every earlier candidate failed its feedback, so a rule that rejects them sees the successors the agent
   wrote for that history. Share of candidates written after a passed feedback, and FAR of the main rules on the cut
   streams.
4. Screening: the chosen repositories that came from IDoFT screening against the unchosen usable pairs, on the
   flaky tests' failure rate, the suite's failure rate without them, and seconds per block.
5. Latency at 8 blocks at a time for B4 beside SeqAccept (blocks per accept, no calibration).
6. Break-even dependence: SeqAccept's FAR on 15 and 50 broken candidates at p = 0.06 against lag-1 correlation phi
   from 0.05 to 0.5 (2,000 runs, seed 51), and the smallest phi of the grid with FAR above alpha.
7. Yamicache with flaky tests kept: SeqAccept's false acceptances in the main replays (same seeds), how many fall on
   candidates whose label has more than a 5% posterior chance of being wrong, and on how many distinct candidates.
8. Near-eps arm scored on all blocks (near_eps_checks.py step 7, same seeds): false acceptances of B4, B5 and SeqAccept
   per distinct candidate.
9. Discarding on constructed streams of 15 and 50 candidates at p = 0.051 and 0.06 (2,000 runs, seed 41).
10. Discarding against the white-box adversary of caps_adversary.py, which sees the pool before each candidate: policy
    "near" (every candidate at p = 0.051) and "switch" (near candidates while the pool is at least alpha / 2, then
    alternating quickly rejected ones at p = 0.9 with near ones), 15 and 50 candidates, 2,000 runs.
12. The default at alpha = 0.04 on every arm (with alpha_0 = 0.01 it protects the true p0 + 0.109 at total level
    0.05, the plug-in corollary), and the default at 0.05 and 0.04 scored against labels at p0hat + 0.109.
13. Deflated-eps accept wallet under lag-1 dependence: SeqAccept run at eps' = eps (1 - phi_max), phi_max 0.25 and 0.5,
    on 15 broken candidates at p = 0.06 with phi 0, 0.2 and 0.5 (2,000 runs, seed 51; FAR scored at eps), and its FRR
    and cost on natural and real-flaky loops. Under a two-state lag-1 chain a broken candidate fails after a pass with
    probability p (1 - phi) > eps', so the wallet stays a supermartingale for phi <= phi_max.
14. If results/flaky_truth.json exists (neareps_truth.py on yamicache@realflaky, Docker): yamicache relabelled with all
    of each re-measured candidate's blocks as pool and label, SeqAccept's FAR there, and the re-measured rates of the
    candidates behind its false acceptances.
11. Exposure: share of recorded loops with at least 1 and at least 5 Broken candidates (stream positions) failing within
    one point above eps on their truth blocks, the counts at which B4 and B5 exceed alpha on the constructed grid; the
    same within 3 points; and the posterior chance (Beta on the truth blocks) of at least one within one point.

  uv run python deploy_checks.py [--resume]     all steps (--resume keeps steps 1-5 from the saved results)
  uv run python deploy_checks.py --exposure     step 11 only
  uv run python deploy_checks.py --dependence-cost   step 1's cost and FRR on recorded loops only
"""
import csv
import json
import random
from functools import partial
from pathlib import Path
from statistics import mean, median

import replay
import sys
from caps_adversary import ambiguous_labels
from extra_analyses import correlated, evaluate_rules, first_green, margin_labels
from intervals import cp_lower
from replay import cp_upper
from near_eps_checks import firmer
from rules import all_rules, bonferroni, discard_spending, fixed_n, seq_accept
from sim import experiment

ALPHA, CORR_P, PHIS, NS = 0.05, 0.06, (0.0, 0.2, 0.5), (1, 15)
eps_of = lambda ep: ep["p0"] + replay.MARGIN
load = lambda d, keep=lambda p: True: [replay.load_episode(p) for p in sorted(Path(d).glob("*.json")) if keep(p)]
MAIN = ("B1 stop at first pass", "B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "B6 SeqAccept, no refund", "SeqAccept")


def variants(eps=0.05):
    return {"e1": partial(seq_accept, eps=eps), "e05": partial(seq_accept, eps=eps, refund=0.5),
            "e0": partial(seq_accept, eps=eps, refund=0.0), "g025": partial(seq_accept, eps=eps, gamma=0.25)}


def p0_blocks(ep):
    return json.loads(Path(f"tasks/{ep['name'].split('-')[0]}.json").read_text())["p0_blocks"]


def deployment(ep):
    e, nb = eps_of(ep), p0_blocks(ep)
    return {"default": partial(seq_accept, eps=e),
            "total05": partial(seq_accept, alpha=0.04, eps=cp_lower(round(ep["p0"] * nb), nb, 0.01) + replay.MARGIN),
            "B4": all_rules(eps=e)["B4 peeking Clopper-Pearson"],
            "fixedn15": partial(fixed_n, slices=bonferroni(15), eps=e, scale=1)}


def row(rs, eps_list):
    s = replay.summarize(rs, eps_list)
    hi = [i for i, ep in enumerate(eps_list) if ep["p0"] >= 0.2]
    acc = sum(r["accepts"] for r in rs)
    calib = sum(p0_blocks(ep) * r["reps"] for ep, r in zip(eps_list, rs))
    return {"far": s["far"], "frr": s["frr"], "blocks_per_accept": s["blocks_per_accept"],
            "frr_high": replay.summarize([rs[i] for i in hi], [eps_list[i] for i in hi])["frr"] if hi else None,
            "bpa_loop": (sum(r["blocks"] for r in rs) + calib) / acc if acc else None,
            "worst_loop": max(r["far"] for r in rs)}


def prefix(episodes):
    """(cut episodes, candidates written after a passed feedback, all candidates)."""
    greens = first_green(episodes)
    cut = [{**ep, "stream": ep["stream"][:g + 1]} if g is not None else ep for ep, g in zip(episodes, greens)]
    after = sum(len(ep["stream"]) - g - 1 for ep, g in zip(episodes, greens) if g is not None)
    return cut, after, sum(len(ep["stream"]) for ep in episodes)


def accepted(rule, episodes, reps):
    """Per episode, the stream index each replay accepted (None for none), with evaluate_rules' seeds."""
    out = []
    for i, ep in enumerate(episodes):
        got = []

        def wrapped(n, block):
            r = rule(ep)(n, block)
            got.append(r.accepted)
            return r
        replay.replay(wrapped, ep, reps, seed=replay.episode_seed(ep, i))
        out.append(got)
    return out


def discard_adversary(policy, n, reps, seed, near_p=0.051, far_p=0.9):
    """FAR of discard_spending when an adversary that sees the pool before each candidate picks its failure rate."""
    rng, hits = random.Random(seed), 0
    for _ in range(reps):
        st = {"pool": ALPHA, "j": -1, "p": near_p, "k": 0}

        def block(j):
            if j != st["j"]:
                st["j"] = j
                if policy == "near" or st["pool"] >= ALPHA / 2:
                    st["p"] = near_p
                else:
                    st["k"] += 1
                    st["p"] = far_p if st["k"] % 2 else near_p
            return int(rng.random() < st["p"])
        hits += discard_spending(n, block, trace=lambda pool: st.__setitem__("pool", pool)).accepted is not None
    return hits / reps


def screening():
    rows = list(csv.DictReader(open("screening/summary.csv")))
    cand = {r["slug"].rsplit("_", 1)[0]: r for r in rows if r["verdict"] == "candidate"}
    repos = {json.loads(p.read_text()).get("repo") for p in Path("tasks").glob("*.json")} - {None}
    chosen = {r.replace("/", "_").replace("-", "_").lower() for r in repos}
    out = {}
    for tag, keep in (("chosen", lambda s: s in chosen), ("unchosen", lambda s: s not in chosen)):
        g = [r for s, r in cand.items() if keep(s)]
        f = lambda k: [float(r[k]) for r in g if r.get(k) not in (None, "")]
        out[tag] = {"n": len(g), "flaky_rate": median(f("flaky_rate")), "p0": median(f("p0_estimate")),
                    "seconds_per_block": median(f("seconds_per_block"))}
    return out


def dependence_cost(out, arms):
    """Step 1, cost side: blocks per accept and FRR of each refund variant on the three arms (same seeds)."""
    rows = {arm: {k: row(rs, e_) for k, rs in evaluate_rules(e_, lambda ep: variants(eps_of(ep)), reps=reps).items()}
            for arm, e_, reps in arms}
    out["dependence_cost"] = {a: {k: v["blocks_per_accept"] for k, v in d.items()} for a, d in rows.items()}
    out["dependence_frr"] = {a: {k: v["frr"] for k, v in d.items()} for a, d in rows.items()}


def run15(out, natural, near, flaky, arms):
    print("1. refund share and gamma under dependent blocks")
    out["dependence"] = {f"{phi}-{n}": {k: correlated(r, CORR_P, n, phi, 1000, 51) for k, r in variants().items()}
                         for phi in PHIS for n in NS}
    dependence_cost(out, arms)
    print("   " + json.dumps({k: {v: round(100 * x, 1) for v, x in d.items()} for k, d in out["dependence"].items()}))
    print("   " + json.dumps({a: {k: round(v or 0) for k, v in d.items()} for a, d in out["dependence_cost"].items()}))

    print("2. deployment rows, labels at p0hat + 0.05")
    out["deployment"] = {arm: {k: row(rs, e_) for k, rs in evaluate_rules(e_, deployment, reps=reps,
                                                                           labels_of=margin_labels(replay.MARGIN)).items()}
                         for arm, e_, reps in arms}
    print("   " + json.dumps({a: {k: [round(100 * v["far"], 2), round(100 * (v["frr"] or 0), 2), round(v["bpa_loop"] or 0)]
                                  for k, v in d.items()} for a, d in out["deployment"].items()}))

    print("3. on-policy prefix")
    out["prefix"] = {}
    for arm, e_ in (("natural", natural), ("realflaky", flaky)):
        cut, after, total = prefix(e_)
        rr = evaluate_rules(cut, lambda ep: {n: all_rules(eps=eps_of(ep))[n] for n in MAIN})
        full = evaluate_rules(e_, lambda ep: {n: all_rules(eps=eps_of(ep))[n] for n in MAIN})
        out["prefix"][arm] = {"after_green": after, "candidates": total,
                              "far_prefix": {n: mean(r["far"] for r in rs) for n, rs in rr.items()},
                              "far_full": {n: mean(r["far"] for r in rs) for n, rs in full.items()}}
    print("   " + json.dumps(out["prefix"]))

    print("4. screening: chosen against unchosen usable pairs")
    out["screening"] = screening()
    print("   " + json.dumps(out["screening"]))

    print("5. latency at 8 blocks at a time")
    secs = lambda e_: mean(min(x, replay.CAP_SECONDS) for ep in e_ for f in ep["fixes"].values() for x in f.get("seconds", []))
    out["latency"] = {arm: {k: out["deployment"][arm][k]["blocks_per_accept"] * secs(e_) / 8 / 60
                            for k in ("default", "B4", "fixedn15") if out["deployment"][arm][k]["blocks_per_accept"]}
                      for arm, e_, _ in arms}
    print("   " + json.dumps({a: {k: round(v, 1) for k, v in d.items()} for a, d in out["latency"].items()}))


def exposure(out, natural, flaky, near):
    """Step 11. Share of loops with at least 1 (5) Broken stream positions whose truth rate lies in (eps, eps + w], for
    w = 1 point (where B4 exceeds alpha with one candidate on the constructed grid) and w = 3 points (the near-eps arm's
    window, wider than the label's standard error); and, matched to label resolution, the posterior chance under
    Beta(1 + fails, 1 + passes) on the truth blocks that a loop holds at least one candidate with p in (eps, eps + 0.01],
    averaged over loops (distinct candidates of a loop treated as independent)."""
    from label_risk import beta_cdf
    print("11. exposure to candidates near eps")
    out["exposure"] = {}
    for arm, e_ in (("natural", natural), ("realflaky", flaky), ("neareps", near)):
        r = {"loops": len(e_)}
        for w, tag in ((0.01, ""), (0.03, "_w3")):
            counts = [sum(eps_of(ep) < mean(ep["fixes"][f]["truth"]) <= eps_of(ep) + w for f in ep["stream"]) for ep in e_]
            r[f"at_least_1{tag}"] = sum(c >= 1 for c in counts) / len(e_)
            r[f"at_least_5{tag}"] = sum(c >= 5 for c in counts) / len(e_)
        post = []
        for ep in e_:
            e, miss = eps_of(ep), 1.0
            for f in set(ep["stream"]):
                t = ep["fixes"][f]["truth"]
                a, b = 1 + sum(t), 1 + len(t) - sum(t)
                miss *= 1 - max(0.0, beta_cdf(min(e + 0.01, 1), a, b) - beta_cdf(e, a, b))
            post.append(1 - miss)
        r["posterior_1"] = mean(post)
        out["exposure"][arm] = r
    print("   " + json.dumps(out["exposure"]))


if __name__ == "__main__":
    episodes = load("episodes", lambda p: not replay.excluded(p))
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = load("episodes_flaky", lambda p: not replay.excluded(p))
    near = load("episodes_neareps")
    arms = (("natural", natural, replay.REPS), ("neareps", near, replay.NEAR_REPS), ("realflaky", flaky, replay.REPS))
    path = Path("results/deploy_checks.json")
    if "--exposure" in sys.argv or "--dependence-cost" in sys.argv:  # one step only, into the existing results
        out = json.loads(path.read_text())
        if "--exposure" in sys.argv:
            exposure(out, natural, flaky, near)
        else:
            dependence_cost(out, arms)
            print("   " + json.dumps(out["dependence_cost"]) + "\n   " + json.dumps(out["dependence_frr"]))
        path.write_text(json.dumps(out, indent=1, default=str))
        raise SystemExit("saved results/deploy_checks.json (one step only)")
    out = json.loads(path.read_text()) if "--resume" in sys.argv and path.exists() else {}
    if "deployment" in out:
        print("steps 1-5 kept from results/deploy_checks.json (--resume)")
    else:
        run15(out, natural, near, flaky, arms)

    print("6. break-even dependence")
    grid = [round(0.05 * k, 2) for k in range(1, 11)]
    out["breakeven"] = {str(n): {str(phi): correlated(seq_accept, CORR_P, n, phi, 2000, 51) for phi in grid} for n in (15, 50)}
    out["breakeven_phi"] = {n: next((float(phi) for phi, f in d.items() if f > ALPHA), None) for n, d in out["breakeven"].items()}
    print("   " + json.dumps(out["breakeven_phi"]))

    print("7. yamicache false acceptances")
    sa = lambda ep: partial(seq_accept, eps=eps_of(ep))
    acc = accepted(sa, flaky, replay.REPS)
    yam = {"false": 0, "risky": 0, "replays": 0, "distinct": set(), "distinct_risky": set()}
    for ep, got in zip(flaky, acc):
        if not ep["name"].startswith("yamicache@"):
            continue
        labs, amb = replay.labels(ep), ambiguous_labels(ep)
        yam["replays"] += len(got)
        for j in got:
            if j is not None and labs[j] == "broken":
                yam["false"] += 1
                yam["distinct"].add(ep["stream"][j])
                if amb[j] == "marginal":
                    yam["risky"] += 1
                    yam["distinct_risky"].add(ep["stream"][j])
    out["yamicache"] = {k: (len(v) if isinstance(v, set) else v) for k, v in yam.items()}
    out["yamicache"]["ids"] = sorted(yam["distinct"])
    print("   " + json.dumps(out["yamicache"]))

    print("8. near-eps false acceptances per distinct candidate (scored on all blocks)")
    pooled = firmer(near, json.loads(Path("results/neareps_truth.json").read_text()), pooled=True)
    out["near_candidates"] = {}
    for name in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "SeqAccept"):
        got = accepted(lambda ep, n=name: all_rules(eps=eps_of(ep))[n], pooled, replay.NEAR_REPS)
        per, den = {}, {}
        for ep, g in zip(pooled, got):
            labs, task = replay.labels(ep), ep["name"].split("-")[0]
            for f in {f for f, l in zip(ep["stream"], labs) if l == "broken"}:
                den[(task, f)] = den.get((task, f), 0) + len(g)  # replays of loops that hold this candidate
            for j in g:
                if j is not None and labs[j] == "broken":
                    key = (task, ep["stream"][j])
                    per[key] = per.get(key, 0) + 1
        rates = {k: (per.get(k, 0), den[k]) for k in den}
        worst = max(rates.items(), key=lambda kv: kv[1][0] / kv[1][1])
        out["near_candidates"][name] = {"distinct": len(per), "false": sum(per.values()),
                                       "top_share": max(per.values()) / sum(per.values()) if per else 0.0,
                                       "candidates": len(den),
                                       "worst": {"task": worst[0][0], "far": worst[1][0] / worst[1][1], "n": worst[1][1],
                                                 "lower": cp_lower(*worst[1]), "upper": cp_upper(*worst[1])},
                                       "above_alpha_lower": sum(cp_lower(k, n) > ALPHA for k, n in rates.values()),
                                       "above_alpha_point": sum(k / n > ALPHA for k, n in rates.values())}
    print("   " + json.dumps(out["near_candidates"]))

    print("9. discarding on constructed near-eps streams")
    out["discarding"] = {f"{n}-{p}": experiment(discard_spending, [p] * n, reps=2000, seed=41)["far"]
                         for n in (15, 50) for p in (0.051, 0.06)}
    print("   " + json.dumps(out["discarding"]))

    print("10. discarding against the white-box adversary")
    out["discard_adversary"] = {f"{pol}-{n}": discard_adversary(pol, n, 2000, 43) for pol in ("near", "switch") for n in (15, 50)}
    print("   " + json.dumps(out["discard_adversary"]))

    exposure(out, natural, flaky, near)

    print("12. default at alpha 0.04; default scored at p0hat + 0.109")
    sp_delta = json.loads(Path("results/design_checks.json").read_text())["split"]["total05"]["delta"]
    out["default04"] = {"delta": sp_delta}
    for arm, e_, reps in arms:
        d04 = evaluate_rules(e_, lambda ep: {"default04": partial(seq_accept, alpha=0.04, eps=eps_of(ep))}, reps=reps,
                             labels_of=margin_labels(replay.MARGIN))["default04"]
        wide = evaluate_rules(e_, lambda ep: {"default": partial(seq_accept, eps=eps_of(ep)),
                                              "default04": partial(seq_accept, alpha=0.04, eps=eps_of(ep))},
                              reps=reps, labels_of=margin_labels(sp_delta))
        out["default04"][arm] = {**row(d04, e_), "far_wide": {k: mean(r["far"] for r in rs) for k, rs in wide.items()},
                                 "worst_wide": {k: max(r["far"] for r in rs) for k, rs in wide.items()}}
    print("   " + json.dumps({a: [round(100 * v["far"], 2), round(100 * (v["frr"] or 0), 2), round(v["bpa_loop"] or 0),
                                  {k: round(100 * x, 2) for k, x in v["far_wide"].items()}]
                              for a, v in out["default04"].items() if a != "delta"}))

    print("13. deflated-eps accept wallet under lag-1 dependence")
    out["deflated"] = {}
    for pm in (0.25, 0.5):
        rule = partial(seq_accept, eps=0.05 * (1 - pm))
        d = {"far": {str(phi): correlated(rule, CORR_P, 15, phi, 2000, 51) for phi in PHIS}}
        for arm, e_, reps in (("natural", natural, replay.REPS), ("realflaky", flaky, replay.REPS)):
            rs = evaluate_rules(e_, lambda ep: {"d": partial(seq_accept, eps=eps_of(ep) * (1 - pm))}, reps=reps)["d"]
            d[arm] = row(rs, e_)
        out["deflated"][str(pm)] = d
    print("   " + json.dumps({pm: [{k: round(100 * v, 1) for k, v in d["far"].items()},
                                  [round(100 * (d[a]["frr"] or 0), 1) for a in ("natural", "realflaky")],
                                  [round(d[a]["blocks_per_accept"] or 0) for a in ("natural", "realflaky")]]
                              for pm, d in out["deflated"].items()}))

    ft = Path("results/flaky_truth.json")
    if ft.exists():
        print("14. yamicache relabelled with the re-measured blocks")
        truth = json.loads(ft.read_text())
        yam_eps = [ep for ep in flaky if ep["name"].startswith("yamicache@")]
        pooled_y = firmer(yam_eps, truth, pooled=True)
        rs = evaluate_rules(pooled_y, lambda ep: {"SeqAccept": partial(seq_accept, eps=eps_of(ep))})["SeqAccept"]
        rate = {f: v["truth200"] for t in truth.values() for f, v in t.items()}
        firm = {f: mean(v["extra"]) for t in truth.values() for f, v in t.items()}
        out["yamicache_firm"] = {"far": mean(r["far"] for r in rs), "worst": max(r["far"] for r in rs),
                                 "accepted": {f: {"truth200": rate.get(f), "extra": firm.get(f)} for f in out["yamicache"]["ids"]},
                                 "eps": eps_of(yam_eps[0])}
        print("   " + json.dumps(out["yamicache_firm"]))
    path.write_text(json.dumps(out, indent=1, default=str))
    print("saved results/deploy_checks.json")

