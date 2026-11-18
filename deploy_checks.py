"""Deployment checks (EASE round 9, offline, no LLM or Docker). Writes results/deploy_checks.json.

1. Refund share under dependent blocks: SeqAccept with a share e in {0, 0.5, 1} of the maximal refund, and with
   gamma = 1/4, on 1 and 15 broken candidates at p = 0.06 whose blocks follow a Markov chain with lag-1 correlation
   phi in {0, 0.2, 0.5} (1,000 runs, seed 51, as extra_analyses.py step 6); and the same variants' cost on natural,
   near-eps and real-flaky loops. Every variant is covered by Theorem 4.2.
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

  uv run python deploy_checks.py
"""
import csv
import json
from functools import partial
from pathlib import Path
from statistics import mean, median

import replay
import sys
from caps_adversary import ambiguous_labels
from extra_analyses import correlated, evaluate_rules, first_green, margin_labels
from intervals import cp_lower
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


def run15(out, natural, near, flaky, arms):
    print("1. refund share and gamma under dependent blocks")
    out["dependence"] = {f"{phi}-{n}": {k: correlated(r, CORR_P, n, phi, 1000, 51) for k, r in variants().items()}
                         for phi in PHIS for n in NS}
    out["dependence_cost"] = {arm: {k: row(rs, e_)["blocks_per_accept"]
                                    for k, rs in evaluate_rules(e_, lambda ep: variants(eps_of(ep)), reps=reps).items()}
                              for arm, e_, reps in arms}
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


if __name__ == "__main__":
    episodes = load("episodes", lambda p: not replay.excluded(p))
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = load("episodes_flaky", lambda p: not replay.excluded(p))
    near = load("episodes_neareps")
    arms = (("natural", natural, replay.REPS), ("neareps", near, replay.NEAR_REPS), ("realflaky", flaky, replay.REPS))
    path = Path("results/deploy_checks.json")
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
    print("   " + json.dumps(out["yamicache"]))

    print("8. near-eps false acceptances per distinct candidate (scored on all blocks)")
    pooled = firmer(near, json.loads(Path("results/neareps_truth.json").read_text()), pooled=True)
    out["near_candidates"] = {}
    for name in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "SeqAccept"):
        got = accepted(lambda ep, n=name: all_rules(eps=eps_of(ep))[n], pooled, replay.NEAR_REPS)
        per = {}
        for ep, g in zip(pooled, got):
            labs = replay.labels(ep)
            for j in g:
                if j is not None and labs[j] == "broken":
                    key = (ep["name"].split("-")[0], ep["stream"][j])
                    per[key] = per.get(key, 0) + 1
        out["near_candidates"][name] = {"distinct": len(per), "false": sum(per.values()),
                                       "top_share": max(per.values()) / sum(per.values()) if per else 0.0}
    print("   " + json.dumps(out["near_candidates"]))

    print("9. discarding on constructed near-eps streams")
    out["discarding"] = {f"{n}-{p}": experiment(discard_spending, [p] * n, reps=2000, seed=41)["far"]
                         for n in (15, 50) for p in (0.051, 0.06)}
    print("   " + json.dumps(out["discarding"]))
    path.write_text(json.dumps(out, indent=1, default=str))
    print("saved results/deploy_checks.json")

