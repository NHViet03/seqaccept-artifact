"""Near-eps arm and cost checks (offline, no LLM or Docker). Writes results/near_eps_checks.json.

1. Near-eps arm, every comparator, at the shared defaults and at rho = 20, T = 800: FAR, worst loop, loops above alpha,
   FRR, blocks per loop and per accept, per accept with the task's p0 calibration, minutes per accept; per task.
2. Near-eps labels: distinct near-eps Broken candidates, how many have more than a 5% posterior chance of p <= eps,
   FAR with those set Marginal, FAR scored against pool rates, and FAR with truth and pool blocks swapped.
3. A block cap of 10,000 for B4, B5 and SeqAccept on every arm, beside the derived cap.
4. Refund break-even: the wallet kept at the end of a candidate (min(E, 1)) on each arm, and the stream position from
   which SeqAccept's slice falls below harmonic spending's for that kept share.
5. Real-flaky arm with eps redrawn per replay from a bootstrap of the task's 400 calibration blocks, scored against
   labels at the recorded p0 + 0.05.
6. Cost model with a fixed start-up per run.
7. Firmer labels (only once neareps_truth.py has run): the near-eps arm with 2,000 extra blocks per candidate within
   3 points of eps; label changes and candidates still uncertain. Every comparator's FAR, worst loop, loops and tasks
   above alpha (also by a one-sided 95% Clopper-Pearson lower bound), scored two ways: truth_only (labels from
   200 + extra truth blocks, pools unchanged) and pooled (all of a candidate's blocks are its pool and its label).

  uv run python near_eps_checks.py
"""
import json
import random
import sys
from functools import partial
from pathlib import Path
from statistics import mean

import replay
from caps_adversary import ambiguous_labels
from extra_analyses import evaluate_rules, pool_labels
from intervals import cp_lower
from rules import (all_rules, bonferroni, fixed_n, fixed_spending, harmonic, naive_cs, peeking_cp, seq_accept)

ALPHA, GAMMA, BIG_CAP = 0.05, 0.5, 10_000
eps_of = lambda ep: ep["p0"] + replay.MARGIN
task_of = lambda ep: ep["name"].split("-")[0]
load = lambda d, keep=lambda p: True: [replay.load_episode(p) for p in sorted(Path(d).glob("*.json")) if keep(p)]
MAIN = ("B1 stop at first pass", "B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "B6 SeqAccept, no refund")


def near_rules(ep):
    e, r = eps_of(ep), all_rules(eps=eps_of(ep))
    old = dict(reject_at=20, max_blocks=800)
    return {**{n: r[n] for n in MAIN}, "Harmonic spending": partial(fixed_spending, slices=harmonic(), eps=e),
            "Bonferroni-100 spending": partial(fixed_spending, slices=bonferroni(100), eps=e),
            "Fixed-n Bonferroni-15 (1n0)": partial(fixed_n, slices=bonferroni(15), eps=e, scale=1),
            "SeqAccept": r["SeqAccept"],
            "old B4": partial(peeking_cp, eps=e, max_blocks=800), "old B5": partial(naive_cs, eps=e, **old),
            "old B6": partial(seq_accept, eps=e, refund=False, **old), "old SeqAccept": partial(seq_accept, eps=e, **old)}


def calib(eps_list):
    """Per loop: (calibration blocks, calibration seconds), the task's p0 blocks shared by its loops."""
    n = {}
    for ep in eps_list:
        n[task_of(ep)] = n.get(task_of(ep), 0) + 1
    out = []
    for ep in eps_list:
        t = json.loads(Path(f"tasks/{task_of(ep)}.json").read_text())
        secs = mean(s for fx in ep["fixes"].values() for s in fx.get("seconds", [])) if any(fx.get("seconds") for fx in ep["fixes"].values()) else 0
        out.append((t["p0_blocks"] / n[task_of(ep)], t["p0_blocks"] / n[task_of(ep)] * secs))
    return out


def row(rs, eps_list, reps):
    s = replay.summarize(rs, eps_list)
    acc = sum(r["accepts"] for r in rs)
    cb = calib(eps_list)
    ks = [round(r["far"] * r["reps"]) for r in rs]
    return {"far": s["far"], "frr": s["frr"], "blocks_per_loop": sum(r["blocks"] for r in rs) / sum(r["reps"] for r in rs),
            "blocks_per_accept": s["blocks_per_accept"],
            "bpa_calib": (sum(r["blocks"] for r in rs) + sum(c[0] * r["reps"] for c, r in zip(cb, rs))) / acc if acc else None,
            "min_per_accept": (sum(r["seconds"] for r in rs) + sum(c[1] * r["reps"] for c, r in zip(cb, rs))) / acc / 60 if acc else None,
            "worst_loop": max(ks) / reps, "worst_loop_upper": replay.cp_upper(max(ks), reps),
            "loops_above_alpha": sum(r["far"] > ALPHA for r in rs), "loops": len(rs),
            "per_task": {t: mean(r["far"] for r, ep in zip(rs, eps_list) if task_of(ep) == t) for t in sorted({task_of(e) for e in eps_list})}}


def swapped(ep):
    return {**ep, "fixes": {f: {**v, "truth": v["pool"], "pool": v["truth"]} for f, v in ep["fixes"].items()}}


def kept_share(eps_list, reps=20):
    """Mean min(E, 1) at the end of every candidate that SeqAccept ends without acceptance, over replays."""
    kept = []
    for i, ep in enumerate(eps_list):
        rng = random.Random(replay.episode_seed(ep, i))
        pools = [ep["fixes"][f]["pool"] for f in ep["stream"]]
        for _ in range(reps):
            last = {"E": 1.0, "a": 0.0}

            def trace(pool, a, E):
                if a == 0.0 and last["a"] > 0:  # a candidate just ended without acceptance
                    kept.append(min(last["E"], 1.0))
                last.update(E=E, a=a)
            seq_accept(len(pools), lambda j: rng.choice(pools[j]), eps=eps_of(ep), trace=trace)
    return mean(kept), len(kept)


def crossing(kept, gamma=GAMMA, alpha=ALPHA):
    """First position j >= 2 at which SeqAccept's slice gamma alpha q^(j-1), q = 1 - gamma (1 - kept), falls below
    harmonic spending's alpha / (j (j + 1)), when every earlier candidate ended keeping `kept` of its slice."""
    q = 1 - gamma * (1 - kept)
    for j in range(2, 10 ** 6):
        if gamma * alpha * q ** (j - 1) < alpha / (j * (j + 1)):
            return j
    return None


def redrawn_eps(flaky, reps=200, draws=20):
    """Replays with eps = p0* + 0.05, p0* the failure share of a bootstrap of the task's calibration blocks, redrawn
    every reps/draws replays; labels stay at the recorded p0 + 0.05."""
    names = ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "SeqAccept")
    out = {n: [] for n in names}
    for i, ep in enumerate(flaky):
        cal = [json.loads(l)["failed"] for l in Path(f"runs/blocks/{task_of(ep)}-p0.jsonl").read_text().splitlines()]
        rng = random.Random(replay.episode_seed(ep, i))
        res = {n: [] for n in names}
        for d in range(draws):
            p0s = mean(rng.choice(cal) for _ in cal)
            r = all_rules(eps=p0s + replay.MARGIN)
            for n in names:
                res[n].append(replay.replay(r[n], ep, reps // draws, seed=rng.randrange(2 ** 31)))
        for n in names:
            out[n].append(mean(x["far"] for x in res[n]))
    by_task = {}
    for ep, v in zip(flaky, out["SeqAccept"]):
        by_task.setdefault(task_of(ep), []).append(v)
    return {"far": {n: mean(v) for n, v in out.items()}, "worst_task_sa": max((mean(v), t) for t, v in by_task.items()),
            "reps": reps, "draws": draws}


def overhead_model(natural, starts=(0, 100, 1000)):
    """Run time = s0 + c * tests. Within each natural task a full-suite run and a block differ only in their tests; the
    pooled difference per extra test (full - block seconds over suite - block tests) is reported, and where it is not
    positive the per-test cost cannot be told from start-up on these suites. The regime map's SeqAccept / full-suite
    B1 ratio of run time per accept is then given for a start-up equal to running k tests (k = s0 / c)."""
    dt, dn, tasks = 0.0, 0, 0
    for task in sorted({task_of(ep) for ep in natural}):
        t = json.loads(Path(f"tasks/{task}.json").read_text())
        eps_ = [ep for ep in natural if task_of(ep) == task]
        blk = [min(x, replay.CAP_SECONDS) for ep in eps_ for f in ep["fixes"].values() for x in f.get("seconds", [])]
        full = [x for ep in eps_ for f in ep["fixes"] for p in [Path(f"fullsuite/{task}__{f}.json")] if p.exists()
                for x in json.loads(p.read_text())["seconds"]]
        if blk and full:
            dt += mean(full) - mean(blk)
            dn += len(t["test_ids"]) - t["block_size"]
            tasks += 1
    be = json.loads(Path("results/caps_adversary.json").read_text())["breakeven"]
    reg = json.loads(Path("results/sensitivity.json").read_text())["regime"]
    ratio = {str(k): {key: be["B1"]["ratio"] * (k + v["block"]) / (k + int(key.split("-")[0])) for key, v in reg.items()}
             for k in starts}
    return {"seconds_per_extra_test": dt / dn, "tasks": tasks, "block_s": be["block_s"], "full_s": be["full_s"],
            "ratio_B1": ratio}


def firmer(eps_list, truth, pooled=False):
    """The episodes with each re-measured candidate's truth extended by its extra blocks, pools unchanged. pooled=True:
    every candidate's blocks (truth, pool and extra) are both its pool and its truth, so each label is exact for the
    replay's draws (a clearly broken candidate's 40 blocks serve as both already, so they count once)."""
    def fix(ep, f, v):
        extra = truth.get(task_of(ep), {}).get(f, {}).get("extra", [])
        if not pooled:
            return {**v, "truth": v["truth"] + extra} if extra else v
        blocks = (v["truth"] if len(v["seeds"]) == len(v["truth"]) else v["truth"] + v["pool"]) + extra
        return {**v, "truth": blocks, "pool": blocks}
    return [{**ep, "fixes": {f: fix(ep, f, v) for f, v in ep["fixes"].items()}} for ep in eps_list]


def save(out):
    Path("results/near_eps_checks.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    episodes = load("episodes", lambda p: not replay.excluded(p))
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = load("episodes_flaky", lambda p: not replay.excluded(p))
    near = load("episodes_neareps")
    R = replay.NEAR_REPS
    out = json.loads(Path("results/near_eps_checks.json").read_text()) if "--resume" in sys.argv and Path("results/near_eps_checks.json").exists() else {}

    print("1. near-eps arm, every comparator")
    if "neareps" not in out:
        res = evaluate_rules(near, near_rules, reps=R)
        out["neareps"] = {n: row(rs, near, R) for n, rs in res.items()}
    print("   " + json.dumps({n[:12]: [round(100 * v["far"], 2), round(100 * v["worst_loop"], 1), v["loops_above_alpha"],
                                    round(v["blocks_per_loop"])] for n, v in out["neareps"].items()}))

    save(out)
    print("2. near-eps labels")
    rate = lambda ep, f: mean(ep["fixes"][f]["truth"])
    near_bad, risky = set(), set()
    for ep, amb in zip(near, map(ambiguous_labels, near)):
        for f, lab, a in zip(ep["stream"], replay.labels(ep), amb):
            if lab == "broken" and rate(ep, f) <= eps_of(ep) + 0.03:
                near_bad.add((task_of(ep), f))
                if a == "marginal":
                    risky.add((task_of(ep), f))
    four = lambda ep: {n: near_rules(ep)[n] for n in ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence",
                                                      "B6 SeqAccept, no refund", "SeqAccept")}
    if "neareps_labels" not in out:
        out["neareps_labels"] = {"near_distinct": len(near_bad), "near_risky": len(risky), "far": {}}
    for tag, kw, eps_ in (("unamb", dict(labels_of=ambiguous_labels), near), ("pool", dict(labels_of=pool_labels), near),
                          ("swap", {}, [swapped(ep) for ep in near])):
        if tag in out["neareps_labels"]["far"]:
            continue
        rr = evaluate_rules(eps_, four, reps=R, **kw)
        out["neareps_labels"]["far"][tag] = {n: {"far": mean(r["far"] for r in rs),
                                                 "worst_loop": max(r["far"] for r in rs),
                                                 "loops_above_alpha": sum(r["far"] > ALPHA for r in rs)} for n, rs in rr.items()}
    print("   " + json.dumps(out["neareps_labels"], default=str)[:600])

    save(out)
    print("3. block cap 10,000 for B4, B5 and SeqAccept (20 replays per loop on every arm)")
    big = "big_cap" not in out
    out.setdefault("big_cap", {})
    for arm, eps_, reps in (("natural", natural, replay.REPS), ("realflaky", flaky, replay.REPS), ("neareps", near, replay.REPS)) if big else ():
        rr = evaluate_rules(eps_, lambda ep: {"B4": partial(peeking_cp, eps=eps_of(ep), max_blocks=BIG_CAP),
                                              "B5": partial(naive_cs, eps=eps_of(ep), max_blocks=BIG_CAP),
                                              "SeqAccept": partial(seq_accept, eps=eps_of(ep), max_blocks=BIG_CAP)}, reps=reps)
        out["big_cap"][arm] = {n: {k: row(rs, eps_, reps)[k] for k in ("far", "frr", "blocks_per_loop", "blocks_per_accept")}
                               for n, rs in rr.items()}
        print(f"   {arm}: " + json.dumps({n: [round(100 * v['far'], 2), round(100 * (v['frr'] or 0), 2), round(v['blocks_per_loop'])]
                                          for n, v in out["big_cap"][arm].items()}))

    save(out)
    print("4. refund break-even against harmonic spending")
    if "refund_crossing" not in out:
        out["refund_crossing"] = {"by_kept": {str(k): crossing(k) for k in (0.0, 0.5, 0.9, 0.99)}}
        for arm, eps_ in (("natural", natural), ("neareps", near), ("realflaky", flaky)):
            k, n = kept_share(eps_)
            out["refund_crossing"][arm] = {"kept": k, "ended": n, "crossing": crossing(k)}
    print("   " + json.dumps(out["refund_crossing"]))

    save(out)
    print("5. real-flaky arm with eps redrawn from the calibration blocks")
    if "redrawn_eps" not in out:
        out["redrawn_eps"] = redrawn_eps(flaky)
    print("   " + json.dumps(out["redrawn_eps"], default=str))

    print("6. cost model with a fixed start-up per run")
    if "overhead" not in out:
        out["overhead"] = overhead_model(natural)
    print("   " + json.dumps({k: v for k, v in out["overhead"].items() if k != "ratio_B1"})
          + " " + json.dumps({s: {k: round(v, 2) for k, v in d.items() if k.split("-")[0] != "640"} for s, d in out["overhead"]["ratio_B1"].items()}))

    save(out)
    tp = Path("results/neareps_truth.json")
    if tp.exists():
        print("7. near-eps arm relabelled with the extra truth blocks")
        truth = json.loads(tp.read_text())
        firm = firmer(near, truth)
        old = {(task_of(ep), f): l for ep in near for f, l in zip(ep["stream"], replay.labels(ep))}
        new = {(task_of(ep), f): l for ep in firm for f, l in zip(ep["stream"], replay.labels(ep))}
        amb = {(task_of(ep), f): l for ep in firm for f, l in zip(ep["stream"], ambiguous_labels(ep))}
        moved = [k for k in new if k[1] in truth.get(k[0], {})]
        d = {"remeasured": len(moved), "extra_blocks": sorted({len(v["extra"]) for t in truth.values() for v in t.values()}),
             "changes": {f"{a}->{b}": sum(old[k] == a and new[k] == b for k in moved)
                         for a in ("correct", "marginal", "broken") for b in ("correct", "marginal", "broken")},
             "still_uncertain": sum(new[k] != amb[k] for k in moved)}
        pool = firmer(near, truth, pooled=True)
        d["pooled_labels"] = {l: sum(x == l for ep in pool for x in replay.labels(ep)) for l in ("correct", "marginal", "broken")}
        for tag, world in (("truth_only", firm), ("pooled", pool)):  # pooled: labels exact for the draws; the main scoring
            d[tag] = {}
            for n, rs in evaluate_rules(world, near_rules, reps=R).items():
                r_ = row(rs, world, R)
                ks, by = [round(x["far"] * x["reps"]) for x in rs], {}
                for k, ep in zip(ks, world):
                    by.setdefault(task_of(ep), []).append(k)
                r_["loops_lower_above"] = sum(cp_lower(k, R) > ALPHA for k in ks)
                r_["tasks_lower"] = {t: cp_lower(sum(v), R * len(v)) for t, v in by.items()}
                r_["tasks_lower_above"] = sum(v > ALPHA for v in r_["tasks_lower"].values())
                d[tag][n] = r_
            print(f"   {tag}: " + json.dumps({n[:12]: [round(100 * v["far"], 2), round(100 * v["worst_loop"], 1), v["loops_above_alpha"],
                                                      v["loops_lower_above"], v["tasks_lower_above"], round(v["blocks_per_loop"])]
                                                  for n, v in d[tag].items()}))
        out["firmer"] = d
        print("   " + json.dumps({k: v for k, v in d.items() if k not in ("truth_only", "pooled")}))
        save(out)
    print("saved results/near_eps_checks.json")
