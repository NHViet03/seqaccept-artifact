"""EASE review round 2 follow-ups (offline: no LLM, no Docker). Writes results/intervals.json.

  uv run python intervals.py

1. Bug- and repository-clustered intervals for full-suite FAR (natural loops, B1-B3).
2. Bug- and repository-clustered intervals on the boundary-stress arm (m = 0.94) for B4, B5 and SeqAccept.
3. Dependence restricted to near-eps candidates (pool rate in (p0, 0.5]): runs test and lag-1 autocorrelation, and an
   order-preserving replay of the real-flaky arm (circular reading of each pool in execution order).
4. Real-flaky arm with eps = L + 0.05, L the one-sided 95% Clopper-Pearson lower bound on p0 (Corollary 4.3).
5. Cost with p0 calibration included, and full-suite minutes per accept.
"""
import json
import math
import random
from pathlib import Path
from statistics import mean

import replay
from comparators import bug_of
from extra_analyses import fullsuite_episodes, lag1_corr, runs_test
from rules import all_rules
from sim import run_episodes

MAIN3 = ("B4 peeking Clopper-Pearson", "B5 naive confidence sequence", "SeqAccept")
FIXED = ("B1 stop at first pass", "B2 rerun-3 all green", "B3 majority of 5")


def cp_lower(k, n, alpha=0.05):
    """One-sided (1 - alpha) Clopper-Pearson lower bound for k events in n trials (0 for k = 0)."""
    if k == 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(60):  # Pr[X >= k | p] rises with p; find p with Pr[X >= k] = alpha
        mid = (lo + hi) / 2
        sf = 1 - sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k))
        lo, hi = (lo, mid) if sf > alpha else (mid, hi)
    return lo


def ci(values, clusters):
    return list(replay.cluster_ci(values, clusters))


def summary(results, episodes):
    fars = [r["far"] for r in results]
    return {"far": mean(fars), "far_ci_bug": ci(fars, [bug_of(ep) for ep in episodes]),
            "far_ci_repo": ci(fars, [ep.get("repo", "?") for ep in episodes]),
            "frr": replay.summarize(results, episodes)["frr"],
            "blocks_per_accept": replay.summarize(results, episodes)["blocks_per_accept"]}


def run(rule_of, episodes, reps=replay.REPS):
    return [replay.replay(rule_of(ep), ep, reps, seed=replay.episode_seed(ep, i)) for i, ep in enumerate(episodes)]


def ordered_replay(rule, episode, reps, seed=0):
    """Replay that keeps each pool's execution order: every candidate starts at a random offset and then reads its
    pool block after block, wrapping around (a circular block bootstrap with one block of the whole pool)."""
    fixes = episode["fixes"]
    pools = [fixes[f]["pool"] for f in episode["stream"]]
    rng = random.Random(seed)

    def make_block():
        pos = [rng.randrange(len(p)) for p in pools]

        def draw(j):
            y = pools[j][pos[j] % len(pools[j])]
            pos[j] += 1
            return y
        return draw

    return run_episodes(rule, replay.labels(episode), make_block, reps)


def near_eps_dependence(episodes):
    tested = rejected = 0
    lags, tasks = [], []
    for ep in episodes:
        for f in ep["fixes"].values():
            if not f["pool"] or len(f["truth"]) + len(f["pool"]) != len(f["seeds"]):
                continue  # shortcut candidates share probe blocks; they fail far above eps anyway
            rate = mean(f["pool"])
            if not ep["p0"] < rate <= 0.5:
                continue
            seq = f["truth"] + f["pool"]
            r = runs_test(seq)
            if r is None:
                continue
            tested += 1
            rejected += r
            lags.append(lag1_corr(seq))
            tasks.append(ep["name"].split("-")[0])
    if not tested:
        return {"tested": 0}
    return {"tested": tested, "rejected": rejected, "rejected_upper": replay.cp_upper(rejected, tested),
            "lag1_mean": mean(lags), "lag1_ci_task": ci(lags, tasks), "lag1_abs_over_02": sum(abs(x) > 0.2 for x in lags),
            "tasks": len(set(tasks))}


if __name__ == "__main__":
    paths = sorted(Path("episodes").glob("*.json"))
    episodes = [replay.load_episode(p) for p in paths if not replay.excluded(p)]
    episodes = [ep for ep in episodes if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS]
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = [replay.load_episode(p) for p in sorted(Path("episodes_flaky").glob("*.json"))]
    out = {}

    print("1. full-suite FAR with clustered intervals (natural)")
    full = [ep for ep in (fullsuite_episodes(episodes) or []) if ep.get("flakiness", "natural") == "natural"]
    rules = all_rules()
    out["fullsuite"] = {"loops": len(full), **{n: summary(run(lambda ep, n=n: rules[n], full), full) for n in FIXED}}
    print(f"   {json.dumps({n: [round(100 * v['far'], 2), [round(100 * x, 2) for x in v['far_ci_bug']]] for n, v in out['fullsuite'].items() if n != 'loops'})}")

    print("2. boundary stress m = 0.94 with clustered intervals")
    masked = [replay.masked(ep, 0.94) for ep in natural]
    out["stress094"] = {"episodes": len(masked), "reps": replay.NEAR_REPS,
                        **{n: summary(run(lambda ep, n=n: all_rules(eps=ep["p0"] + replay.MARGIN)[n], masked,
                                          replay.NEAR_REPS), masked) for n in MAIN3}}
    print(f"   {json.dumps({n: [round(100 * v['far'], 2), [round(100 * x, 2) for x in v['far_ci_bug']]] for n, v in out['stress094'].items() if n in MAIN3})}")

    print("3. dependence near eps; order-preserving replay of the real-flaky arm")
    out["near_eps_dependence"] = {"natural": near_eps_dependence(natural), "realflaky": near_eps_dependence(flaky)}
    print(f"   {out['near_eps_dependence']}")
    out["ordered_realflaky"] = {}
    for n in MAIN3:
        iid = run(lambda ep, n=n: all_rules(eps=ep["p0"] + replay.MARGIN)[n], flaky)
        ordd = [ordered_replay(all_rules(eps=ep["p0"] + replay.MARGIN)[n], ep, replay.REPS, seed=replay.episode_seed(ep, i))
                for i, ep in enumerate(flaky)]
        out["ordered_realflaky"][n] = {"iid": summary(iid, flaky), "ordered": summary(ordd, flaky)}
    print(f"   {json.dumps({n: [round(100 * v['iid']['far'], 2), round(100 * v['ordered']['far'], 2)] for n, v in out['ordered_realflaky'].items()})}")

    print("4. real-flaky arm with eps = L + 0.05")
    tasks = {}
    for ep in flaky:
        t = json.loads(Path(f"tasks/{ep['name'].split('-')[0]}.json").read_text())
        tasks[ep["name"]] = cp_lower(round(t["p0"] * t["p0_blocks"]), t["p0_blocks"])
    hi = [ep for ep in flaky if ep["p0"] >= 0.2]
    out["realflaky_lcb"] = {"max_gap": max(ep["p0"] - tasks[ep["name"]] for ep in flaky)}
    for n in MAIN3:
        rs = run(lambda ep, n=n: all_rules(eps=tasks[ep["name"]] + replay.MARGIN)[n], flaky)
        hrs = [r for r, ep in zip(rs, flaky) if ep["p0"] >= 0.2]
        out["realflaky_lcb"][n] = {**summary(rs, flaky), "frr_high": replay.summarize(hrs, hi)["frr"]}
    print(f"   {json.dumps({n: [round(100 * v['far'], 2), round(100 * v['frr'], 1)] for n, v in out['realflaky_lcb'].items() if n in MAIN3})}")

    print("5. cost with calibration; full-suite minutes per accept")
    per = {Path(e["file"]).stem: e for e in json.loads(Path("results/results.json").read_text())["per_episode"]}
    by_task = {}
    for ep in natural:
        by_task.setdefault(ep["name"].split("-")[0], []).append(ep)
    calib_blocks, calib_seconds = {}, {}
    for task, eps_ in by_task.items():
        t = json.loads(Path(f"tasks/{task}.json").read_text())
        secs = [min(s, replay.CAP_SECONDS) for ep in eps_ for f in ep["fixes"].values() for s in f.get("seconds", [])]
        for ep in eps_:  # the task's calibration is shared by its loops
            calib_blocks[ep["name"]] = t["p0_blocks"] / len(eps_)
            calib_seconds[ep["name"]] = calib_blocks[ep["name"]] * mean(secs)
    out["cost_calibrated"] = {}
    for n in list(all_rules()):
        rs = [per[ep["name"]]["rules"][n] for ep in natural]
        reps = sum(r["reps"] for r in rs)
        acc = sum(r["accepts"] for r in rs)
        eps_based = n not in FIXED
        cb = sum(calib_blocks[ep["name"]] * r["reps"] for ep, r in zip(natural, rs)) if eps_based else 0
        cs = sum(calib_seconds[ep["name"]] * r["reps"] for ep, r in zip(natural, rs)) if eps_based else 0
        out["cost_calibrated"][n] = {"blocks_per_accept": (sum(r["blocks"] for r in rs) + cb) / acc,
                                     "minutes_per_accept": (sum(r["seconds"] for r in rs) + cs) / acc / 60,
                                     "blocks_per_loop": (sum(r["blocks"] for r in rs) + cb) / reps}
    for n in FIXED:
        rs = run(lambda ep, n=n: rules[n], full)
        acc = sum(r["accepts"] for r in rs)
        out["cost_calibrated"][f"{n} (full suite)"] = {"runs_per_accept": sum(r["blocks"] for r in rs) / acc,
                                                       "minutes_per_accept": sum(r["seconds"] for r in rs) / acc / 60}
    print(f"   {json.dumps({n: round(v['minutes_per_accept'], 2) for n, v in out['cost_calibrated'].items()})}")

    Path("results/intervals.json").write_text(json.dumps(out, indent=1, default=str))
    print("saved results/intervals.json")
