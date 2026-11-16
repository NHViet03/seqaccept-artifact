"""EASE review round 1 follow-ups (offline: no LLM, no Docker). Writes results/ease2.json.

  uv run python ease_round2.py

1. SeqAccept (maximal refund at rejection and at the cap) against no refund at the cap, the earlier default
   (capped refund, none at the cap) and a one-block prefilter, on the recorded loops and the real-flaky arm.
2. Constructed worst case at p = 0.10 (eps + 0.05), the margin a practitioner cares about.
3. Per-bug FAR on natural loops and the leave-one-repository-out range.
4. Full-suite FAR by bug kind (natural loops).
5. FAR per flakiness mode (natural, Bernoulli, Markov) for every main rule.
6. Valid competitors (harmonic, fixed-n, discarding, Rerun-k) and the main rules on natural loops: FAR, FRR,
   blocks per accept, seconds per accept.
7. Standard error of p0 for every task.
"""
import json
from functools import partial
from pathlib import Path

import replay
from extra_analyses import extra_rules, fullsuite_episodes, replay_rules_fixed
from rules import all_rules, seq_accept
from sim import experiment

VARIANTS = {"SeqAccept": {}, "no refund at cap": {"setaside_refund": False}, "prefilter 1": {"prefilter": 1},
            "earlier default": {"refund": True, "setaside_refund": False}}  # capped refund, none at the cap


def bug_of(ep):
    return ep["name"].split("-")[0].split("@")[0]


def run(make_rule, episodes, reps=replay.REPS):
    return [replay.replay(make_rule(ep["p0"] + replay.MARGIN), ep, reps, seed=replay.episode_seed(ep, i))
            for i, ep in enumerate(episodes)]


def row(results, episodes):
    s = replay.summarize(results, episodes)
    per_loop = sum(r["blocks"] for r in results) / sum(r["reps"] for r in results)
    return {"far": s["far"], "frr": s["frr"], "blocks_per_accept": s["blocks_per_accept"], "blocks_per_loop": per_loop,
            "seconds_per_accept": (sum(r["seconds"] for r in results) / sum(r["accepts"] for r in results)
                                   if sum(r["accepts"] for r in results) else None)}


if __name__ == "__main__":
    paths = sorted(Path("episodes").glob("*.json"))
    episodes = [replay.load_episode(p) for p in paths if not replay.excluded(p)]
    episodes = [ep for ep in episodes if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS]
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    flaky = [replay.load_episode(p) for p in sorted(Path("episodes_flaky").glob("*.json"))]
    out = {}

    print("1. set-aside refund and prefilter")
    out["variants"] = {}
    for name, kw in VARIANTS.items():
        mk = lambda eps, kw=kw: partial(seq_accept, eps=eps, **kw)
        out["variants"][name] = {"natural": row(run(mk, natural), natural), "realflaky": row(run(mk, flaky), flaky)}
        hi = [ep for ep in flaky if ep["p0"] >= 0.2]
        out["variants"][name]["realflaky_high"] = row(run(mk, hi), hi)
        print(f"   {name}: {out['variants'][name]}")

    print("2. worst case at p = 0.10")
    out["worst010"] = {}
    for n in (1, 15, 50):
        reps = 1000 if n < 50 else 400
        out["worst010"][n] = {name: experiment(rule, [0.10] * n, reps=reps, seed=31)["far"]
                              for name, rule in all_rules(eps=0.05).items() if name not in replay_rules_fixed}
    print(f"   {out['worst010']}")

    print("3. per-bug FAR and leave-one-repository-out (natural)")
    per = {Path(e["file"]).stem: e for e in json.loads(Path("results/results.json").read_text())["per_episode"]}
    names = list(all_rules())
    bugs = sorted({bug_of(ep) for ep in natural})
    out["per_bug"] = {b: {n: sum(per[ep["name"]]["rules"][n]["far"] for ep in natural if bug_of(ep) == b) /
                          sum(1 for ep in natural if bug_of(ep) == b) for n in names} for b in bugs}
    repos = sorted({ep.get("repo") for ep in natural})
    loro = {n: [] for n in names}
    for rp in repos:
        keep = [ep for ep in natural if ep.get("repo") != rp]
        for n in names:
            loro[n].append(sum(per[ep["name"]]["rules"][n]["far"] for ep in keep) / len(keep))
    out["leave_one_repo_out"] = {n: [min(v), max(v)] for n, v in loro.items()}
    out["bug_kind"] = {b: json.loads(Path(f"tasks/{b}.json").read_text()).get("bug_kind") for b in bugs}
    print(f"   LORO {out['leave_one_repo_out']}")

    print("4. full suite by bug kind")
    full = [ep for ep in (fullsuite_episodes(episodes) or []) if ep.get("flakiness", "natural") == "natural"]
    out["fullsuite_kind"] = {}
    for kind in ("ordinary", "order"):
        mine = [ep for ep in full if out["bug_kind"].get(bug_of(ep)) == kind]
        res = {n: [replay.replay(all_rules()[n], ep, replay.REPS, seed=replay.episode_seed(ep, i)) for i, ep in enumerate(mine)]
               for n in replay_rules_fixed}
        out["fullsuite_kind"][kind] = {"loops": len(mine), **{n: replay.summarize(rs, mine)["far"] for n, rs in res.items()}}
        sampled = [ep for ep in natural if out["bug_kind"].get(bug_of(ep)) == kind]
        out["fullsuite_kind"][kind]["sampled_B1"] = sum(per[ep["name"]]["rules"]["B1 stop at first pass"]["far"] for ep in sampled) / len(sampled)
    print(f"   {out['fullsuite_kind']}")

    print("5. FAR per flakiness mode")
    out["by_mode"] = {}
    for fl in replay.MAIN_FLAKINESS:
        mine = [ep for ep in episodes if ep.get("flakiness", "natural") == fl]
        out["by_mode"][fl] = {"episodes": len(mine), **{n: sum(per[ep["name"]]["rules"][n]["far"] for ep in mine) / len(mine)
                                                         for n in names}}
    print(f"   {out['by_mode']}")

    print("6. valid competitors and main rules on natural loops")
    oc = json.loads(Path("results/operating_curve.json").read_text())
    k_match = round(next(r["blocks"] for r in oc["single"] if r["p"] == 0.0))
    comp = {}
    for n in names:
        rs = [per[ep["name"]]["rules"][n] for ep in natural]
        comp[n] = row(rs, natural)
    for n in ("Harmonic spending", "Discard spending", "Fixed-n Bonferroni-15 (1n0)", "Fixed-n Bonferroni-15 (2n0)",
              "Fixed-n Bonferroni-15 (4n0)", f"Rerun-{k_match} (all green)"):
        comp[n] = row(run(lambda eps, n=n: extra_rules(eps, k_match)[n], natural), natural)
    out["natural_rules"] = comp
    print(f"   {json.dumps({k: round(v['blocks_per_accept'] or 0, 1) for k, v in comp.items()})}")

    print("7. p0 standard errors")
    tasks = {bug_of(ep): ep["p0"] for ep in episodes}
    out["p0_se"] = {"natural_max": max((p * (1 - p) / 400) ** 0.5 for b, p in
                                       ((bug_of(ep), ep["p0"]) for ep in natural)),
                    "injected_max": max((ep["p0"] * (1 - ep["p0"]) / 400) ** 0.5 for ep in episodes
                                        if ep.get("flakiness", "natural") != "natural")}
    print(f"   {out['p0_se']}")

    Path("results/ease2.json").write_text(json.dumps(out, indent=1, default=str))
    print("saved results/ease2.json")
