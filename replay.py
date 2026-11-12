"""Replay acceptance rules on recorded episodes (StopSafe-SE style, fixed candidate streams, §9.4).

Episode JSON: {"p0": background failure rate of the original code,
               "stream": [fix_id, ...]            # candidates in the order the agent produced them
               "fixes": {fix_id: {"truth": [0/1, ...], "pool": [0/1, ...], "seeds": [...], "seconds": [...]}},
               "model", "repo", "repo_kind" (OD/NOD), "bug_kind" (order/ordinary), "image", "hardware": metadata}
`truth` blocks only grade the decision; rules draw their blocks from `pool`, never from `truth`.

Usage: uv run python replay.py [episodes_dir]
  -> Table 1 composition, Table 2 rows with cluster-bootstrap CIs, the primary endpoint, FAR splits,
     results/chart2_*.png, results/chart3_*.png and results/results.json (all numbers + per-episode outcomes, §14)
"""
import json
import math
import random
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from collect import log
from rules import all_rules, stop_on_first_pass
from sim import label, run_episodes

REPS = 20  # §9.4: 20 replications of every policy on every episode
MARGIN = 0.05  # eps = p0 + MARGIN (plan 2.3)
MARGINS = (0.03, 0.05, 0.10)  # plan day 15, Chart 3 (§10.5 sensitivity to eps)
PREFIXES = (1, 3, 5, 10, 15)  # plan day 15, Chart 2 (§10.1 FAR vs candidates tried)
LATE = 5  # paper Table 3 does not say what "late" is: we call a stream late when its first Correct fix is #6 or later
PRIMARY = ("B3 majority of 5", "SeqAccept", 0.03)  # §9.4: preregistered FAR contrast and its practical margin
SPLITS = ("flakiness", "bug_kind", "repo_kind", "model", "q_band")  # §8 natural vs injected; plan week 3; H2; §12 q
ALPHA = 0.05  # SeqAccept's nominal level, for the calibration column
CAP_SECONDS = 180  # D7/D9: a block recorded before the 180 s cap (up to 300 s) is charged as if killed at the cap
MASKS = (0.80, 0.90, 0.94)  # D14 boundary stress: a failed block reads as passed with this known probability;
# an always-failing patch then fails 20%, 10%, 6% of blocks (0.95 would put it at 5% = eps: Marginal, never Broken)
NEAR_REPS = 200  # D14 only: SeqAccept's FAR near eps is ~0.1-0.3%, too rare to see in 20 replications
MAIN_FLAKINESS = ("natural", "injected-bernoulli", "injected-markov")  # the main table; D17 sweep episodes are apart
LATE_K = (5, 10, 20, 40)  # D15: Broken fixes placed before the Correct one in built late streams
STYLES = [dict(ls=ls, marker=m, alpha=0.8) for ls, m in zip(("-", "--", "-.", ":", "--", "-.", "-"), "osD^Pxv")]  # equal lines stay visible


EXCLUDED_RUNS = {("freezegun", 1), ("freezegun", 2),  # D18: (task, run) left out for every model, so all models score the same episodes
                 ("kanren_b2", 1), ("kanren_b2", 2)}  # D23: grok-4.7's kanren_b2 run 1 lost to the end of the Azure credit


def excluded(path):
    """D18: True for the episode files of an excluded (task, run), injected and sweep variants included."""
    stem = Path(path).stem
    if "@b7" in stem:  # D21 withdrawn on 2026-11-06: the 20 recorded B7 streams and 7 episodes stay on disk, unused
        return True
    return (stem.split("-")[0].split("@")[0], int(stem.rsplit("-", 1)[1])) in EXCLUDED_RUNS


def load_episode(path):
    return {**json.loads(open(path).read()), "name": Path(path).stem}  # name: stable replay seed


def labels(episode):
    fixes, keep = episode["fixes"], 1 - episode.get("mask", 0.0)
    return [label(keep * sum(fixes[f]["truth"]) / len(fixes[f]["truth"]), episode["p0"]) for f in episode["stream"]]


def masked(episode, m):
    """D14: the episode as seen through an oracle that misses a failure with known probability m (a rarely triggered
    bug). Every failure rate, p0 included, scales by 1 - m; labels use the scaled truth rate, and replay hides each
    drawn failed pool block with probability m, fresh on every draw."""
    return {**episode, "p0": episode["p0"] * (1 - m), "mask": m, "flakiness": f"injected-mask-{m}"}


def late_streams(episodes, k, n=5, seed=0):
    """D15: Table 3 late streams built from recorded candidates. Per task (repo + flakiness, so one p0): n streams of k
    Broken fixes drawn with replacement from all its episodes, then one of its Correct fixes."""
    rng, tasks, out = random.Random(seed), {}, []
    for ep in episodes:
        tasks.setdefault((ep.get("repo"), ep.get("flakiness", "natural")), []).append(ep)
    for key in sorted(tasks, key=str):
        eps, fixes = tasks[key], {}
        for ep in eps:
            for f in ep["stream"]:
                fixes.setdefault(f, ep["fixes"][f])
        names = sorted(fixes)
        labs = dict(zip(names, labels({"p0": eps[0]["p0"], "fixes": fixes, "stream": names})))
        bad = [f for f in names if labs[f] == "broken"]
        good = [f for f in names if labs[f] == "correct"]
        if not bad or not good:
            continue
        for _ in range(n):
            stream = [rng.choice(bad) for _ in range(k)] + [rng.choice(good)]
            out.append({**eps[0], "stream": stream, "fixes": {f: fixes[f] for f in set(stream)}})
    return out


def replay(rule, episode, reps, seed=0):
    fixes = episode["fixes"]
    for fid in episode["stream"]:
        if not fixes[fid]["pool"]:
            raise ValueError(f"fix {fid} has no pool results to replay")
    rng = random.Random(seed)
    pools = [fixes[f]["pool"] for f in episode["stream"]]
    costs = [mean(min(s, CAP_SECONDS) for s in fixes[f]["seconds"]) if fixes[f].get("seconds") else 0.0
             for f in episode["stream"]]
    mask = episode.get("mask", 0.0)

    def draw(j):  # with mask 0 no extra random number is drawn, so unmasked replays are unchanged
        y = rng.choice(pools[j])
        return 0 if y and mask and rng.random() < mask else y

    return run_episodes(rule, labels(episode), lambda: draw, reps, cost=lambda j: costs[j])


def protected_view(ep):
    """B7 (§9.2, D21): the episode as seen by "run the hidden suite once per candidate": its pool is the hidden-suite
    runs, so B1's rule on this view accepts the first candidate whose hidden run passes."""
    return {**ep, "fixes": {f: {**v, "pool": v["hidden"], "seconds": v.get("hidden_seconds", v.get("seconds"))}
                            for f, v in ep["fixes"].items()}}


def episode_seed(ep, i):
    """An episode's replay seed comes from its file name, so adding or dropping other episodes leaves its draws
    unchanged. Built late streams have no name and use their position."""
    return zlib.crc32(ep["name"].encode()) if "name" in ep else i


def cp_upper(k, n, alpha=0.05):
    """One-sided (1 - alpha) Clopper-Pearson upper bound for k events in n trials (0 events: 1 - alpha^(1/n))."""
    if n == 0 or k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):  # bisection on Binom(n, p).cdf(k) = alpha, which falls as p rises
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > alpha else (lo, mid)
    return hi


def far_upper(results, episodes):
    """§9.4 zero-event reporting: the share of episodes (and of repositories) whose replays ever accepted a Broken
    candidate, over those with a Broken candidate, with its one-sided 95% upper bound."""
    pairs = [(r["far"] > 0, ep.get("repo", "?")) for r, ep in zip(results, episodes) if "broken" in labels(ep)]
    repos = {}
    for hit, repo in pairs:
        repos[repo] = repos.get(repo, False) or hit
    k, n, kr, nr = sum(h for h, _ in pairs), len(pairs), sum(repos.values()), len(repos)
    return {"episodes": [k, n, cp_upper(k, n)], "repos": [kr, nr, cp_upper(kr, nr)]}


def evaluate(episodes, margin=MARGIN, reps=REPS, seed=0):
    """Every rule on every episode, SeqAccept at eps = p0 + margin. Returns {rule: [result per episode]}."""
    out = {}
    for i, ep in enumerate(episodes):
        for name, rule in all_rules(eps=ep["p0"] + margin).items():
            out.setdefault(name, []).append(replay(rule, ep, reps, seed=seed + episode_seed(ep, i)))
    return out


def cluster_ci(values, clusters, n_boot=2000, seed=0):
    """95% CI of the mean of per-episode values, resampling whole repositories (§9.4 cluster bootstrap)."""
    groups = {}
    for v, c in zip(values, clusters):
        groups.setdefault(c, []).append(v)
    keys, rng = list(groups), random.Random(seed)
    stats = sorted(mean(v for k in (rng.choice(keys) for _ in keys) for v in groups[k]) for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def summarize(results, episodes):
    """Pool one rule's per-episode results into a Table 2/3 row (§9.3; late-accept as in Table 3)."""
    labs = [labels(ep) for ep in episodes]
    with_correct = [r["frr"] for r, lab in zip(results, labs) if "correct" in lab]
    late = [r["correct_accept"] for r, lab in zip(results, labs) if "correct" in lab and lab.index("correct") >= LATE]
    accepts = sum(r["accepts"] for r in results)
    good = sum(r["correct_accepts"] for r in results)
    fars = [r["far"] for r in results]
    injected = [r["far"] for r, ep in zip(results, episodes) if ep.get("flakiness", "natural").startswith("injected")]
    return {
        "calibration": mean(injected) / ALPHA if injected else None,  # Table 2: Pr[accept bad] / nominal, injected only
        "far": mean(fars),
        "far_ci": cluster_ci(fars, [ep.get("repo", "?") for ep in episodes]),
        "far_upper": far_upper(results, episodes),
        "frr": mean(with_correct) if with_correct else None,
        "marginal_accept": mean(r["marginal_accept"] for r in results),
        "blocks_per_accept": sum(r["blocks"] for r in results) / accepts if accepts else None,
        "late_accept": mean(late) if late else None,
        "budget_seconds": sum(r["seconds"] for r in results) / sum(r["reps"] for r in results),  # per episode
        "time_to_accept_min": sum(r["seconds_to_correct"] for r in results) / good / 60 if good else None,
    }


def primary_endpoint(results, episodes):
    """§9.4: paired per-episode FAR difference baseline - SeqAccept, its cluster CI, and whether it beats the margin."""
    base, ours, margin = PRIMARY
    diffs = [b["far"] - s["far"] for b, s in zip(results[base], results[ours])]
    lo, hi = cluster_ci(diffs, [ep.get("repo", "?") for ep in episodes])
    return {"contrast": f"{base} - {ours}", "diff": mean(diffs), "ci": (lo, hi), "margin": margin, "exceeds": lo > margin}


def q_band(episode):
    """§12 stratum: mean per-run false-pass rate q of the episode's Broken candidates."""
    fixes, labs = episode["fixes"], labels(episode)
    qs = [1 - sum(fixes[f]["truth"]) / len(fixes[f]["truth"]) for f, lab in zip(episode["stream"], labs) if lab == "broken"]
    if not qs:
        return "no broken"
    q = mean(qs)
    return "q < 0.01" if q < 0.01 else ("0.01 <= q < 0.1" if q < 0.1 else "q >= 0.1")


def composition(episodes):
    """Table 1 rows that the recorded episodes determine."""
    labs = [labels(ep) for ep in episodes]
    flat = [lab for ls in labs for lab in ls]
    qs = [1 - sum(ep["fixes"][f]["truth"]) / len(ep["fixes"][f]["truth"])
          for ep, ls in zip(episodes, labs) for f, lab in zip(ep["stream"], ls) if lab == "broken"]
    return {
        "episodes": len(episodes),
        "repositories": len({ep.get("repo") for ep in episodes}),
        "candidates": len(flat),
        "candidates_median_per_episode": median(len(ls) for ls in labs),
        **{k: flat.count(k) for k in ("correct", "broken", "marginal")},
        "episodes_with_late_correct": sum("correct" in ls and ls.index("correct") >= LATE for ls in labs),
        "mean_false_pass_rate_q": mean(qs) if qs else None,
    }


def split(key, episodes, results):
    """FAR by group (plan week 3 flakiness/bug type; H2 per model with attempts and % broken; §12 by q band)."""
    groups = {}
    for i, ep in enumerate(episodes):
        groups.setdefault(ep.get(key, "?"), []).append(i)
    rows = {}
    for g, idx in sorted(groups.items()):
        labs = [lab for i in idx for lab in labels(episodes[i])]
        rows[g] = {"streams": len(idx), "attempts": mean(len(episodes[i]["stream"]) for i in idx),
                   "broken_share": labs.count("broken") / len(labs),
                   "far": {n: mean(results[n][i]["far"] for i in idx) for n in results}}
    return rows


def pct(v):
    return "     -" if v is None else f"{100 * v:6.1f}"


def num(v):
    return "     -" if v is None else f"{v:6.1f}"


def print_report(rep):
    print("Table 1: composition")
    for k, v in rep["composition"].items():
        print(f"  {k:32} {v if not isinstance(v, float) else round(v, 3)}")
    print(f"\nTable 2/3 ({rep['episodes']} episodes, {REPS} replications, eps = p0 + {MARGIN}; calib. = FAR / alpha on "
          "injected-flakiness episodes)")
    print(f"{'rule':26} {'FAR %':>6} {'95% CI':>13} {'FRR %':>6} {'budget xB1':>10} {'t-accept min':>12} "
          f"{'blocks/acc':>10} {'late-acc %':>10} {'calib.':>6}")
    for name, s in rep["table"].items():
        ci = f"[{100 * s['far_ci'][0]:.1f},{100 * s['far_ci'][1]:.1f}]"
        cal = "     -" if s["calibration"] is None else f"{s['calibration']:5.1f}x"
        print(f"{name:26} {pct(s['far'])} {ci:>13} {pct(s['frr'])} {num(s['budget_x_b1']):>10} "
              f"{num(s['time_to_accept_min']):>12} {num(s['blocks_per_accept']):>10} {pct(s['late_accept']):>10} {cal:>6}")
    p = rep["primary"]
    print(f"\nPrimary endpoint ({p['contrast']}): {100 * p['diff']:.1f} points, CI [{100 * p['ci'][0]:.1f}, "
          f"{100 * p['ci'][1]:.1f}], margin {100 * p['margin']:.0f}: {'exceeded' if p['exceeds'] else 'NOT exceeded'}")
    for key, rows in rep["splits"].items():
        names = list(rep["table"])
        print(f"\nFAR % by {key}")
        print(f"{key:16} {'streams':>7} {'attempts':>8} {'broken %':>8} " + " ".join(f"{n[:12]:>12}" for n in names))
        for g, r in rows.items():
            print(f"{g:16} {r['streams']:>7} {r['attempts']:>8.1f} {100 * r['broken_share']:>8.1f} "
                  + " ".join(f"{100 * r['far'][n]:>12.1f}" for n in names))


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "episodes").glob("*.json"))
    dropped = [p for p in paths if excluded(p)]
    paths = [p for p in paths if not excluded(p)]
    print(f"D18: {len(dropped)} episode files left out ({sorted(EXCLUDED_RUNS)}), so every model scores the same episodes")
    loaded = [load_episode(p) for p in paths]
    is_main = [ep.get("flakiness", "natural") in MAIN_FLAKINESS for ep in loaded]
    paths = [p for p, keep in zip(paths, is_main) if keep]
    episodes = [ep for ep, keep in zip(loaded, is_main) if keep]
    swept = [ep for ep, keep in zip(loaded, is_main) if not keep and ep.get("flakiness", "").startswith("injected-")]  # D17
    b7 = [ep for ep in loaded if ep.get("flakiness") == "b7" and all("hidden" in f for f in ep["fixes"].values())]
    if not episodes:
        sys.exit("no episodes found")
    for ep in episodes:
        ep["q_band"] = q_band(ep)
    out = Path("results")
    out.mkdir(exist_ok=True)

    log(f"{len(episodes)} episodes x {len(all_rules())} rules x {REPS} replications: main table")
    results = evaluate(episodes)
    table = {name: summarize(rs, episodes) for name, rs in results.items()}
    b1 = table["B1 stop at first pass"]["budget_seconds"]
    for s in table.values():
        s["budget_x_b1"] = s["budget_seconds"] / b1 if b1 else None

    models = sorted({ep.get("model", "?") for ep in episodes})
    chart2 = {}
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        mine = [ep for ep in episodes if ep.get("model", "?") == model]
        log(f"chart 2 ({model}): FAR at the first {', '.join(map(str, PREFIXES))} fixes")
        by_k = [evaluate([{**ep, "stream": ep["stream"][:k]} for ep in mine]) for k in PREFIXES]
        chart2[model] = {name: [summarize(r[name], mine)["far"] for r in by_k] for name in results}
        for (name, ys), style in zip(chart2[model].items(), STYLES):
            ax.plot(PREFIXES, ys, label=name, **style)
        ax.axhline(0.05, color="gray", ls="--", lw=1)
        ax.set(title=model, xlabel="fixes tried", ylim=(-0.01, None))
    axes[0][0].set_ylabel("FAR")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "chart2_far_vs_fixes_tried.png", dpi=150)

    log(f"chart 3: FAR at margins {', '.join(map(str, MARGINS))}")
    by_m = [evaluate(episodes, margin=m) for m in MARGINS]
    chart3 = {name: [summarize(r[name], episodes)["far"] for r in by_m] for name in results}
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for (name, ys), style in zip(chart3.items(), STYLES):
        ax.plot(MARGINS, ys, label=name, **style)
    ax.axhline(0.05, color="gray", ls="--", lw=1)
    ax.set(xlabel="margin (eps = p0 + margin)", ylabel="FAR", ylim=(-0.01, None))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "chart3_far_vs_margin.png", dpi=150)

    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]  # each stream once
    nearmiss = {}
    for m in MASKS:
        log(f"boundary stress (D14): {len(natural)} natural episodes, failed blocks hidden with probability {m}")
        mine = [masked(ep, m) for ep in natural]
        res = evaluate(mine, reps=NEAR_REPS)
        rows = {name: summarize(rs, mine) for name, rs in res.items()}
        for name, s in rows.items():
            s["budget_x_b1"] = s["budget_seconds"] / rows["B1 stop at first pass"]["budget_seconds"]
            s["false_accepts"] = round(sum(r["far"] * r["reps"] for r in res[name]))
        gaps = [(1 - m) * sum(ep["fixes"][f]["truth"]) / len(ep["fixes"][f]["truth"]) - ep["p0"]
                for ep in mine for f in ep["stream"]]
        nearmiss[m] = {"composition": {**composition(mine), "broken_within_0.05_of_eps": sum(0.05 < g <= 0.10 for g in gaps)},
                       "runs": len(mine) * NEAR_REPS, "table": rows}

    late = {}
    for k in LATE_K:
        mine = late_streams(episodes, k)
        log(f"late streams (D15): {len(mine)} streams, first Correct fix at position {k + 1}")
        late[k] = {"streams": len(mine), "table": {name: summarize(rs, mine) for name, rs in evaluate(mine).items()}}

    sweep = {}
    if swept:
        repos = {ep.get("repo") for ep in swept}
        groups = {"injected-bernoulli": [ep for ep in episodes if ep.get("flakiness") == "injected-bernoulli"
                                         and ep.get("repo") in repos and "_b" not in ep["name"].split("@")[0]]}
        for ep in swept:
            groups.setdefault(ep["flakiness"], []).append(ep)
        for fl, mine in sorted(groups.items()):
            log(f"injection sweep (D17): {fl}, {len(mine)} episodes")
            sweep[fl] = {"false_pass": mine[0]["inject"]["false_pass"], "episodes": len(mine),
                         "composition": composition(mine),
                         "table": {name: summarize(rs, mine) for name, rs in evaluate(mine).items()}}

    b7_out = {}
    if b7:  # D21: every rule on the streams recorded against the visible tests, plus B7 itself
        log(f"B7 (D21): {len(b7)} episodes recorded with a hidden third of the tests")
        res = evaluate(b7)
        res["B7 protected hidden tests"] = [replay(stop_on_first_pass, protected_view(ep), REPS, seed=episode_seed(ep, i))
                                            for i, ep in enumerate(b7)]
        rows = {name: summarize(rs, b7) for name, rs in res.items()}
        for name, row in rows.items():
            row["budget_x_b1"] = row["budget_seconds"] / rows["B1 stop at first pass"]["budget_seconds"]
        b7_out = {"episodes": len(b7), "models": sorted({ep.get("model") for ep in b7}), "composition": composition(b7),
                  "table": rows}

    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {"reps": REPS, "margin": MARGIN, "late": LATE, "primary": PRIMARY},
        "episodes": len(episodes),
        "composition": composition(episodes),
        "table": table,
        "primary": primary_endpoint(results, episodes),
        "splits": {key: split(key, episodes, results) for key in SPLITS},
        "chart2_far_by_prefix": {"prefixes": PREFIXES, "far": chart2},
        "chart3_far_by_margin": {"margins": MARGINS, "far": chart3},
        "nearmiss": nearmiss,
        "late_streams": late,
        "injection_sweep": sweep,
        "b7": b7_out,
        "per_episode": [{"file": p.name, **{k: ep.get(k) for k in ("repo", "model", "repo_kind", "bug_kind", "q_band",
                                                                     "flakiness", "block_size")},
                         "labels": labels(ep), "rules": {n: results[n][i] for n in results}}
                        for i, (p, ep) in enumerate(zip(paths, episodes))],
    }
    log("report")
    print_report(report)
    print("\nBoundary stress (D14): FAR % on natural episodes, failed blocks hidden with probability m")
    for m, d in nearmiss.items():
        c = d["composition"]
        print(f"  m = {m}: {c['broken']} broken ({c['broken_within_0.05_of_eps']} within 0.05 of eps), {c['marginal']} marginal | "
              + " ".join(f"{n[:12]} {100 * s['far']:.1f}" for n, s in d["table"].items())
              + f" | SeqAccept marginal-accept {100 * d['table']['SeqAccept']['marginal_accept']:.1f}"
              + f" | SeqAccept false accepts {d['table']['SeqAccept']['false_accepts']} of {d['runs']} runs")
    print("\nLate streams (D15): k Broken fixes then one Correct; correct-accept % / FAR %")
    for k, d in late.items():
        print(f"  k = {k:2} ({d['streams']} streams): " + " ".join(
            f"{n[:12]} {100 * s['late_accept']:.1f}/{100 * s['far']:.1f}" for n, s in d["table"].items()))
    (out / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out / 'results.json'}")
