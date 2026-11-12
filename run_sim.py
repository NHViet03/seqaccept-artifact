"""Week 1: Chart 1 (false accepts vs candidates tried) + measured cost of SeqAccept + its operating curve near eps
(results/operating_curve.json). Usage: uv run python run_sim.py"""
import json
import math
import sys
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rules import all_rules, seq_accept
from sim import experiment

Q = 0.04  # a broken fix passes a block by luck 4% of the time (paper: 0.041)
RULES = all_rules()
LENGTHS = [1, 2, 3, 5, 8, 12, 18, 25, 30, 50]  # plan week 1 day 4: up to "all broken, 50 attempts"

SCENARIOS = {  # per-block failure probability of every (broken) candidate in the stream
    "lucky pass: broken fix fails 96% of blocks": 1 - Q,
    "sneaky: broken fix fails only 10% of blocks": 0.10,
}

WORST_P = (0.051, 0.06, 0.08)  # every broken candidate fails just above eps = 0.05: where the rules differ (§10.1, RQ2)
WORST_N = (1, 5, 15, 50)  # candidates in the stream

if __name__ == "__main__":
    out = Path("results")
    out.mkdir(exist_ok=True)
    print("Worst case (RQ2): all-broken streams just above eps, every rule; FAR and blocks per run")
    worst = {"eps": 0.05, "rules": list(RULES), "rows": []}
    for p in WORST_P:
        for n in WORST_N:
            reps = 1000 if n < 50 else 400
            cells = {name: experiment(rule, [p] * n, reps=reps, seed=31) for name, rule in RULES.items()}
            worst["rows"].append({"p": p, "n": n, "reps": reps, "far": {k: c["far"] for k, c in cells.items()},
                                  "blocks": {k: c["blocks"] / c["reps"] for k, c in cells.items()}})
            print(f"  p = {p:.3f}, n = {n:2}: " + " ".join(f"{k[:12]} {100 * c['far']:.1f}" for k, c in cells.items()), flush=True)
    (out / "worst_case.json").write_text(json.dumps(worst, indent=2))
    if sys.argv[1:] == ["worst"]:  # `python run_sim.py worst` refreshes only results/worst_case.json
        sys.exit()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, p_fail) in zip(axes, SCENARIOS.items()):
        print(f"\nAll-broken streams, {title}")
        print(f"{'candidates':>10} " + " ".join(f"{name[:14]:>14}" for name in RULES))
        curves = {name: [] for name in RULES}
        for n in LENGTHS:
            row = [experiment(rule, [p_fail] * n, reps=1000 if name.startswith("B") else 400, seed=n)["far"]
                   for name, rule in RULES.items()]
            for name, v in zip(RULES, row):
                curves[name].append(v)
            print(f"{n:>10} " + " ".join(f"{v:>14.3f}" for v in row))
        for name, ys in curves.items():
            ax.plot(LENGTHS, ys, marker="o", label=name)
        ax.axhline(0.05, color="gray", ls="--", lw=1, label="alpha = 5%")
        ax.set(xlabel="candidates the agent tried", title=title)
        ax.title.set_fontsize(9)
    axes[0].set_ylabel("false accepts (FAR)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "chart1_far_vs_candidates.png", dpi=150)

    print("\nCalibration: SeqAccept FAR on 50 all-broken fixes at the hardest broken rate p0 + 0.06 (eps = p0 + 0.05)")
    for p0 in (0.0, 0.02, 0.05):
        far = experiment(partial(seq_accept, eps=p0 + 0.05), [p0 + 0.06] * 50, reps=200, seed=11, p0=p0)["far"]
        print(f"  p0 = {p0:.2f}: FAR {far:.3f}  (nominal alpha 0.05)")

    print("\nCost: blocks spent to accept ONE perfect fix, by background flakiness p0 (eps = p0 + 0.05)")
    print(f"{'p0':>6} {'eps':>6} {'blocks/accept':>14}   (floor = best case, every block passes)")
    for p0 in (0.0, 0.02, 0.05):
        eps = p0 + 0.05
        got = experiment(partial(seq_accept, eps=eps), [p0], reps=200, seed=7, p0=p0)["blocks_per_accept"]
        floor = math.log(40) / math.log(1 / (1 - eps))
        print(f"{p0:>6.2f} {eps:>6.2f} {got:>14.1f}   floor ~{floor:.0f}")

    print("\nOperating curve (eps = 0.05): Pr[SeqAccept accepts one candidate that fails a share p of blocks]")
    curve = {"eps": 0.05, "single": [], "stream15": []}
    for p in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.055, 0.06, 0.07, 0.08, 0.10, 0.15):
        r = experiment(seq_accept, [p], reps=2000, seed=21)
        curve["single"].append({"p": p, "accept": r["accepts"] / r["reps"], "blocks": r["blocks"] / r["reps"]})
        print(f"  p = {p:.3f}: accept {r['accepts'] / r['reps']:.4f}, {r['blocks'] / r['reps']:.0f} blocks on average")
    print("Worst case: FAR on streams of 15 broken candidates just above eps (nominal alpha 0.05)")
    for p in (0.051, 0.06, 0.08, 0.10):
        far = experiment(seq_accept, [p] * 15, reps=1000, seed=22)["far"]
        curve["stream15"].append({"p": p, "far": far})
        print(f"  p = {p:.3f}: FAR {far:.4f}")
    (out / "operating_curve.json").write_text(json.dumps(curve, indent=2))
