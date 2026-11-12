"""Second external review (offline, no LLM or Docker): replay-pool sensitivity, label uncertainty near eps, and a
margin-planning table. Writes results/review2.json.

  uv run python review2.py
"""
import json
import math
import random
from functools import partial
from pathlib import Path

import replay
from rules import all_rules, seq_accept
from sim import run_episodes

ALPHA = 0.05
SHORT_POOL = 100
PLAN_P0, PLAN_MARGINS = (0.0, 0.05, 0.2), (0.05, 0.10)


def counted_replay(rule, ep, reps, seed, pool_len=None):
    """replay.replay with an optional shortened pool, also counting how many blocks each run draws per candidate."""
    pools = [(ep["fixes"][f]["pool"][:pool_len] if pool_len else ep["fixes"][f]["pool"]) for f in ep["stream"]]
    rng, draws = random.Random(seed), []

    def make_block():
        n = [0] * len(pools)
        draws.append(n)

        def draw(j):
            n[j] += 1
            return rng.choice(pools[j])
        return draw

    res = run_episodes(rule, replay.labels(ep), make_block, reps)
    return res, [(len(pools[j]), k) for n in draws for j, k in enumerate(n) if k]


def beta_cdf(x, a, b, steps=2000):
    """Regularised incomplete beta by the midpoint rule on the log density (a, b >= 1); enough for tail checks."""
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    h = x / steps
    return sum(math.exp(lb + (a - 1) * math.log((i + .5) * h) + (b - 1) * math.log1p(-(i + .5) * h)) for i in range(steps)) * h


def label_risk(episodes):
    """Distinct (task, candidate) labels with a Beta(1 + fails, 1 + passes) posterior on the truth blocks: the chance that
    a Broken candidate truly has p <= eps, or a Correct one p > eps."""
    seen, out = set(), {"broken": 0, "broken_risk05": 0, "broken_risk50": 0, "correct": 0, "correct_risk05": 0, "correct_risk50": 0}
    for ep in episodes:
        task, eps = ep["name"].split("-")[0], ep["p0"] + replay.MARGIN
        for f, lab in zip(ep["stream"], replay.labels(ep)):
            if lab == "marginal" or (task, f) in seen:
                continue
            seen.add((task, f))
            t = ep["fixes"][f]["truth"]
            below = beta_cdf(eps, 1 + sum(t), 1 + len(t) - sum(t))  # P(p <= eps)
            if lab == "broken":
                out["broken"] += 1
                out["broken_risk05"] += below > 0.05
                out["broken_risk50"] += below > 0.5
            else:
                out["correct"] += 1
                out["correct_risk05"] += (1 - below) > 0.05
                out["correct_risk50"] += (1 - below) > 0.5
    return out


def kl(p, q):
    a = p * math.log(p / q) if p > 0 else 0.0
    return a + (1 - p) * math.log((1 - p) / (1 - q))


if __name__ == "__main__":
    paths = sorted(Path("episodes").glob("*.json"))
    episodes = [replay.load_episode(p) for p in paths if not replay.excluded(p)]
    episodes = [ep for ep in episodes if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS]
    flaky = [replay.load_episode(p) for p in sorted(Path("episodes_flaky").glob("*.json"))]
    out = {"short_pool": SHORT_POOL}

    print("1. replay-pool sensitivity (all main episodes)")
    rules = {"SeqAccept": lambda eps: partial(seq_accept, eps=eps),
             "B4 peeking Clopper-Pearson": lambda eps: all_rules(eps=eps)["B4 peeking Clopper-Pearson"]}
    out["pool"] = {}
    for name, mk in rules.items():
        for tag, pool_len in (("full", None), ("short", SHORT_POOL)):
            res, drawn = [], []
            for i, ep in enumerate(episodes):
                r, d = counted_replay(mk(ep["p0"] + replay.MARGIN), ep, replay.REPS, replay.episode_seed(ep, i), pool_len)
                res.append(r)
                drawn += d
            s = replay.summarize(res, episodes)
            out["pool"].setdefault(name, {})[tag] = {
                "far": s["far"], "frr": s["frr"], "blocks_per_accept": s["blocks_per_accept"],
                "evaluations": len(drawn), "over_pool": sum(k > size for size, k in drawn),
                "over_half_pool": sum(k > size / 2 for size, k in drawn)}
            print(f"   {name[:10]} {tag}: {out['pool'][name][tag]}")

    print("2. label uncertainty")
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    out["labels"] = {"natural": label_risk(natural), "realflaky": label_risk(flaky)}
    print(f"   {out['labels']}")

    print("3. planning: KL lower bound on blocks for a correct candidate (p = p0) and the first slice alpha / 2")
    out["planning"] = [{"p0": p0, "margin": mg, "blocks": math.log(2 / ALPHA) / kl(p0, p0 + mg)}
                       for p0 in PLAN_P0 for mg in PLAN_MARGINS]
    print(f"   {out['planning']}")

    Path("results/review2.json").write_text(json.dumps(out, indent=2))
    print("saved results/review2.json")
