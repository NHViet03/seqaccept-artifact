"""External-review analyses (offline, no LLM or Docker): power of a mixture accept wallet against SeqAccept's Kelly
wallet, a gamma / rho sweep, and the KL lower bound on blocks to accept. Writes results/power_tuning.json.

  uv run python power_tuning.py
"""
import json
import math
from functools import partial
from pathlib import Path

import replay
from rules import seq_accept
from sim import experiment

ALPHA, EPS = 0.05, 0.05
OP_P = (0.0, 0.02, 0.03, 0.04, 0.05)
OP_REPS = 2000
GAMMAS, RHOS = (0.25, 0.5, 0.75), (10, 20, 50)


def kl(p, q):
    """KL(Bern(p) || Bern(q)): the best possible average log-growth of a wallet against q when the truth is p."""
    a = p * math.log(p / q) if p > 0 else 0.0
    return a + (1 - p) * math.log((1 - p) / (1 - q))


def single(rule, p):
    """One candidate failing a share p of blocks: acceptance probability and mean blocks when accepted."""
    r = experiment(rule, [p], reps=OP_REPS, seed=71)
    acc = r["accepts"] / r["reps"]
    return {"accept": acc, "blocks_per_accept": r["blocks_per_accept"]}


def replay_summary(episodes, make_rule, reps=replay.REPS):
    res = [replay.replay(make_rule(ep["p0"] + replay.MARGIN), ep, reps, seed=replay.episode_seed(ep, i))
           for i, ep in enumerate(episodes)]
    s = replay.summarize(res, episodes)
    return {k: s[k] for k in ("far", "frr", "blocks_per_accept", "late_accept")}


if __name__ == "__main__":
    paths = sorted(Path("episodes").glob("*.json"))
    episodes = [replay.load_episode(p) for p in paths if not replay.excluded(p)]
    episodes = [ep for ep in episodes if ep.get("flakiness", "natural") in replay.MAIN_FLAKINESS]
    out = {"alpha": ALPHA, "eps": EPS, "mix_k": __import__("rules").MIX_K}

    print("1. operating curve: Kelly vs mixture, and the KL lower bound for the first candidate (alpha_1 = alpha / 2)")
    a1 = ALPHA / 2
    out["operating"] = []
    for p in OP_P:
        row = {"p": p, "kl_bound": math.log(1 / a1) / kl(p, EPS) if p < EPS else None,
               "kelly": single(seq_accept, p), "mixture": single(partial(seq_accept, bet="mixture"), p)}
        out["operating"].append(row)
        print(f"   p = {p}: kelly {row['kelly']}, mixture {row['mixture']}, KL bound {row['kl_bound']}")

    print("2. worst case p = 0.051: FAR")
    out["worst"] = {}
    for n in (15, 50):
        reps = 1000 if n < 50 else 400
        out["worst"][n] = {b: experiment(partial(seq_accept, bet=b), [0.051] * n, reps=reps, seed=31)["far"]
                           for b in ("kelly", "mixture")}
        print(f"   n = {n}: {out['worst'][n]}")

    print("3. recorded loops (all episodes) with the mixture wallet")
    out["recorded_mixture"] = replay_summary(episodes, lambda eps: partial(seq_accept, eps=eps, bet="mixture"))
    print(f"   {out['recorded_mixture']}")

    print("4. gamma / rho sweep on the recorded loops and on late streams (k = 10, 40)")
    late = {k: replay.late_streams(episodes, k) for k in (10, 40)}
    out["sweep"] = []
    for g in GAMMAS:
        for r in RHOS:
            if g != 0.5 and r != 20:
                continue  # one-at-a-time around the defaults
            mk = lambda eps, g=g, r=r: partial(seq_accept, eps=eps, gamma=g, reject_at=r)
            row = {"gamma": g, "rho": r, "recorded": replay_summary(episodes, mk),
                   "late": {k: replay_summary(v, mk) for k, v in late.items()}}
            out["sweep"].append(row)
            print(f"   gamma {g}, rho {r}: bpa {row['recorded']['blocks_per_accept']:.1f} frr {row['recorded']['frr']:.4f} "
                  f"far {row['recorded']['far']:.4f} late10 {row['late'][10]['late_accept']} late40 {row['late'][40]['late_accept']}")

    Path("results/power_tuning.json").write_text(json.dumps(out, indent=2, default=str))
    print("saved results/power_tuning.json")
