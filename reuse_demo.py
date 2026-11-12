"""Theorem 8 demo (plan day 18): carrying evidence across candidates makes FAR a constant independent of alpha.

D = K equally likely input branches; a block runs one random branch. The agent is Thm. 8's adversary (it sees blocks):
  1. it proposes c1, correct everywhere, and abandons it just before acceptance (E >= stop_at / alpha_1);
  2. it proposes c2 = c1 broken on n_bad branches that none of c1's blocks ran, so p2 = n_bad / K > eps.
Reuse (the variant Thm. 8 rules out): c2 inherits c1's blocks, as none of them ran an edited branch.
Without reuse c2 starts from wallet 1, as SeqAccept does (§5.4). By Ville, FAR with reuse <= stop_at * gamma for any alpha.

Usage: uv run python reuse_demo.py
"""
import random

from rules import bets


def episode(rng, reuse, alpha=0.05, eps=0.05, gamma=0.5, K=100, n_bad=6, stop_at=0.9, max_blocks=800, reject_at=20):
    """True if the broken c2 is accepted. Error slices as in Alg. 1 lines 2-3 (c1 abandoned: no refund)."""
    a1 = gamma * alpha
    a2 = gamma * (alpha - a1)
    E, fails, t, seen = 1.0, 0, 0, set()
    while E < stop_at / a1:  # c1 is correct: every block passes
        E *= 1 + bets(fails, t, eps)[0] * eps
        t += 1
        seen.add(rng.randrange(K))
    unseen = [b for b in range(K) if b not in seen]
    if len(unseen) < n_bad:
        return False  # no untested branches left to hide the bug in
    bad = set(rng.sample(unseen, n_bad))
    if not reuse:
        E, fails, t = 1.0, 0, 0
    R = 1.0
    for _ in range(max_blocks):
        lam, mu = bets(fails, t, eps)
        y = int(rng.randrange(K) in bad)
        fails, t = fails + y, t + 1
        E *= 1 + lam * (eps - y)
        R *= 1 + mu * (y - eps)
        if E >= 1 / a2:
            return True
        if R >= reject_at:
            return False
    return False


def far(reuse, alpha, reps=2000, seed=0):
    rng = random.Random(seed)
    return sum(episode(rng, reuse, alpha=alpha) for _ in range(reps)) / reps


if __name__ == "__main__":
    print("FAR of a broken c2 (p2 = 0.06 > eps = 0.05) after a correct c1 abandoned just before acceptance")
    print(f"{'alpha':>6} {'fresh blocks (SeqAccept)':>25} {'reused blocks':>14}")
    rows = []
    for alpha in (0.05, 0.01, 0.001):
        rows.append({"alpha": alpha, "fresh": far(False, alpha), "reused": far(True, alpha)})
        print(f"{alpha:>6} {rows[-1]['fresh']:>25.3f} {rows[-1]['reused']:>14.3f}")
    import json
    from pathlib import Path

    params = {k: v.default for k, v in __import__("inspect").signature(episode).parameters.items() if k not in ("rng", "reuse")}
    Path("results/reuse_demo.json").write_text(json.dumps({"reps": 2000, "params": params, "rows": rows}, indent=2))
