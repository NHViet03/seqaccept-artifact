"""Acceptance rules for an agentic repair loop under flaky CI.

A rule walks a stream of candidate patches 0..n-1. `block(j)` runs one test block on
candidate j and returns 1 if it FAILED, 0 if it passed. A rule returns the index of the
candidate it accepts (or None) and how many blocks it spent.
"""
import math
from dataclasses import dataclass
from functools import lru_cache, partial


@dataclass(frozen=True)
class Result:
    accepted: int | None
    blocks: int


def stop_on_first_pass(n, block):  # B1: current practice
    blocks = 0
    for j in range(n):
        blocks += 1
        if block(j) == 0:
            return Result(j, blocks)
    return Result(None, blocks)


def rerun_k(n, k, block):  # B2: k blocks in a row must all pass
    blocks = 0
    for j in range(n):
        for _ in range(k):
            blocks += 1
            if block(j):
                break
        else:
            return Result(j, blocks)
    return Result(None, blocks)


def majority_of_5(n, block):  # B3: fixed 5 blocks, accept on >= 3 passes
    blocks = 0
    for j in range(n):
        fails = sum(block(j) for _ in range(5))
        blocks += 5
        if fails <= 2:
            return Result(j, blocks)
    return Result(None, blocks)


def bets(fails, t, eps):
    """§7 truncated Kelly bets for the accept wallet (lam) and the reject wallet (mu, D2), from the 1/2-smoothed
    failure rate of the t blocks seen so far (known BEFORE the next block)."""
    guess = (fails + 0.5) / (t + 1)
    lam = min(max((eps - guess) / (eps * (1 - eps)), 0.0), 0.9 / (1 - eps))
    mu = min(max((guess - eps) / (eps * (1 - eps)), 0.0), 0.9 / eps)
    return lam, mu


MIX_K = 10  # grid size of the mixture wallet


def mix_lambdas(eps, k=MIX_K):
    """Fixed bets of the mixture wallet: an even grid up to SeqAccept's cap 0.9 / (1 - eps)."""
    return [(i / k) * 0.9 / (1 - eps) for i in range(1, k + 1)]


def seq_accept(n, block, alpha=0.05, eps=0.05, gamma=0.5, refund=True, max_blocks=800, reject_at=20, bet="kelly"):
    """SeqAccept (refund=False gives B6). One betting wallet per candidate, error spent from a shared pool.

    Accept j when its wallet E >= 1/alpha_j; reject when the opposing wallet R >= reject_at (an anytime-valid
    "lower confidence bound > eps" at level 1/reject_at, Alg. 1 line 9). Only rejected candidates refund
    alpha_j * min(E, 1 - s_j). Every candidate starts from wallet 1: evidence is never carried between candidates.
    bet="mixture" replaces the plug-in Kelly accept wallet by the average of fixed-bet wallets (mix_lambdas); each is a
    nonnegative supermartingale under p > eps, so their average is too, and Theorem 6 applies unchanged.
    """
    pool, blocks = alpha, 0
    for j in range(n):
        alpha_j = gamma * pool  # slice of the error budget this candidate may spend
        pool -= alpha_j
        E = R = 1.0  # E: bets the candidate is fine (p <= eps); R: bets it is broken (p > eps)
        mix = [1.0] * MIX_K if bet == "mixture" else None
        lams = mix_lambdas(eps) if mix else None
        peak = 1.0
        fails = 0
        for t in range(max_blocks):
            lam, mu = bets(fails, t, eps)
            y = block(j)
            blocks += 1
            fails += y
            if mix:
                mix = [w * (1 + l * (eps - y)) for w, l in zip(mix, lams)]
                E = sum(mix) / MIX_K
            else:
                E *= 1 + lam * (eps - y)
            R *= 1 + mu * (y - eps)
            peak = max(peak, E)
            if E >= 1 / alpha_j:
                return Result(j, blocks)
            if R >= reject_at:  # Alg. 1 line 9: rejected
                if refund == "uncapped":  # the refund the online subset ran (D20, D24): not covered by Theorem 6
                    pool += alpha_j * (1 - alpha_j * peak)
                elif refund:  # line 11: refund the unspent part, s_j = alpha_j * peak, never more than the wallet
                    pool += alpha_j * min(E, 1 - alpha_j * peak)  # holds (D24): W + alpha_j E stays a supermartingale
                break
        # out of blocks: NO refund, unlike Alg. 1 line 14 (it lifts FAR above alpha, see test_paper_conformance)
    return Result(None, blocks)


def discard_spending(n, block, alpha=0.05, eps=0.05, gamma=0.5, tau=0.5, max_blocks=800, reject_at=20):
    """Comparator for refunds: online FWER spending with discarding (ADDIS-spending style, Tian and Ramdas 2021), with
    SeqAccept's wallets. Candidate j may spend s_j = gamma * pool and is tested at level tau * s_j. Its anytime p-value
    is 1 / (peak wallet); if that never fell to tau, the candidate is discarded and its whole s_j returns to the pool,
    otherwise s_j is spent. Validity in the adaptive setting is not covered by Theorem 6."""
    pool, blocks = alpha, 0
    for j in range(n):
        s_j = gamma * pool
        E = R = peak = 1.0
        fails = 0
        for t in range(max_blocks):
            lam, mu = bets(fails, t, eps)
            y = block(j)
            blocks += 1
            fails += y
            E *= 1 + lam * (eps - y)
            R *= 1 + mu * (y - eps)
            peak = max(peak, E)
            if E >= 1 / (tau * s_j):
                return Result(j, blocks)
            if R >= reject_at:
                break
        if peak >= 1 / tau:  # not discarded: its slice is spent
            pool -= s_j
    return Result(None, blocks)


def fixed_spending(n, block, slices, eps=0.05, max_blocks=800, reject_at=20):
    """No-refund error spending on a fixed schedule: candidate j gets alpha_j = slices(j) (sum <= alpha), with
    SeqAccept's wallets. Valid by Theorem 6 with r_j = 0. A slice of 0 means the candidate cannot be accepted."""
    blocks = 0
    for j in range(n):
        alpha_j = slices(j)
        E = R = 1.0
        fails = 0
        for t in range(max_blocks):
            lam, mu = bets(fails, t, eps)
            y = block(j)
            blocks += 1
            fails += y
            E *= 1 + lam * (eps - y)
            R *= 1 + mu * (y - eps)
            if alpha_j > 0 and E >= 1 / alpha_j:
                return Result(j, blocks)
            if R >= reject_at:
                break
    return Result(None, blocks)


def fixed_n(n, block, slices, eps=0.05, scale=2):
    """Valid fixed-sample rule: each candidate j with alpha_j = slices(j) > 0 runs up to m blocks and is accepted when
    the exact binomial test of p >= eps rejects at level alpha_j (P_eps(X <= fails) <= alpha_j) after all m, else
    rejected. It stops early, rejecting, once fails exceeds the critical value: acceptance is then impossible, and
    acceptance only ever happens at m, so stopping early changes no acceptance. m = scale * the fewest blocks at which
    a candidate with no failure can be accepted. Union bound: FAR <= sum alpha_j."""
    blocks = 0
    for j in range(n):
        alpha_j = slices(j)
        if alpha_j <= 0:
            continue
        m = math.ceil(scale * math.ceil(math.log(alpha_j) / math.log(1 - eps)))
        crit = critical(m, eps, alpha_j)
        fails = 0
        for _ in range(m):
            fails += block(j)
            blocks += 1
            if fails > crit:
                break
        if fails <= crit:
            return Result(j, blocks)
    return Result(None, blocks)


@lru_cache(maxsize=None)
def critical(m, eps, alpha):
    """Largest failure count k with P_eps(X <= k) <= alpha for X ~ Bin(m, eps); -1 if even k = 0 fails the test."""
    k = -1
    while k + 1 <= m and binom_cdf(k + 1, m, eps) <= alpha:
        k += 1
    return k


def binom_cdf(k, m, p):
    return sum(math.comb(m, i) * p ** i * (1 - p) ** (m - i) for i in range(k + 1))


def harmonic(alpha=0.05):  # alpha_j = alpha / (j (j + 1)), j = 1, 2, ...: sums to alpha
    return lambda j: alpha / ((j + 1) * (j + 2))


def bonferroni(n_max, alpha=0.05):  # alpha / n_max for each of the first n_max candidates, 0 after
    return lambda j: alpha / n_max if j < n_max else 0.0


@lru_cache(maxsize=None)
def _cp_thresholds(eps, alpha, max_blocks):
    """B4 lookup tables. A one-sided Clopper-Pearson upper bound on k fails in t blocks is < eps exactly when
    Binom(t, eps).cdf(k) < alpha, and the lower bound is > eps exactly when Binom(t, eps).sf(k - 1) < alpha.
    accept_t[k]: fewest blocks after which k fails accept; reject_k[t]: fewest fails that reject after t blocks."""
    def pmf(k, t):
        return math.exp(math.lgamma(t + 1) - math.lgamma(k + 1) - math.lgamma(t - k + 1)
                        + k * math.log(eps) + (t - k) * math.log1p(-eps))

    accept_t, k = [], 0
    while True:
        t = next((t for t in range(max(k, 1), max_blocks + 1) if sum(pmf(i, t) for i in range(k + 1)) < alpha), None)
        if t is None:
            break
        accept_t.append(t)
        k += 1
    reject_k = [None]
    for t in range(1, max_blocks + 1):
        tail, k = 0.0, t + 1  # t + 1: no number of fails rejects yet
        for i in range(t, 0, -1):  # tail = Pr[X >= i]
            tail += pmf(i, t)
            if tail >= alpha:
                break
            k = i
        reject_k.append(k)
    return accept_t, reject_k


def peeking_cp(n, block, alpha=0.05, eps=0.05, max_blocks=800):  # B4: fixed-sample CI checked after every block
    accept_t, reject_k = _cp_thresholds(eps, alpha, max_blocks)
    blocks = 0
    for j in range(n):
        fails = 0
        for t in range(1, max_blocks + 1):
            fails += block(j)
            blocks += 1
            if fails < len(accept_t) and t >= accept_t[fails]:
                return Result(j, blocks)
            if fails >= reject_k[t]:
                break
    return Result(None, blocks)


def naive_cs(n, block, alpha=0.05, eps=0.05, max_blocks=800, reject_at=20):
    """B5: SeqAccept's per-candidate betting wallets, but every candidate gets the full alpha (no error spending)."""
    blocks = 0
    for j in range(n):
        E = R = 1.0
        fails = 0
        for t in range(max_blocks):
            lam, mu = bets(fails, t, eps)
            y = block(j)
            blocks += 1
            fails += y
            E *= 1 + lam * (eps - y)
            R *= 1 + mu * (y - eps)
            if E >= 1 / alpha:
                return Result(j, blocks)
            if R >= reject_at:
                break
    return Result(None, blocks)


def all_rules(eps=0.05):
    """The rules the pilot compares (plan 2.6). eps (= p0 + margin) only matters to the SeqAccept variants."""
    return {
        "B1 stop at first pass": stop_on_first_pass,
        "B2 rerun-3 all green": lambda n, block: rerun_k(n, 3, block),
        "B3 majority of 5": majority_of_5,
        "B4 peeking Clopper-Pearson": partial(peeking_cp, eps=eps),
        "B5 naive confidence sequence": partial(naive_cs, eps=eps),
        "B6 SeqAccept, no refund": partial(seq_accept, eps=eps, refund=False),
        "SeqAccept": partial(seq_accept, eps=eps),
    }
