"""Simulated candidate streams with known truth, and the metrics from the paper (FAR, FRR, cost).

A stream is a list of true per-block failure probabilities, one per candidate, in the order the
agent produced them. `label` turns a failure probability into the paper's Correct/Marginal/Broken.
"""
import random


def label(p, p0=0.0):
    """§8: Correct p < 0.01, Broken p > 0.05, else Marginal; shifted by p0 = background failure rate of the
    original code (plan 2.3: flaky tests also fail correct fixes)."""
    if p < p0 + 0.01:
        return "correct"
    if p > p0 + 0.05:
        return "broken"
    return "marginal"


def run_episodes(rule, labels, make_block, reps, cost=None):
    """Run `rule` `reps` times over a stream whose true labels are `labels`.

    make_block() returns a fresh block(j) for one episode; cost(j) is the seconds one block of candidate j takes.
    §9.3 metrics:
    FAR: share of episodes that end by accepting a Broken candidate.
    FRR: share of episodes containing a Correct candidate that end by accepting nothing.
    correct_accept: share ending by accepting a Correct candidate (Table 3 late-accept, on late streams).
    marginal_accept: share ending by accepting a Marginal candidate (counted by neither FAR nor FRR).
    blocks / accepts / seconds / seconds_to_correct / correct_accepts: totals over reps, so blocks per accepted fix,
    budget and time-to-correct-acceptance (Table 2) can be pooled across episodes.
    """
    has_correct = "correct" in labels
    far = frr = good = marginal = accepts = blocks = 0
    seconds = to_correct = 0.0
    for _ in range(reps):
        spent = [0.0]
        draw = make_block()

        def block(j):
            spent[0] += cost(j) if cost else 0.0
            return draw(j)

        res = rule(len(labels), block)
        blocks += res.blocks
        seconds += spent[0]
        if res.accepted is None:
            frr += has_correct
        else:
            accepts += 1
            far += labels[res.accepted] == "broken"
            marginal += labels[res.accepted] == "marginal"
            if labels[res.accepted] == "correct":
                good += 1
                to_correct += spent[0]
    return {
        "far": far / reps,
        "frr": frr / reps,
        "correct_accept": good / reps,
        "marginal_accept": marginal / reps,
        "reps": reps,
        "blocks": blocks,
        "accepts": accepts,
        "blocks_per_accept": blocks / accepts if accepts else None,
        "seconds": seconds,
        "correct_accepts": good,
        "seconds_to_correct": to_correct,
    }


def experiment(rule, stream, reps, seed=0, p0=0.0):
    """Simulated stream: candidate j fails each block with probability stream[j]."""
    rng = random.Random(seed)
    make_block = lambda: (lambda j: int(rng.random() < stream[j]))
    return run_episodes(rule, [label(p, p0) for p in stream], make_block, reps)
