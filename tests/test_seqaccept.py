import math
import random

import pytest

from rules import Result, seq_accept

PASS, FAIL = 0, 1


def always(outcome):
    return lambda j: outcome


def test_accepting_even_a_perfect_candidate_needs_at_least_the_math_floor():
    # A pass can grow the wallet by at most 1/(1-eps) per block, and fix #1 needs 1/(gamma*alpha)=40.
    floor = math.ceil(math.log(40) / math.log(1 / 0.95))  # 72 blocks
    res = seq_accept(1, always(PASS), alpha=0.05, eps=0.05, gamma=0.5)
    assert res.accepted == 0
    assert floor <= res.blocks <= 150


def test_a_candidate_that_always_fails_is_rejected_in_a_handful_of_blocks():
    res = seq_accept(1, always(FAIL))
    assert res == Result(accepted=None, blocks=res.blocks) and res.blocks <= 10


def test_refund_makes_the_next_candidate_cheaper_to_accept():
    block = lambda j: FAIL if j == 0 else PASS  # cand 0 clearly broken, cand 1 perfect
    with_refund = seq_accept(2, block, refund=True)
    without = seq_accept(2, block, refund=False)
    assert with_refund.accepted == without.accepted == 1
    assert with_refund.blocks < without.blocks


def test_an_undecidable_candidate_is_abandoned_after_max_blocks():
    # fails exactly every 20th block (rate = eps): neither accept nor reject is ever justified
    counter = {"n": 0}

    def block(j):
        counter["n"] += 1
        return FAIL if counter["n"] % 20 == 0 else PASS

    res = seq_accept(1, block, max_blocks=60)
    assert res == Result(accepted=None, blocks=60)


def test_an_abandoned_candidate_gets_no_refund_unlike_alg1_line_14():
    # cand 0 fails exactly every 20th block (rate = eps) so it is abandoned at max_blocks; cand 1 is perfect
    def make_block():
        n = {"0": 0}

        def block(j):
            if j:
                return PASS
            n["0"] += 1
            return FAIL if n["0"] % 20 == 0 else PASS

        return block

    with_refund = seq_accept(2, make_block(), max_blocks=60, refund=True)
    without = seq_accept(2, make_block(), max_blocks=60, refund=False)
    assert with_refund == without  # giving up refunds nothing, so refund=True changes nothing here


@pytest.mark.parametrize("p0", [0.0, 0.02, 0.05])  # plan day 4: background rates 0%, 2%, 5%
def test_false_accept_rate_on_all_broken_streams_stays_below_alpha(p0):
    rng = random.Random(0)
    p_fail = p0 + 0.06  # just above eps = p0 + 0.05: the hardest "broken" to tell apart
    reps, accepted_broken = 300, 0
    for _ in range(reps):
        res = seq_accept(20, lambda j: int(rng.random() < p_fail), alpha=0.05, eps=p0 + 0.05)
        accepted_broken += res.accepted is not None
    assert accepted_broken / reps <= 0.05
