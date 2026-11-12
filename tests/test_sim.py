from functools import partial

import pytest

from rules import seq_accept, stop_on_first_pass
from sim import experiment, label

Q = 0.04  # chance a broken fix passes one block by luck (paper: q = 0.041)


def test_labels_follow_the_papers_thresholds_shifted_by_the_background_rate():
    assert label(0.00) == "correct"
    assert label(0.009) == "correct"
    assert label(0.01) == "marginal"  # §8: Correct is strictly below 0.01
    assert label(0.03) == "marginal"
    assert label(0.06) == "broken"
    assert label(0.025, p0=0.02) == "correct"  # 0.025 < p0 + 0.01
    assert label(0.07, p0=0.02) == "marginal"
    assert label(0.08, p0=0.02) == "broken"


@pytest.mark.parametrize("n", [1, 10, 30])
def test_stop_on_first_pass_false_accepts_follow_1_minus_one_minus_q_to_the_n(n):
    stream = [1 - Q] * n  # every candidate broken, each block fails 96% of the time
    got = experiment(stop_on_first_pass, stream, reps=3000, seed=1)["far"]
    assert got == pytest.approx(1 - (1 - Q) ** n, abs=0.03)


def test_seqaccept_false_accepts_stay_under_alpha_however_long_the_stream():
    stream = [1 - Q] * 30
    assert experiment(seq_accept, stream, reps=300, seed=2)["far"] <= 0.05


def test_without_refunds_a_late_correct_candidate_is_almost_never_accepted():
    stream = [0.5] * 60 + [0.0]  # 60 clearly broken fixes, then the one correct fix
    with_refund = experiment(seq_accept, stream, reps=50, seed=3)
    no_refund = experiment(partial(seq_accept, refund=False), stream, reps=50, seed=3)
    assert with_refund["frr"] <= 0.1
    assert no_refund["frr"] >= 0.9  # its shrinking budget needs more than max_blocks passes in a row


def test_without_refunds_even_a_moderately_late_correct_candidate_costs_several_times_more():
    stream = [0.5] * 30 + [0.0]
    with_refund = experiment(seq_accept, stream, reps=50, seed=3)
    no_refund = experiment(partial(seq_accept, refund=False), stream, reps=50, seed=3)
    assert no_refund["blocks_per_accept"] > 2 * with_refund["blocks_per_accept"]  # measured ~2.6x (719 vs 279)


def test_experiment_reports_blocks_spent_per_accepted_fix():
    out = experiment(seq_accept, [0.0], reps=20, seed=4)
    assert out["far"] == 0.0 and out["frr"] == 0.0
    assert 72 <= out["blocks_per_accept"] <= 150


def test_a_correct_fix_is_not_abandoned_when_the_original_code_itself_fails_5_percent_of_blocks():
    from rules import seq_accept as rule

    # background flakiness p0=5% -> eps=10%; a perfect fix (fails exactly p0) must still usually get accepted
    out = experiment(partial(rule, eps=0.10), [0.05], reps=100, seed=5, p0=0.05)
    assert out["frr"] <= 0.10
