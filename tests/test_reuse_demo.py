from reuse_demo import far


def test_reused_evidence_gives_a_false_accept_rate_that_does_not_shrink_with_alpha():  # Thm. 8
    for alpha in (0.05, 0.001):
        assert 0.25 <= far(True, alpha, reps=400) <= 0.45  # Ville: <= stop_at * gamma = 0.45


def test_fresh_evidence_keeps_it_under_alpha():  # §5.4
    assert far(False, 0.05, reps=400) <= 0.05
