from rules import Result, majority_of_5, naive_cs, peeking_cp, rerun_k, seq_accept, stop_on_first_pass
from sim import experiment

PASS, FAIL = 0, 1


def scripted(outcomes):
    """block(j) pops the next scripted outcome for candidate j (1 = block failed)."""
    queues = {j: list(o) for j, o in outcomes.items()}
    return lambda j: queues[j].pop(0)


def test_b1_accepts_first_candidate_whose_single_block_passes():
    block = scripted({0: [FAIL], 1: [FAIL], 2: [PASS], 3: [PASS]})
    assert stop_on_first_pass(4, block) == Result(accepted=2, blocks=3)


def test_b1_accepts_nothing_when_every_block_fails():
    block = scripted({0: [FAIL], 1: [FAIL]})
    assert stop_on_first_pass(2, block) == Result(accepted=None, blocks=2)


def test_b2_needs_k_passes_in_a_row_and_stops_a_candidate_at_its_first_failure():
    block = scripted({0: [PASS, PASS, FAIL], 1: [PASS, FAIL], 2: [PASS, PASS, PASS]})
    # cand 0 uses 3 blocks (fails on 3rd), cand 1 uses 2, cand 2 passes all 3
    assert rerun_k(3, 3, block) == Result(accepted=2, blocks=8)


def test_b3_runs_all_five_blocks_and_accepts_on_three_passes():
    block = scripted({0: [PASS, PASS, FAIL, FAIL, FAIL], 1: [FAIL, PASS, PASS, FAIL, PASS]})
    assert majority_of_5(2, block) == Result(accepted=1, blocks=10)


def test_b4_accepts_a_clean_fix_after_59_blocks_the_clopper_pearson_minimum():
    assert peeking_cp(1, lambda j: PASS) == Result(accepted=0, blocks=59)  # 0.95 ** 59 < 0.05 < 0.95 ** 58


def test_b4_peeking_and_b5_no_spending_lose_the_bound_that_seqaccept_keeps():
    stream = [0.06] * 15  # every candidate just above eps = 0.05
    assert experiment(peeking_cp, stream, reps=100, seed=1)["far"] > 0.3
    assert experiment(naive_cs, stream, reps=100, seed=1)["far"] > experiment(seq_accept, stream, reps=100, seed=1)["far"]
