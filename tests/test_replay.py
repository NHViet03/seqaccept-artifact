import json

import pytest

from replay import LATE, cluster_ci, composition, evaluate, load_episode, primary_endpoint, q_band, replay, summarize
from rules import seq_accept, stop_on_first_pass

GOOD = {"truth": [0] * 200, "pool": [0] * 200}  # never fails: Correct
BAD = {"truth": [1] * 200, "pool": [1] * 200}  # always fails: Broken
# fails 20% of the time in truth, but the pool is all-green: a rule that sees only the pool is fooled
SNEAKY = {"truth": [1, 0, 0, 0, 0] * 40, "pool": [0] * 200}


def episode(stream, **fixes):
    return {"p0": 0.0, "stream": stream, "fixes": fixes}


def test_replay_grades_each_accept_against_the_truth_runs_not_the_pool():
    ep = episode(["sneaky"], sneaky=SNEAKY)
    out = replay(stop_on_first_pass, ep, reps=10, seed=0)
    assert out["far"] == 1.0  # accepted a fix whose truth failure rate is 0.2 > 0.05


def test_replay_finds_the_correct_fix_after_a_broken_one():
    ep = episode(["bad", "good"], bad=BAD, good=GOOD)
    out = replay(seq_accept, ep, reps=10, seed=0)
    assert out["far"] == 0.0 and out["frr"] == 0.0


def test_the_same_fix_appearing_twice_in_a_stream_shares_one_set_of_results():
    ep = episode(["bad", "bad", "good"], bad=BAD, good=GOOD)
    assert replay(seq_accept, ep, reps=5, seed=0)["far"] == 0.0


def test_load_episode_reads_json(tmp_path):
    path = tmp_path / "ep.json"
    path.write_text(json.dumps(episode(["good"], good=GOOD)))
    assert load_episode(path)["stream"] == ["good"]


def test_replay_rejects_a_fix_with_no_pool_results():
    ep = episode(["x"], x={"truth": [0] * 10, "pool": []})
    with pytest.raises(ValueError, match="pool"):
        replay(seq_accept, ep, reps=1, seed=0)


def test_seqaccept_is_run_at_eps_p0_plus_margin_so_a_perfect_fix_in_a_flaky_repo_is_accepted():
    flaky_good = {"truth": ([1] + [0] * 19) * 10, "pool": ([1] + [0] * 19) * 10}  # fails 5% = the background rate
    ep = {**episode(["good"], good=flaky_good), "p0": 0.05}
    assert evaluate([ep], reps=20)["SeqAccept"][0]["frr"] <= 0.2
    assert replay(seq_accept, ep, reps=20)["frr"] >= 0.8  # at the default eps = 0.05 it is (almost) never accepted


def test_late_accept_counts_only_streams_whose_first_correct_fix_is_late():
    early = episode(["good"], good=GOOD)
    late = episode(["bad"] * LATE + ["good"], bad=BAD, good=GOOD)
    eps = [early, late]
    s = summarize(evaluate(eps, reps=5)["SeqAccept"], eps)
    assert s["late_accept"] == 1.0 and s["far"] == 0.0 and s["frr"] == 0.0
    assert summarize(evaluate([early], reps=5)["SeqAccept"], [early])["late_accept"] is None


def test_accepting_a_marginal_fix_is_counted_apart_from_far_and_frr():
    marg = {"truth": [1] + [0] * 39, "pool": [0] * 200}  # fails 2.5%: Marginal; the all-green pool gets it accepted
    eps = [episode(["m"], m=marg)]
    s = summarize(evaluate(eps, reps=5)["B1 stop at first pass"], eps)
    assert (s["marginal_accept"], s["far"]) == (1.0, 0.0)


def test_frr_is_averaged_only_over_streams_that_contain_a_correct_fix():
    eps = [episode(["bad"], bad=BAD), episode(["good"], good=GOOD)]
    assert summarize(evaluate(eps, reps=5)["B1 stop at first pass"], eps)["frr"] == 0.0


TIMED_GOOD = {**GOOD, "seconds": [2.0] * 200}
TIMED_BAD = {**BAD, "seconds": [1.0] * 200}


def test_budget_and_time_to_accept_charge_each_drawn_block_its_recorded_seconds():
    ep = {**episode(["bad", "good"], bad=TIMED_BAD, good=TIMED_GOOD), "repo": "a"}
    s = summarize(evaluate([ep], reps=5)["B1 stop at first pass"], [ep])
    assert s["budget_seconds"] == 3.0  # one 1 s block on bad, one 2 s block on good
    assert s["time_to_accept_min"] == 3.0 / 60


def test_cluster_bootstrap_resamples_whole_repositories():
    assert cluster_ci([0.1, 0.1, 0.3], ["a", "a", "b"]) == (0.1, 0.3)
    assert cluster_ci([0.2] * 4, ["a", "b", "a", "b"]) == (0.2, 0.2)


def test_primary_endpoint_is_the_paired_b3_minus_seqaccept_difference():
    half = {"truth": [1, 1, 0, 0, 0] * 40, "pool": [1, 1, 0, 0, 0] * 40}  # Broken, fails 40%: B3 often accepts it
    eps = [{**episode(["h"], h=half), "repo": r} for r in "ab"]
    res = evaluate(eps, reps=20)
    p = primary_endpoint(res, eps)
    assert p["contrast"] == "B3 majority of 5 - SeqAccept" and p["diff"] > 0.3
    assert p["exceeds"] == (p["ci"][0] > 0.03)


def test_composition_and_q_band_follow_table_1_and_section_12():
    lucky = {"truth": [1] * 190 + [0] * 10, "pool": [1] * 200}  # Broken, passes 5% of runs: q = 0.05
    ep = episode(["lucky", "good"], lucky=lucky, good=GOOD)
    c = composition([ep])
    assert (c["candidates"], c["correct"], c["broken"]) == (2, 1, 1) and abs(c["mean_false_pass_rate_q"] - 0.05) < 1e-9
    assert q_band(ep) == "0.01 <= q < 0.1" and q_band(episode(["good"], good=GOOD)) == "no broken"


def test_an_episode_replays_the_same_whatever_else_is_in_the_set():
    import replay
    ep = {"name": "x-gpt-4o-1", "p0": 0.0, "stream": ["a", "b"], "repo": "r",
          "fixes": {"a": {"truth": [1] * 200, "pool": [0, 1] * 100}, "b": {"truth": [0] * 200, "pool": [0] * 200}}}
    other = {**ep, "name": "y-gpt-4o-1"}
    alone, after = replay.evaluate([ep]), replay.evaluate([other, ep])
    assert all(alone[r][0] == after[r][1] for r in alone)


def test_zero_events_get_a_one_sided_upper_bound():
    import replay
    assert abs(replay.cp_upper(0, 88) - (1 - 0.05 ** (1 / 88))) < 1e-9
    assert abs(replay.cp_upper(0, 10) - 0.2589) < 1e-3 and replay.cp_upper(3, 3) == 1.0
    assert 0.0017 < replay.cp_upper(1, 2700) < 0.0018
