from build_episode import build_episode
from replay import replay
from rules import seq_accept
from run_block import BlockResult

STREAM = [
    {"fix_id": "bad", "code": "BAD", "agent_failed": 1},
    {"fix_id": "good", "code": "GOOD", "agent_failed": 0},
    {"fix_id": "bad", "code": "BAD", "agent_failed": 1},  # same fix proposed again
]


def runner(calls):
    def run(code, seed):
        calls.append((code, seed))
        return BlockResult(1 if code == "BAD" else 0, "", 0.5)

    return run


def test_each_unique_fix_is_run_once_and_never_twice_with_the_same_seed():
    calls = []
    ep = build_episode(STREAM, p0=0.0, run=runner(calls), n_truth=10, n_pool=10)
    assert ep["stream"] == ["bad", "good", "bad"]
    assert [c for c, _ in calls].count("GOOD") == 20
    assert len({s for _, s in calls}) == len(calls)


def test_good_fix_gets_separate_truth_and_pool_runs():
    ep = build_episode(STREAM, p0=0.0, run=runner([]), n_truth=10, n_pool=15)
    assert len(ep["fixes"]["good"]["truth"]) == 10 and len(ep["fixes"]["good"]["pool"]) == 15


def test_a_clearly_broken_fix_stops_after_40_blocks_and_reuses_them_as_its_pool():
    calls = []
    ep = build_episode(STREAM, p0=0.0, run=runner(calls), n_truth=200, n_pool=200)
    assert [c for c, _ in calls].count("BAD") == 40
    assert ep["fixes"]["bad"]["pool"] == ep["fixes"]["bad"]["truth"] == [1] * 40


def test_the_built_episode_replays():
    ep = build_episode(STREAM, p0=0.0, run=runner([]), n_truth=200, n_pool=200)
    out = replay(seq_accept, ep, reps=5, seed=0)
    assert out["far"] == 0.0 and out["frr"] == 0.0


def test_every_blocks_seed_is_saved_in_run_order():
    calls = []
    ep = build_episode(STREAM, p0=0.0, run=runner(calls), n_truth=10, n_pool=10)
    seeds = {c: [s for code, s in calls if code == c] for c in ("BAD", "GOOD")}
    assert ep["fixes"]["good"]["seeds"] == seeds["GOOD"] and ep["fixes"]["bad"]["seeds"] == seeds["BAD"]


def test_block_seconds_and_parameters_are_saved_for_the_budget_and_time_to_accept_columns():
    ep = build_episode(STREAM, p0=0.0, run=runner([]), n_truth=10, n_pool=10)
    assert ep["fixes"]["good"]["seconds"] == [0.5] * 20 and len(ep["fixes"]["bad"]["seconds"]) == 20
    assert ep["params"] == {"n_truth": 10, "n_pool": 10, "probe": 20, "probe_frac": 0.75}


def test_progress_log_reports_each_fix():
    lines = []
    build_episode(STREAM, p0=0.0, run=runner([]), n_truth=10, n_pool=10, log=lines.append)
    assert any("fix 1/2 bad" in l and "clearly broken" in l for l in lines)
    assert any("fix 2/2 good" in l and "truth failure rate 0.000" in l for l in lines)
