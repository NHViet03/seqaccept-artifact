import subprocess

from run_block import MARKER

from collect import blocks, check_llm, check_task, fail_rate, make_run, record

TASK = {"repo": "a/b", "repo_kind": "OD", "bug_kind": "order", "image": "img:1", "relpath": "pkg/mod.py",
        "tests": ["tests/t.py"], "report": "add() is off by one", "p0": 0.0}


def fake_docker(calls):
    """`docker run` fails the block iff the mounted candidate file contains BAD; `image inspect` returns a digest."""
    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == "image":
            return subprocess.CompletedProcess(cmd, 0, stdout="sha256:abc\n", stderr="")
        if cmd[1] == "info":
            return subprocess.CompletedProcess(cmd, 0, stdout='{"NCPU": 18, "MemTotal": 25}', stderr="")
        path = cmd[cmd.index("-v") + 1].split(":")[0]
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{MARKER}{int('BAD' in open(path).read())}", stderr="")
    return run


def test_the_candidate_code_is_what_the_container_sees():
    run = make_run(TASK, runner=fake_docker([]))
    assert fail_rate(run, "BAD", 5) == 1.0 and fail_rate(run, "GOOD", 5) == 0.0


def test_default_order_is_passed_through_for_the_order_dependence_demo():
    calls = []
    make_run(TASK, default_order=True, runner=fake_docker(calls))("GOOD", 1)
    assert "-p no:randomly" in calls[0][-1]


def test_record_starts_the_agent_on_the_buggy_file_and_keeps_agent_seeds_apart(tmp_path):
    (tmp_path / "mod.py").write_text("BAD")
    calls, seen = [], []
    llm = lambda messages: seen.append(messages[-1]["content"]) or "```python\nGOOD\n```"
    stream = record({**TASK, "buggy": str(tmp_path / "mod.py")}, llm, runner=fake_docker(calls))
    assert "BAD" in seen[0] and len(stream) == 15
    assert "--randomly-seed=1000000" in calls[0][-1]  # evidence blocks use seeds 1, 2, ...


def test_blocks_builds_a_replayable_episode_with_metadata_seeds_and_image_digest():
    stream = [{"fix_id": "bad", "code": "BAD", "agent_failed": 1}, {"fix_id": "good", "code": "GOOD", "agent_failed": 0}]
    ep = blocks(TASK, {"model": "gpt-4.1", "stream": stream}, runner=fake_docker([]))
    assert ep["model"] == "gpt-4.1" and ep["repo_kind"] == "OD" and ep["image"] == "sha256:abc"
    assert ep["fixes"]["bad"]["pool"] == [1] * 40  # clearly broken: probe only
    assert len(ep["fixes"]["good"]["seeds"]) == 400 and len(ep["fixes"]["good"]["seconds"]) == 400
    assert ep["hardware"]["NCPU"] == 18 and ep["blocks_started"] <= ep["blocks_finished"]


def test_blocks_refuses_a_task_whose_background_rate_was_never_measured():
    import pytest

    with pytest.raises(ValueError, match="p0"):
        blocks({**TASK, "p0": None}, {"model": "m", "stream": []}, runner=fake_docker([]))


def files(tmp_path):
    (tmp_path / "orig.py").write_text("GOOD")
    (tmp_path / "bug.py").write_text("BAD")
    return {**TASK, "original": str(tmp_path / "orig.py"), "buggy": str(tmp_path / "bug.py"), "p0": 0.0}


def test_preflight_passes_a_ready_task(tmp_path):
    out = check_task(files(tmp_path), runner=fake_docker([]), n=3)
    assert out["problems"] == [] and out["buggy_rate"] == 1.0


def test_preflight_names_a_missing_test_dependency_and_an_unmeasured_p0(tmp_path):
    def docker(cmd, **kw):
        if cmd[1] == "image":
            return subprocess.CompletedProcess(cmd, 0, stdout="sha256:abc\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=f"E ModuleNotFoundError: No module named 'pandas'\n{MARKER}1", stderr="")

    problems = check_task({**files(tmp_path), "p0": None}, runner=docker, n=2)["problems"]
    assert any("pandas" in p for p in problems) and any("p0 not measured" in p for p in problems)


def test_preflight_reports_an_image_that_was_never_built(tmp_path):
    def docker(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    assert "not built" in check_task(files(tmp_path), runner=docker)["problems"][0]


def test_an_infrastructure_error_is_retried_with_the_same_seed_then_counted_normally():
    calls, codes = [], iter([125, 125, 0])  # docker hiccups twice, then the block runs

    def docker(cmd, **kw):
        calls.append(cmd[-1])
        code = next(codes)
        return subprocess.CompletedProcess(cmd, code, stdout="" if code else f"{MARKER}0", stderr="daemon busy")

    assert make_run(TASK, runner=docker)("GOOD", 7).failed == 0
    assert len(calls) == 3 and all("--randomly-seed=7" in c for c in calls)


def test_preflight_llm_check_catches_replies_without_code(tmp_path):
    assert "no ```python block" in check_llm(files(tmp_path), lambda m: "I can't help", runner=fake_docker([]))[0]
    assert check_llm(files(tmp_path), lambda m: "```py\nGOOD\n```", runner=fake_docker([])) == []
