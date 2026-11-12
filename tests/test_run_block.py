import subprocess

import pytest

from run_block import MARKER, BlockError, DockerDown, run_block


def fake_runner(returncode, stdout="", raises=None, calls=None):
    def run(cmd, **kw):
        if calls is not None:
            calls.append((cmd, kw))
        if raises:
            raise raises
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{stdout}\n{MARKER}{returncode}\n", stderr="")

    return run


def block(runner, **kw):
    return run_block("img:1", "pkg/mod.py", "/tmp/cand.py", ["tests/test_a.py::test_x"], seed=7, runner=runner, **kw)


def test_all_tests_passing_is_a_passing_block():
    assert block(fake_runner(0)).failed == 0


@pytest.mark.parametrize("code", [1, 2, 3])  # failed, collection error (broken import), internal error (patch broke pytest)
def test_failing_tests_and_a_patch_that_cannot_even_be_imported_both_count_as_failures(code):
    assert block(fake_runner(code, stdout="E   AssertionError")).failed == 1


@pytest.mark.parametrize("code, stdout", [(4, "ERROR: usage: pytest [options]"), (4, "error: unrecognized arguments: -x"),
                                          (5, "no tests ran")])
def test_infrastructure_problems_are_errors_not_silent_failures(code, stdout):
    with pytest.raises(BlockError):
        block(fake_runner(code, stdout=stdout))


@pytest.mark.parametrize("stdout", ["ERROR: not found: /repo/t.py::test_x", "ERROR: found no collectors for /repo/t.py",
                                    "ERROR tests/test_a.py\n24 errors in 0.17s"])
def test_a_candidate_that_makes_selected_tests_uncollectable_fails_the_block(stdout):
    assert block(fake_runner(4, stdout=stdout)).failed == 1


@pytest.mark.parametrize("code", [1, 125, 137])  # daemon refused, docker error, OOM kill: pytest never reported
def test_a_docker_failure_is_never_counted_as_a_failing_test(code):
    run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, code, stdout="", stderr="docker: error")
    with pytest.raises(BlockError):
        block(run)


def test_docker_daemon_down_is_its_own_error():
    msg = "Cannot connect to the Docker daemon at unix:///x/docker.sock. Is the docker daemon running?"
    with pytest.raises(DockerDown):
        block(lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr=msg))


def test_a_hung_test_run_counts_as_a_failure_and_its_container_is_killed():
    calls = []
    res = block(fake_runner(0, raises=subprocess.TimeoutExpired("docker", 1), calls=calls))
    assert res.failed == 1
    run_cmd, kill_cmd = calls[0][0], calls[1][0]
    assert kill_cmd[:2] == ["docker", "kill"] and kill_cmd[2] == run_cmd[run_cmd.index("--name") + 1]


def test_a_candidate_that_runs_away_with_memory_or_time_fails_only_its_own_block():
    res = block(fake_runner(137, stdout="..."))  # pytest killed inside the container, marker still printed
    assert res.failed == 1 and "killed" in res.output


def test_each_container_has_a_memory_cap_and_an_in_container_timeout():
    calls = []
    block(fake_runner(0, calls=calls))
    cmd = calls[0][0]
    assert cmd[cmd.index("--memory") + 1] == cmd[cmd.index("--memory-swap") + 1] == "2g"
    assert "timeout -s KILL 180 python -m pytest" in cmd[-1]


def test_every_block_uses_a_fresh_offline_container_and_the_given_random_seed():
    calls = []
    block(fake_runner(0, calls=calls))
    cmd = " ".join(calls[0][0])
    assert "--rm" in cmd and "--network none" in cmd
    assert "--randomly-seed=7" in cmd and "tests/test_a.py::test_x" in cmd
    assert "/tmp/cand.py:/candidate:ro" in cmd


def test_only_the_tail_of_the_test_output_is_kept_for_the_agent():
    res = block(fake_runner(1, stdout="x" * 10_000 + "THE END"))
    assert res.output.endswith("THE END") and len(res.output) <= 3000


def test_default_order_disables_randomization_for_the_order_dependence_demo():
    calls = []
    block(fake_runner(0, calls=calls), default_order=True)
    script = calls[0][0][-1]
    assert "-p no:randomly" in script and "--randomly-seed" not in script


def test_test_ids_with_brackets_or_spaces_are_quoted_for_the_shell():
    calls = []
    run_block("img:1", "pkg/mod.py", "/tmp/c.py", ["t.py::test_x[a b]", "--deselect=t.py::y[(1, 2)]"], seed=1,
              runner=fake_runner(0, calls=calls))
    assert "'t.py::test_x[a b]' '--deselect=t.py::y[(1, 2)]'" in calls[0][0][-1]


def test_screening_runs_the_repo_as_is_without_a_candidate_file():
    calls = []
    run_block("img:1", None, None, ["t.py"], seed=1, runner=fake_runner(0, calls=calls))
    assert "-v" not in calls[0][0] and "cp /candidate" not in calls[0][0][-1]
