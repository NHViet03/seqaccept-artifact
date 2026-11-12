import json
import subprocess

from run_block import MARKER

import pytest

from collect import cached
from run_all import MAX_TRIES, Ledger
from run_block import BlockResult, DockerDown
from screen_idoft import screen


def test_ten_jobs_stopped_after_the_fifth_continue_at_the_sixth(tmp_path):
    ran, stop = [], [True]

    def job(i):
        if i == 6 and stop:
            stop.clear()
            raise KeyboardInterrupt  # the machine stops while job 6 runs, once
        ran.append(i)

    ledger = Ledger(tmp_path / "ledger.json")
    with pytest.raises(KeyboardInterrupt):
        for i in range(1, 11):
            ledger.run(f"job{i}", lambda i=i: job(i))
    assert ran == [1, 2, 3, 4, 5]

    ledger = Ledger(tmp_path / "ledger.json")  # next run, fresh process
    for i in range(1, 11):
        ledger.run(f"job{i}", lambda i=i: job(i))
    assert ran == list(range(1, 11))  # 1-5 skipped, 6-10 ran once each


def test_a_failing_job_is_retried_on_later_runs_then_given_up_and_the_rest_still_runs(tmp_path):
    calls = []

    def bad():
        calls.append(1)
        raise RuntimeError("docker down")

    for _ in range(MAX_TRIES + 2):
        ledger = Ledger(tmp_path / "ledger.json")
        assert ledger.run("bad", bad) is False
        assert ledger.run("good", lambda: None) is True
    st = json.loads((tmp_path / "ledger.json").read_text())
    assert len(calls) == MAX_TRIES and st["bad"]["failures"] == MAX_TRIES and "docker down" in st["bad"]["error"]


def test_stops_do_not_count_as_failures(tmp_path):
    for _ in range(MAX_TRIES + 1):
        with pytest.raises(KeyboardInterrupt):
            Ledger(tmp_path / "l.json").run("long", lambda: (_ for _ in ()).throw(KeyboardInterrupt))
    assert Ledger(tmp_path / "l.json").run("long", lambda: None) is True


def test_block_checkpoint_resumes_without_rerunning_finished_blocks(tmp_path):
    calls = []

    def run(code, seed):
        calls.append(seed)
        return BlockResult(seed % 2, "", 1.0)

    path = tmp_path / "blocks.jsonl"
    first = cached(run, path)
    assert [first("x", s).failed for s in range(1, 6)] == [1, 0, 1, 0, 1]
    with path.open("a") as f:
        f.write('{"code": "cut')  # the machine stopped mid-line
    again = cached(run, path)
    assert again.resumed == 5
    assert [again("x", s).failed for s in range(1, 8)] == [1, 0, 1, 0, 1, 0, 1]
    assert calls == [1, 2, 3, 4, 5, 6, 7]  # 1-5 came from the file
    assert again("other code", 1).failed == 1 and calls[-1] == 1  # another fix's seed 1 is a different block


ROW = {"url": "https://github.com/a/B", "sha": "abcdef123456", "tests": ["t.py::test_x", "t.py::test_y"], "kinds": ["OD"]}


def docker(flaky_fails, clean_code=0, build_ok=True, calls=None):
    """Fake docker: random-order runs with the IDoFT tests fail on odd seeds if flaky_fails; deselected runs exit clean_code."""
    def run(cmd, **kw):
        if calls is not None:
            calls.append(cmd)
        if cmd[1] == "build":
            if not build_ok:
                raise subprocess.CalledProcessError(1, cmd, stderr="pip failed")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[1] == "rmi":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        script = cmd[-1]
        if "--deselect" in script:
            return subprocess.CompletedProcess(cmd, 0, f"{MARKER}{clean_code}", "")
        odd = "no:randomly" not in script and int(script.split("--randomly-seed=")[1].split()[0]) % 2
        out = "FAILED t.py::test_x - x\n" if odd else ""
        return subprocess.CompletedProcess(cmd, 0, f"{out}{MARKER}{int(bool(flaky_fails and odd))}", "")
    return run


def test_screening_keeps_a_repo_whose_flake_reproduces_with_a_clean_background(tmp_path):
    calls = []
    rec = screen(ROW, runner=docker(True, calls=calls), workers=1, n=4, root=tmp_path)
    assert rec["verdict"] == "candidate" and rec["flaky_rate"] == 0.5 and rec["p0_estimate"] == 0
    assert rec["default_order_failed"] == 0 and rec["failing_tests"] == ["t.py::test_x"]  # Prop. 10 on a real suite
    assert not any(c[1] == "rmi" for c in calls) and (tmp_path / "a_b_abcdef12" / "Dockerfile").is_file()


@pytest.mark.parametrize("kw, verdict", [
    ({"flaky_fails": False}, "flake not reproduced"),
    ({"flaky_fails": True, "build_ok": False}, "build failed"),
    ({"flaky_fails": True, "clean_code": 1}, "p0 > 0.05 without the flaky tests"),
    ({"flaky_fails": True, "clean_code": 5}, "no tests left without the flaky ones"),
])
def test_screening_verdicts_and_rejected_images_are_removed(tmp_path, kw, verdict):
    calls = []
    rec = screen(ROW, runner=docker(calls=calls, **kw), workers=1, n=4, root=tmp_path)
    assert rec["verdict"] == verdict
    assert any(c[1] == "rmi" for c in calls) or verdict == "build failed"


def test_docker_going_down_stops_the_run_without_counting_a_failure(tmp_path):
    def job():
        raise DockerDown("Cannot connect to the Docker daemon")

    with pytest.raises(DockerDown):
        Ledger(tmp_path / "l.json").run("blocks:x", job)
    st = json.loads((tmp_path / "l.json").read_text())["blocks:x"]
    assert st["status"] == "running" and st.get("failures", 0) == 0  # resumes next time, not a failure


def test_a_second_run_all_in_the_same_folder_refuses_to_start(tmp_path, monkeypatch):
    import fcntl

    import run_all

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_all, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    held = open(tmp_path / "runs" / "run_all.lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the first run holds the lock
    with pytest.raises(SystemExit, match="already running"):
        run_all.main(["--only-screen"])
