"""The injected-flakiness plugin flips outcomes at its known rates, reproducibly per seed (paper §8)."""
import os
import subprocess
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parent.parent


def run(tmp_path, body, **env):
    (tmp_path / "test_x.py").write_text(body)
    e = {**os.environ, "PYTHONPATH": str(IMPL), **{f"FLAKE_{k.upper()}": str(v) for k, v in env.items()}}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
                           "-p", "flakeinject", str(tmp_path)], env=e, capture_output=True, text=True).returncode


FAIL = "def test_a():\n    assert False\n"
PASS = "def test_a():\n    assert True\n"


def test_false_pass_makes_a_failing_test_pass_and_false_fail_the_reverse(tmp_path):
    assert run(tmp_path, FAIL, mode="bernoulli", false_pass=1) == 0
    assert run(tmp_path, PASS, mode="bernoulli", false_fail=1) == 1
    assert run(tmp_path, FAIL, mode="bernoulli") == 1 and run(tmp_path, PASS, mode="bernoulli") == 0


def test_rates_are_honoured_and_each_seed_is_reproducible(tmp_path):
    outcomes = [run(tmp_path, FAIL, mode="bernoulli", false_pass=0.5, seed=s) for s in range(40)]
    assert 8 <= outcomes.count(0) <= 32  # about half the failing runs turn green
    assert outcomes[:10] == [run(tmp_path, FAIL, mode="bernoulli", false_pass=0.5, seed=s) for s in range(10)]


def test_markov_only_flips_inside_the_flaky_state(tmp_path):
    assert run(tmp_path, FAIL, mode="markov", false_pass=1, enter=0, stay=0) == 1  # never flaky
    assert run(tmp_path, FAIL, mode="markov", false_pass=1, enter=1, stay=1) == 0  # always flaky


def test_a_patch_that_breaks_import_still_fails(tmp_path):
    assert run(tmp_path, "import no_such_module\n" + PASS, mode="bernoulli", false_pass=1) == 2
