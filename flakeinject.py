"""Injected flakiness with known parameters (paper §8, the calibration subset): a pytest plugin that flips test outcomes.

Loaded inside the container with `-p flakeinject` (run_block mounts this file when the task has "inject"). Settings come
from the environment, so every block is reproducible from its seed:
  FLAKE_MODE=bernoulli  every test may flip, independently
  FLAKE_MODE=markov     a hidden flaky state that the tests walk through in run order (enter FLAKE_ENTER, stay
                        FLAKE_STAY); only tests run while the state is flaky may flip (bursty flakiness)
  FLAKE_FALSE_PASS      chance that a failing test is reported as passed (a broken patch gets lucky)
  FLAKE_FALSE_FAIL      chance that a passing test is reported as failed (a correct patch gets unlucky)
  FLAKE_SEED            block seed
Only the test call is flipped: a patch that breaks import or setup still fails every block.
"""
import os
import random

import pytest

MODE = os.environ.get("FLAKE_MODE", "bernoulli")
FALSE_PASS = float(os.environ.get("FLAKE_FALSE_PASS", "0"))
FALSE_FAIL = float(os.environ.get("FLAKE_FALSE_FAIL", "0"))
ENTER = float(os.environ.get("FLAKE_ENTER", "0"))
STAY = float(os.environ.get("FLAKE_STAY", "0"))
rng = random.Random(os.environ.get("FLAKE_SEED", "0"))
state = {"flaky": MODE == "bernoulli"}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call":
        return
    if MODE == "markov":
        state["flaky"] = rng.random() < (STAY if state["flaky"] else ENTER)
    u = rng.random()  # drawn for every test, so the sequence depends on the seed only
    if not state["flaky"]:
        return
    if rep.failed and u < FALSE_PASS:
        rep.outcome, rep.longrepr = "passed", None
    elif rep.passed and u < FALSE_FAIL:
        rep.outcome, rep.longrepr = "failed", "injected flaky failure (flakeinject)"
