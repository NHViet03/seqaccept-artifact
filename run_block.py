"""Run one test block: a brand-new offline container, the candidate file dropped in, tests in a seeded random order.

The image must already contain the repo at /repo (at the IDoFT commit) with pytest and pytest-randomly installed.
"""
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

TAIL = 3000  # chars of test output kept for the agent
PLUGIN = Path(__file__).with_name("flakeinject.py").resolve()  # mounted only for injected-flakiness tasks (§8)
FAILED = {1, 2, 3}  # pytest: 1 tests failed, 2 interrupted (e.g. broken import), 3 internal error (the candidate broke pytest)


MARKER = "__SEQACCEPT_PYTEST_EXIT__="  # printed by the container after pytest: no marker means pytest never ran
MEMORY = "2g"  # per container: a candidate that eats memory must die alone, not take the Docker VM and the other blocks down
KILLED = 137  # pytest SIGKILLed inside the container: over MEMORY or over the block timeout, i.e. the candidate hung or leaked


class BlockError(RuntimeError):
    """The block could not be run at all (docker down, OOM, no tests collected...). Never counted as pass or fail."""


class DockerDown(BlockError):
    """The Docker daemon is unreachable: nothing can run until it is started again."""


@dataclass(frozen=True)
class BlockResult:
    failed: int  # 1 if the block failed, 0 if it passed
    output: str
    seconds: float = 0.0  # wall clock, container start included (Table 2 budget and time-to-accept)


def run_block(image, relpath, candidate_path, tests, seed, timeout=180, runner=subprocess.run, default_order=False,
              inject=None):
    """default_order=True runs the suite in its own order (no randomization): only for the Prop. 10 demo.
    relpath=None runs the repo as it is, with no candidate file (IDoFT screening).
    inject={"mode": ..., "false_pass": ..., ...} loads flakeinject.py with those known parameters, seeded by `seed`."""
    order = "-p no:randomly" if default_order else f"-p randomly --randomly-seed={seed}"
    plugin = "-p flakeinject " if inject else ""
    # the container always exits 0 after pytest and reports pytest's own exit code in MARKER, so a docker failure
    # (daemon down, missing image, killed container) can never be mistaken for a failing test
    script = (
        (f"cp /candidate /repo/{relpath} && " if relpath else "") + "cd /repo || exit 99; "
        f"timeout -s KILL {timeout} python -m pytest -q -p no:cacheprovider {order} {plugin}" + " ".join(map(shlex.quote, tests))
        + f"; echo {MARKER}$?"
    )
    mount = ["-v", f"{candidate_path}:/candidate:ro"] if relpath else []
    if inject:
        mount += ["-v", f"{PLUGIN}:/flakeinject/flakeinject.py:ro", "-e", "PYTHONPATH=/flakeinject",
                  "-e", f"FLAKE_SEED={seed}"] + [x for k, v in inject.items() for x in ("-e", f"FLAKE_{k.upper()}={v}")]
    name = f"seqaccept-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--memory", MEMORY, "--memory-swap", MEMORY,
           *mount, image, "sh", "-c", script]
    start = time.monotonic()
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=timeout + 60)  # the in-container timeout fires first
    except subprocess.TimeoutExpired:
        try:
            runner(["docker", "kill", name], capture_output=True, text=True, timeout=30)  # killing the CLI leaves the container running
        except Exception:
            pass
        return BlockResult(1, "block timed out", timeout)  # a hung run is a failing run
    seconds = time.monotonic() - start
    out, err = proc.stdout or "", proc.stderr or ""
    if "Cannot connect to the Docker daemon" in err:
        raise DockerDown(err.strip()[-300:])
    if MARKER not in out:
        raise BlockError(f"pytest did not run (docker exit code {proc.returncode}): {(err or out)[-500:]}")
    code = int(out.rsplit(MARKER, 1)[1].split()[0])
    out = out.rsplit(MARKER, 1)[0].rstrip("\n")
    if code == 0:
        return BlockResult(0, out[-TAIL:], seconds)
    # 4 without a usage error: the selected test ids (which collect on the original code) no longer collect, i.e. the
    # candidate broke collection ("not found", "found no collectors", "N errors")
    usage_error = "usage:" in out or "unrecognized arguments" in out
    if code in FAILED or (code == 4 and not usage_error):
        return BlockResult(1, out[-TAIL:], seconds)
    if code == KILLED:  # the marker was printed, so docker and the shell are fine: this candidate's tests ran away
        return BlockResult(1, f"killed: out of memory ({MEMORY}) or over {timeout}s\n{out[-TAIL:]}", seconds)
    raise BlockError(f"pytest exit code {code}: {out[-500:]}")
