"""Full-suite arm (review round 2): run each recorded candidate's whole test selection, with no test sampling.

No LLM calls: candidates come from the recorded streams. Docker must be running. Resumable: one result file per
(task, candidate) in fullsuite/, skipped when present. Each candidate gets RUNS full-suite runs in random order
(seeds from SEED0), and natural tasks also one run in the suite's default order (what plain CI does). Injected tasks
keep their injected flakiness, so the arm separates sampling from per-test flips.

  uv run python fullsuite.py [--workers 8] [--runs 5]       # then: uv run python extra_analyses.py
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import replay
from collect import log, make_run, write_json

RUNS = 5
SEED0 = 5_000_001  # apart from block, agent, online and truth seeds
OUT = Path("fullsuite")


def jobs():
    """(task variant, fix id, code) for every distinct candidate of the main-table episodes."""
    out, seen = [], set()
    for p in sorted(Path("episodes").glob("*.json")):
        if replay.excluded(p):
            continue
        ep = json.loads(p.read_text())
        if ep.get("flakiness", "natural") not in replay.MAIN_FLAKINESS:
            continue
        variant, rest = p.stem.split("-", 1)
        model, run = rest.rsplit("-", 1)
        stream = json.loads(Path(f"streams/{variant.split('@')[0]}-{model}-{run}.json").read_text())["stream"]
        code = {c["fix_id"]: c["code"] for c in stream}
        for fid in ep["fixes"]:
            if (variant, fid) not in seen:
                seen.add((variant, fid))
                out.append((variant, fid, code[fid]))
    return out


def run_candidate(task, code, runs=RUNS, factory=make_run):
    """Full-suite outcomes of one candidate: `runs` random-order runs, plus a default-order run on natural tasks."""
    full = {k: v for k, v in task.items() if k != "test_ids"}  # no test_ids: make_run runs the whole selection
    random_run = factory(full)
    res = [random_run(code, SEED0 + i) for i in range(runs)]
    out = {"random": [r.failed for r in res], "seconds": [round(r.seconds, 3) for r in res], "default": None}
    if not task.get("inject"):
        out["default"] = factory(full, default_order=True)(code, SEED0).failed
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--runs", type=int, default=RUNS)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    todo = [(v, f, c) for v, f, c in jobs() if not (OUT / f"{v}__{f}.json").exists()]
    log(f"full suite: {len(todo)} candidates to run, {a.runs} random-order runs each, {a.workers} at once")

    def one(job):
        variant, fid, code = job
        task = json.loads(Path(f"tasks/{variant}.json").read_text())
        write_json(OUT / f"{variant}__{fid}.json", {"task": variant, "fix_id": fid, **run_candidate(task, code, a.runs)})

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 50 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)} candidates done")
