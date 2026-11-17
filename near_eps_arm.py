"""Natural near-eps arm (EASE round 6): the recorded natural streams re-run with smaller blocks, so that broken
candidates fail close to eps on flake-free code.

No LLM calls: candidates come from the recorded streams. Docker must be running. Each candidate's deterministically
failing tests are inferred from its recorded blocks (sensitivity.failing_set); a candidate that always fails k of a
task's N tests fails a random block of b tests with probability 1 - C(N-k, b)/C(N, b). For every natural bug we pick
the block size b' (below the recorded one) that puts the most distinct candidates within 3 points above
eps = p0 + 0.05, and keep the bug if at least --min-near (default 10) are predicted to land there. Its variant tasks/<bug>@neareps.json has the same tests
and image with block size b' and its own p0 on 400 blocks of the original code; every recorded natural loop of the bug
is then graded on it (same build_episode as the main table) into episodes_neareps/. Blocks are checkpointed per
task, so the run resumes after an interruption and a candidate shared by several loops runs once.

  uv run python near_eps_arm.py --plan [--min-near 10]                        # selection and cost, no Docker
  uv run python near_eps_arm.py [--workers 8] [--min-near 10] 2>&1 | tee -a runs/near_eps_arm.log
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import replay
from collect import blocks, log, measure, write_json
from flaky_arm import jobs
from sensitivity import failing_set, hit

P0_BLOCKS = 400
NEAR = 0.03  # "near eps": within 3 points above it, as in the paper
OUT = Path("episodes_neareps")
CHECKPOINTS = Path("runs/blocks")
PROBE, PROBE_FRAC, PER_FIX = 40, 0.75, 400  # build_episode defaults: clearly broken fixes stop at 40 blocks


def candidates(bug):
    """Distinct candidates of a bug's natural loops: {fix_id: failing set or None}, with the task."""
    task = json.loads(Path(f"tasks/{bug}.json").read_text())
    ids, b = task["test_ids"], task["block_size"]
    out = {}
    for p in sorted(Path("episodes").glob(f"{bug}-*.json")):
        if replay.excluded(p):
            continue
        ep = json.loads(p.read_text())
        if ep.get("flakiness", "natural") != "natural":
            continue
        for f, fx in ep["fixes"].items():
            out.setdefault(f, failing_set(fx, ids, b))
    return task, out


def choose(task, fsets):
    """(b', predicted rates, near count): the block size below the recorded one with the most near-eps candidates
    (ties to the larger block)."""
    N, eps = len(task["test_ids"]), task["p0"] + replay.MARGIN
    best = None
    for b in range(1, task["block_size"]):
        rates = {f: (hit(N, len(F), b) if F else 0.0) for f, F in fsets.items() if F is not None}
        near = sum(eps < r <= eps + NEAR for r in rates.values())
        if best is None or near >= best[2]:
            best = (b, rates, near)
    return best


def plan(min_near=10):
    rows = []
    for bug in sorted({b for b, _, _ in jobs()}):
        task, fsets = candidates(bug)
        b, rates, near = choose(task, fsets)
        if near < max(1, min_near):
            continue
        loops = [j for j in jobs() if j[0] == bug]
        n_blocks = P0_BLOCKS + sum(PROBE if rates.get(f, 0.0) >= PROBE_FRAC else PER_FIX for f in fsets)
        secs = mean(s for p in Path("episodes").glob(f"{bug}-*.json") if not replay.excluded(p)
                    for fx in json.loads(p.read_text())["fixes"].values() for s in fx.get("seconds", []))
        rows.append({"bug": bug, "n_tests": len(task["test_ids"]), "block": task["block_size"], "new_block": b,
                     "candidates": len(fsets), "unexplained": sum(F is None for F in fsets.values()), "near": near,
                     "loops": len(loops), "blocks": n_blocks, "seconds_per_block": secs})
    return rows


def make_variant(bug, b):
    path = Path(f"tasks/{bug}@neareps.json")
    if path.exists():
        return path
    base = json.loads(Path(f"tasks/{bug}.json").read_text())
    task = {**base, "block_size": b, "stream_from": bug, "flakiness": "near-eps", "p0": None,
            "block_size_recorded": base["block_size"]}
    write_json(path, task, indent=2)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plan", action="store_true", help="print the selection and the expected cost; run nothing")
    ap.add_argument("--min-near", type=int, default=10, help="keep a bug with at least this many predicted near-eps candidates")
    a = ap.parse_args()
    rows = plan(a.min_near)
    total = sum(r["blocks"] for r in rows)
    hours = sum(r["blocks"] * r["seconds_per_block"] for r in rows) / a.workers / 3600
    for r in rows:
        log(f"{r['bug']}: {r['n_tests']} tests, block {r['block']} -> {r['new_block']}; {r['near']} of {r['candidates']} "
            f"candidates predicted near eps ({r['unexplained']} unexplained); {r['loops']} loops, ~{r['blocks']} blocks")
    log(f"near-eps arm: {len(rows)} bugs, ~{total} blocks, ~{hours:.1f} h at {a.workers} containers "
        f"(recorded seconds per block; smaller blocks are a little faster)")
    if a.plan:
        raise SystemExit(0)
    OUT.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    keep = {r["bug"]: make_variant(r["bug"], r["new_block"]) for r in rows}
    for bug, v in keep.items():
        if json.loads(v.read_text())["p0"] is None:
            measure(str(v), "original", P0_BLOCKS, save_p0=True, workers=a.workers,
                    checkpoint=CHECKPOINTS / f"{v.stem}-p0.jsonl")
    for bug, stream, name in jobs():
        out = OUT / f"{name.replace('@realflaky', '@neareps')}.json"
        if bug not in keep or out.exists():
            continue
        log(f"{out.stem}: grading")
        task = json.loads(keep[bug].read_text())
        ep = blocks(task, json.loads(stream.read_text()), workers=a.workers, checkpoint=CHECKPOINTS / f"{keep[bug].stem}.jsonl")
        write_json(out, ep)
    log("near-eps arm: done")
