"""Real-flaky arm (review round 4): the recorded natural streams re-run with the repository's IDoFT flaky tests kept.

No LLM calls: candidates come from the recorded streams. Docker must be running. Each natural bug gets a task variant
tasks/<bug>@realflaky.json whose selection keeps the tests IDoFT lists as flaky (other deselects stay), with its own
test ids, block size (same rule) and p0 on 400 blocks of the original code. Every recorded natural loop of that bug is
then graded on the variant (same build_episode as the main table) and written to episodes_flaky/, apart from the main
episodes. Resumable: blocks are checkpointed per task, so a candidate shared by several loops runs once.

  uv run python flaky_arm.py [--workers 8] 2>&1 | tee -a runs/flaky_arm.log      # then: uv run python extra_analyses.py
"""
import argparse
import json
from pathlib import Path

import replay
from collect import block_size, blocks, collect_test_ids, log, measure, write_json

P0_BLOCKS = 400
OUT = Path("episodes_flaky")
CHECKPOINTS = Path("runs/blocks")


def idoft_tests(repo, screening=Path("screening")):
    """Every test IDoFT lists as flaky for this repository, over all its screened commits."""
    url = f"https://github.com/{repo}".lower()
    return {t for p in screening.glob("*.json") for rec in [json.loads(p.read_text())]
            if rec.get("url", "").lower() == url for t in rec.get("tests", [])}


def flaky_selection(tests, flaky):
    """The task's pytest arguments without the deselects of known flaky tests (other deselects stay)."""
    return [x for x in tests if not (x.startswith("--deselect=") and x.split("=", 1)[1] in flaky)]


def make_variant(base_path, collect=collect_test_ids):
    """tasks/<bug>@realflaky.json, or None when the task deselects no known flaky test."""
    base_path = Path(base_path)
    path = base_path.with_name(f"{base_path.stem}@realflaky.json")
    if path.exists():
        return path
    base = json.loads(base_path.read_text())
    tests = flaky_selection(base["tests"], idoft_tests(base["repo"]))
    if tests == base["tests"]:
        return None
    task = {**base, "tests": tests, "stream_from": base_path.stem, "flakiness": "real-flaky", "p0": None}
    task["test_ids"] = collect(task)
    task["block_size"] = block_size(len(task["test_ids"]))
    write_json(path, task, indent=2)
    return path


def jobs():
    """(base task, stream file, episode name) for every recorded natural loop of the main table."""
    out = []
    for p in sorted(Path("episodes").glob("*.json")):
        if replay.excluded(p) or json.loads(p.read_text()).get("flakiness", "natural") != "natural":
            continue
        variant, rest = p.stem.split("-", 1)
        out.append((variant, Path(f"streams/{variant}-{rest}.json"), f"{variant}@realflaky-{rest}"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    todo = jobs()
    variants = {b: make_variant(f"tasks/{b}.json") for b in sorted({b for b, _, _ in todo})}
    log(f"real-flaky arm: {sum(v is not None for v in variants.values())} of {len(variants)} bugs have known flaky "
        f"tests to keep; {len(todo)} natural loops")
    for b, v in variants.items():
        if v and json.loads(v.read_text())["p0"] is None:
            measure(str(v), "original", P0_BLOCKS, save_p0=True, workers=a.workers,
                    checkpoint=CHECKPOINTS / f"{v.stem}-p0.jsonl")
    for b, stream, name in todo:
        out = OUT / f"{name}.json"
        if variants[b] is None or out.exists():
            continue
        log(f"{name}: grading")
        task = json.loads(variants[b].read_text())
        ep = blocks(task, json.loads(stream.read_text()), workers=a.workers, checkpoint=CHECKPOINTS / f"{variants[b].stem}.jsonl")
        write_json(out, ep)
    log("real-flaky arm: done")
