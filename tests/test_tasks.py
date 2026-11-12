"""Every planted bug is one small edit of the original file, and every task points at files that exist."""
import difflib
import json
from pathlib import Path

import pytest

IMPL = Path(__file__).resolve().parent.parent
TASKS = sorted((IMPL / "tasks").glob("*.json"))
KEYS = {"repo", "repo_kind", "bug_kind", "image", "relpath", "tests", "report", "report_status", "buggy", "original", "p0"}


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.stem)
def test_task_is_complete_and_its_bug_is_a_one_line_change(path):
    task = json.loads(path.read_text())
    extra = {"p0_blocks", "p0_workers", "p0_measured_at", "test_ids", "block_size", "p0_full_suite", "inject", "stream_from", "flakiness", "hidden_ids", "runs", "b7_of"}
    keys = KEYS - ({"p0"} if "test_ids" not in task else set())  # a new task (D19) gets p0 when run_all collects it
    assert keys <= set(task) <= KEYS | extra
    assert task["repo_kind"] in ("OD", "NOD") and task["bug_kind"] in ("order", "ordinary")  # plan 2.4
    base = path.stem.split("@")[0].split("_b")[0]  # injected variants and second bugs (<repo>_b2, D19) share the repo's image
    assert (IMPL / "repos" / base / "Dockerfile").is_file() and task["image"] == f"seqaccept/{base}:1"
    orig, buggy = ((IMPL / task[k]).read_text().splitlines() for k in ("original", "buggy"))
    diff = [l for l in difflib.unified_diff(orig, buggy, lineterm="", n=0) if l[:3] not in ("+++", "---")]
    removed, added = ([l for l in diff if l[:1] == sign and l[1:].strip()] for sign in "-+")
    assert diff and len(removed) <= 2 and len(added) <= 2, diff  # "a small, realistic change": <= 2 source lines (plan 2.4)


def test_the_plan_mix_of_order_sensitive_and_ordinary_bugs():
    kinds = [json.loads(p.read_text())["bug_kind"] for p in TASKS]
    assert "order" in kinds and "ordinary" in kinds
