"""Real-flaky arm with disjoint seeds per loop (external review 2). The high-p0 tasks are re-graded so that no two
loops share block seeds, then SeqAccept is replayed on them and compared with the shared-seed episodes.
No LLM calls; Docker must be running. Resumable per task (runs/blocks/<variant>-disjoint.jsonl).

  uv run python flaky_seeds.py run [--tasks penman yamicache] [--workers 8]   # then: uv run python flaky_seeds.py analyze
"""
import argparse
import json
from functools import partial
from pathlib import Path

import flaky_arm
import replay
from build_episode import build_episode
from collect import cached, log, logged, make_run, pool_map, write_json
from rules import all_rules

OUT = Path("episodes_flaky_seeds")
HIGH = ["penman", "penman_b2", "yamicache", "yamicache_b2"]  # the tasks with p0 >= 0.2 in the real-flaky arm
SEED_STRIDE = 1_000_000  # loop k uses seeds from (k + 1) * SEED_STRIDE + 1: disjoint across loops and from the arm


def loops(bugs):
    """(bug, stream path, episode name, loop index) for every recorded natural loop of the given bugs, in a fixed order."""
    every = [(b, s, n) for b, s, n in flaky_arm.jobs() if b in HIGH]  # index over all high-p0 loops: fixed seeds
    return [(b, s, n, k) for k, (b, s, n) in enumerate(every) if b in bugs]


def grade(bug, stream, name, k, workers):
    variant = Path(f"tasks/{bug}@realflaky.json")
    task = json.loads(variant.read_text())
    run = cached(make_run(task), f"runs/blocks/{variant.stem}-disjoint.jsonl")
    ep = build_episode(json.loads(Path(stream).read_text())["stream"], task["p0"], logged(run, "evidence", every=50),
                       log=log, pmap=pool_map(workers), first_seed=(k + 1) * SEED_STRIDE + 1)
    write_json(OUT / f"{name}.json", {**ep, "repo": task["repo"], "flakiness": "real-flaky", "seed_stride": SEED_STRIDE})


def compare(names):
    """SeqAccept and B4 on the same loops with shared seeds (episodes_flaky) and with disjoint seeds."""
    out = {}
    for tag, folder in (("shared", flaky_arm.OUT), ("disjoint", OUT)):
        eps = [replay.load_episode(Path(folder) / f"{n}.json") for n in names]
        res = {}
        for i, ep in enumerate(eps):
            rules = all_rules(eps=ep["p0"] + replay.MARGIN)
            for rn in ("B4 peeking Clopper-Pearson", "SeqAccept"):
                res.setdefault(rn, []).append(replay.replay(rules[rn], ep, replay.REPS, seed=replay.episode_seed(ep, i)))
        out[tag] = {rn: {k: replay.summarize(rs, eps)[k] for k in ("far", "frr", "blocks_per_accept")} for rn, rs in res.items()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--tasks", nargs="+", default=HIGH, choices=HIGH)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    todo = loops(a.tasks)
    if a.cmd == "run":
        for i, (b, s, n, k) in enumerate(todo, 1):
            if (OUT / f"{n}.json").exists():
                continue
            log(f"[{i}/{len(todo)}] {n}: grading with disjoint seeds")
            grade(b, s, n, k, a.workers)
        log("flaky_seeds: done")
    else:
        names = [n for _, _, n, _ in todo if (OUT / f"{n}.json").exists()]
        out = {"loops": len(names), "tasks": sorted({b for b, _, n, _ in todo if n in names}), **compare(names)}
        print(json.dumps(out, indent=2))
        Path("results/flaky_seeds.json").write_text(json.dumps(out, indent=2))
        print("saved results/flaky_seeds.json")
