"""Assumption 4.1 in execution order (external review 2): run fresh blocks of each task's original code, in parallel
and serially, logging start and end time, worker and host load, then test independence in execution order.
No LLM calls; Docker must be running. Resumable: finished blocks are kept in runs/exec_order/<task>-<mode>.jsonl.

  uv run python exec_order.py run [--workers 8]      # then: uv run python exec_order.py analyze
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

from collect import log, make_run, pool_map
from extra_analyses import lag1_corr, runs_test

OUT = Path("runs/exec_order")
TASKS = ["anonapi@realflaky", "filestack@realflaky", "freezegun_b2@realflaky", "qtaf@realflaky",
         "penman@realflaky", "yamicache@realflaky"]  # one task per repository with known flaky tests, p0 > 0
SERIAL = ["penman@realflaky", "yamicache@realflaky"]
N_PARALLEL, N_SERIAL = 400, 150
SEED_BASE = {"parallel": 900_001, "serial": 950_001}  # disjoint from every seed used elsewhere


def done_seeds(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                out[r["seed"]] = r
            except (ValueError, KeyError):
                pass
    return out


def run_task(name, mode, n, workers):
    task = json.loads(Path(f"tasks/{name}.json").read_text())
    code = Path(task["original"]).read_text()
    path = OUT / f"{name}-{mode}.jsonl"
    seeds = [s for s in range(SEED_BASE[mode], SEED_BASE[mode] + n) if s not in done_seeds(path)]
    log(f"{name} {mode}: {n - len(seeds)} of {n} blocks done, running {len(seeds)} with {workers} at once")
    run, lock = make_run(task), threading.Lock()

    def one(seed):
        load = os.getloadavg()[0]
        start = time.time()
        res = run(code, seed)
        rec = {"seed": seed, "failed": res.failed, "start": start, "end": time.time(), "load": load,
               "worker": threading.current_thread().name}
        with lock:
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    for _ in pool_map(workers)(one, seeds):
        pass


def two_prop_z(a, b):
    """z statistic for the difference of two failure rates (0/1 lists); None if either is empty or the pooled rate is 0."""
    if not a or not b:
        return None
    pa, pb, pp = sum(a) / len(a), sum(b) / len(b), (sum(a) + sum(b)) / (len(a) + len(b))
    se = (pp * (1 - pp) * (1 / len(a) + 1 / len(b))) ** 0.5
    return (pa - pb) / se if se else None


def summarize(recs):
    """Independence checks on one task's blocks: runs test and lag-1 autocorrelation in execution (start) order and in
    seed order, how often the two orders disagree (share of adjacent start-order pairs out of seed order), and the
    failure rate when the host load average was above vs at or below its median."""
    by_start = sorted(recs, key=lambda r: r["start"])
    by_seed = sorted(recs, key=lambda r: r["seed"])
    conc = [sum(o["start"] < r["start"] < o["end"] for o in recs) for r in by_start]
    med_load = sorted(r["load"] for r in recs)[len(recs) // 2]
    hi = [r["failed"] for r in recs if r["load"] > med_load]
    lo = [r["failed"] for r in recs if r["load"] <= med_load]
    mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
    ys = [r["failed"] for r in by_start]
    return {"blocks": len(recs), "fail_rate": mean(ys),
            "runs_reject_exec": runs_test(ys), "lag1_exec": lag1_corr(ys),
            "runs_reject_seed": runs_test([r["failed"] for r in by_seed]), "lag1_seed": lag1_corr([r["failed"] for r in by_seed]),
            "order_inversions": sum(a["seed"] > b["seed"] for a, b in zip(by_start, by_start[1:])) / max(1, len(recs) - 1),
            "concurrency_median": sorted(conc)[len(conc) // 2], "load_median": med_load,
            "fail_rate_busy": mean(hi), "fail_rate_quiet": mean(lo), "busy_quiet_z": two_prop_z(hi, lo),
            "outcomes": [r["failed"] for r in by_seed]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        for name in TASKS:
            run_task(name, "parallel", N_PARALLEL, a.workers)
        for name in SERIAL:
            run_task(name, "serial", N_SERIAL, 1)
        log("exec_order: done")
    else:
        out = {}
        for p in sorted(OUT.glob("*.jsonl")):
            recs = list(done_seeds(p).values())
            out[p.stem] = summarize(recs)
            print(p.stem, {k: v for k, v in out[p.stem].items() if k != "outcomes"})
        for name in SERIAL:  # same code, different seeds: does running one at a time change the failure rate?
            par, ser = out.get(f"{name}-parallel"), out.get(f"{name}-serial")
            if par and ser:
                out[f"{name}-serial_vs_parallel_z"] = two_prop_z(ser["outcomes"], par["outcomes"])
                print(name, "serial vs parallel z", out[f"{name}-serial_vs_parallel_z"])
        Path("results/exec_order.json").write_text(json.dumps(out, indent=2))
        print("saved results/exec_order.json")
