"""Overnight, resumable run of the pilot (plan weeks 2-3), then screening of every IDoFT repo/commit (plan days 5-7).

  mkdir -p runs && caffeinate -i uv run --env-file .env python run_all.py 2>&1 | tee -a runs/overnight.log

Stop it any time (Ctrl-C, sleep, power off) and run the same command again:
  - finished jobs are skipped (runs/ledger.json: job -> status, times, error),
  - an interrupted block job resumes at its next block (runs/blocks/<job>.jsonl: one line per finished block),
  - an interrupted agent stream is recorded again from its first attempt,
  - a failed job is retried on the next run, until it has failed 3 times (stops do not count); its error stays in the
    ledger and the night goes on.

Stages:
  1. per task: collect its test ids (blocks are random subsets of them, Def. 4), p0 over 400 blocks, bug check over
     100 blocks (buggy must fail > p0 + 0.05); ordinary-bug tasks also get injected-flakiness variants
     tasks/<name>@bernoulli.json, @markov.json (§8) that reuse the base task's streams
  2. record task x model x run streams (one LLM call at a time) while
  3. the evidence blocks of finished streams run, `--workers` containers at once
  4. replay.py -> results/ (Table 1-3 numbers, charts, results.json)
  5. screen every IDoFT repo/commit, `--screen-parallel` at once -> screening/<slug>.json, screening/summary.csv

Options: --models azure:gpt-4.1-mini[,kind:model,...] (default: azure:$LLM_DEPLOYMENT_NAME)  --runs 2  --workers 8
         --screen-parallel 4  --idoft ../../idoft/py-data.csv  --no-screen  --only-screen  --tasks t1,t2 (task stems)
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from collect import (LLM_ENV, block_size, collect_test_ids, episode_file, hidden_runs, hidden_split, log, measure, now,
                     record_stream, stream_name, write_json)
from run_block import DockerDown
from screen_idoft import screen, shortlist, slug, write_summary

RUNS = Path("runs")
MAX_TRIES = 3
INJECT = {  # §8 injected flakiness with known parameters; applied to the ordinary-bug tasks, which tests catch every time
    "bernoulli": {"mode": "bernoulli", "false_pass": 0.3, "false_fail": 0.001},
    "markov": {"mode": "markov", "enter": 0.05, "stay": 0.8, "false_pass": 0.6, "false_fail": 0.005},
}
SWEEP = {  # D17: Bernoulli false-pass rates around INJECT's 0.3, reported apart from the main table
    "bernoulli10": {"mode": "bernoulli", "false_pass": 0.1, "false_fail": 0.001},
    "bernoulli60": {"mode": "bernoulli", "false_pass": 0.6, "false_fail": 0.001},
}
# D18: streams grok-4.7 cannot record (freezegun's 26 KB file plus ten attempt pairs exceeds its 80k tokens per minute)
# llama3.1:8b (D22): freezegun's 26 KB file needs ~130k tokens of history, more than the 96k window
NOT_RECORDED = {"freezegun-grok-4.7-1", "freezegun@b7-grok-4.7-1", "freezegun-llama3.1_8b-1", "freezegun-llama3.1_8b-2"}
B7 = False  # D21 withdrawn on 2026-11-06 (no time for the slow grok streams): no new <task>@b7 streams
SWEEP_SKIP = set()  # kanren (8.6 s per block) was skipped at first and added on 2026-11-05


class Ledger:
    """Job states in one JSON file, written atomically after every change so a stop never corrupts it."""

    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()
        self.jobs = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self):
        write_json(self.path, self.jobs, indent=1)

    def run(self, name, fn):
        """Run fn() once unless already done or out of tries. True when the job is done."""
        with self.lock:
            st = self.jobs.get(name, {})
            if st.get("status") == "done":
                log(f"skip {name}: done at {st['finished']}")
                return True
            if st.get("failures", 0) >= MAX_TRIES:
                log(f"skip {name}: failed {st['failures']} times, last error: {st.get('error')}")
                return False
            # "running" left behind by a stop is not a failure: only exceptions count towards MAX_TRIES
            self.jobs[name] = {**st, "status": "running", "started": now(), "starts": st.get("starts", 0) + 1}
            self.save()
        log(f"start {name} (start {self.jobs[name]['starts']}, {self.jobs[name].get('failures', 0)} failures so far)")
        try:
            fn()
            update = {"status": "done", "finished": now(), "error": None}
        except DockerDown:
            raise  # not the job's fault: the whole run stops, the job stays "running" and resumes next time
        except Exception as e:  # recorded, and the rest of the night goes on
            update = {"status": "failed", "finished": now(), "error": f"{type(e).__name__}: {e}"[:800],
                      "failures": self.jobs[name].get("failures", 0) + 1}
            log(f"FAILED {name}: {update['error']}\n{traceback.format_exc(limit=3)}")
        with self.lock:
            self.jobs[name].update(update)
            self.save()
        if update["status"] == "done":
            log(f"done {name}")
        return update["status"] == "done"


def stream_finished(path):
    return path.exists() and json.loads(path.read_text())["meta"].get("finished") is not None


def collect_subsets(ledger, task_path):
    """Def. 4 blocks: collect the task's test ids once and fix the block size; p0 was measured on the whole suite, so
    it is kept as p0_full_suite and measured again on subsets."""
    def job():
        task = json.loads(Path(task_path).read_text())
        ids = collect_test_ids(task)
        task.update(test_ids=ids, block_size=block_size(len(ids)), p0_full_suite=task.get("p0"), p0=None)
        write_json(task_path, task, indent=2)
        log(f"  {len(ids)} tests, {task['block_size']} per block")

    return ledger.run(f"collect:{Path(task_path).stem}", job)


def make_variants(task_path):
    """Injected-flakiness copies of an ordinary-bug task (same image, bug and test ids; own p0; base task's streams)."""
    base = json.loads(Path(task_path).read_text())
    if base["bug_kind"] != "ordinary":
        return []
    paths, stem = [], Path(task_path).stem
    sweep = {} if stem in SWEEP_SKIP or "_b" in stem else SWEEP  # D17 sweep on the first bugs only (D19)
    for name, inject in {**INJECT, **sweep}.items():
        path = Path(task_path).with_name(f"{stem}@{name}.json")
        if not path.exists():
            write_json(path, {**base, "inject": inject, "stream_from": stem, "p0": None,
                              "flakiness": f"injected-{name}"}, indent=2)
        paths.append(str(path))
    return paths


def make_b7(task_path):
    """B7 (§9.2, D21): a copy of a first-bug task whose agent sees only the visible two thirds of the tests. It records
    its own streams (1 run); evidence comes from the base task's full suite plus runs of the hidden third."""
    base = json.loads(Path(task_path).read_text())
    visible, hidden = hidden_split(base["test_ids"])
    if not visible or not hidden:
        return None
    path = Path(task_path).with_name(f"{Path(task_path).stem}@b7.json")
    write_json(path, {**base, "test_ids": visible, "hidden_ids": hidden, "block_size": min(base["block_size"], len(visible)),
                      "runs": 1, "b7_of": task_path, "flakiness": "b7"}, indent=2)  # p0: the base task's, same suite
    return str(path)


def prepare(ledger, task_path, workers):
    """Stage 1: p0 if missing, then the bug check. True when the task may be recorded."""
    name = Path(task_path).stem
    if "hidden_ids" in json.loads(Path(task_path).read_text()):
        return True  # B7 copy: p0 and the bug check belong to its base task, which is ready
    if json.loads(Path(task_path).read_text()).get("p0") is None:
        if not ledger.run(f"p0:{name}", lambda: measure(task_path, "original", 400, save_p0=True, workers=workers,
                                                        checkpoint=RUNS / "blocks" / f"p0-{name}.jsonl")):
            return False

    def bug_check():
        p0 = json.loads(Path(task_path).read_text())["p0"]
        if p0 > 0.05:
            raise RuntimeError(f"p0 = {p0:.3f} > 0.05 (plan 2.3): deselect more flaky tests or drop the task")
        rate = measure(task_path, "buggy", 100, workers=workers, checkpoint=RUNS / "blocks" / f"bug-{name}.jsonl")
        if rate <= p0 + 0.05:
            raise RuntimeError(f"bug not detected: buggy fails {rate:.3f} <= p0 {p0:.3f} + 0.05 (plan 2.4)")

    return ledger.run(f"bugcheck:{name}", bug_check)


def experiment(ledger, task_paths, models, runs, workers):
    """Stages 1-3. Each model records its streams one at a time (models in parallel: separate deployments); each
    finished stream's blocks run while recording goes on. Injected variants run their own blocks on the base task's
    streams."""
    bases = [t for t in task_paths if "@" not in Path(t).stem and collect_subsets(ledger, t)]
    tasks = bases + [v for t in bases for v in make_variants(t)]
    ready = [t for t in tasks if prepare(ledger, t, workers)]
    if B7:
        ready += [b for b in (make_b7(t) for t in ready if "@" not in Path(t).stem and "_b" not in Path(t).stem) if b]  # D21
    log(f"{len(ready)}/{len(tasks)} tasks ready: {[Path(t).stem for t in ready]}")
    record_pools = {model: ThreadPoolExecutor(1) for _, model in models}
    block_pool, recorded, waiting = ThreadPoolExecutor(1), {}, {}

    def order(t):  # base tasks first (variants wait for their streams), then small files first (slow replies last)
        return "@" in t, len(Path(json.loads(Path(t).read_text())["buggy"]).read_text())

    for task_path in sorted(ready, key=order):
        task = json.loads(Path(task_path).read_text())
        base = task.get("stream_from", Path(task_path).stem)
        for kind, model in models:
            for rep in range(1, task.get("runs", runs) + 1):
                stream = Path(f"streams/{stream_name(base, model, rep)}.json")
                name = stream_name(task_path, model, rep)
                episode = Path(f"episodes/{name}.json")
                if stream.stem in NOT_RECORDED:
                    log(f"skip {name}: D18, the model's token limit blocks this stream")
                    continue

                def rec(task_path=task_path, kind=kind, model=model, rep=rep, stream=stream):
                    status = json.loads(Path(task_path).read_text())["report_status"]
                    if status.startswith(("DRAFT", "pending")):
                        raise RuntimeError(f"the bug report is {status!r} (plan 2.4): write it before recording")
                    if not stream_finished(stream):
                        record_stream(task_path, kind, model, rep)

                def blk(task_path=task_path, task=task, stream=stream, episode=episode, name=name):
                    if not episode.exists():  # a B7 copy is measured on its base task's full suite
                        episode_file(task.get("b7_of", task_path), stream, workers=workers,
                                     checkpoint=RUNS / "blocks" / f"{name}.jsonl", out=episode)
                    if "hidden_ids" in task:
                        ep = json.loads(episode.read_text())
                        if ep.get("flakiness") != "b7" or any("hidden" not in f for f in ep["fixes"].values()):
                            write_json(episode, {**ep, "flakiness": "b7"})
                            hidden_runs(task, stream, episode, workers=workers)

                if "stream_from" not in task:
                    recorded[stream] = record_pools[model].submit(ledger.run, f"record:{stream.stem}", rec)
                done = recorded.get(stream)
                if done is None:
                    log(f"skip {name}: base task {base} is not ready, so its streams do not exist")
                    continue
                waiting.setdefault(done, []).append((name, blk))
    pending = []
    for done in as_completed(waiting):  # a stream's block jobs queue only once it is recorded: slow models stall nothing
        if done.result():
            pending += [block_pool.submit(ledger.run, f"blocks:{n}", b) for n, b in waiting[done]]
    for f in pending:
        f.result()
    if any(Path("episodes").glob("*.json")):
        ledger.run(f"replay:{now()}", lambda: subprocess.run([sys.executable, str(Path(__file__).with_name("replay.py")), "episodes"], check=True))


def screening(ledger, idoft_csv, parallel):
    """Stage 5: every shortlisted IDoFT repo/commit, `parallel` at once (2 containers each)."""
    rows = shortlist(Path(idoft_csv).read_text())
    out = Path("screening")
    out.mkdir(exist_ok=True)
    log(f"screening {len(rows)} IDoFT repo/commit pairs, {parallel} at once")

    def one(row):
        path = out / f"{slug(row)}.json"

        def job():
            rec = screen(row)
            write_json(path, rec, indent=1)
            log(f"  {rec['slug']}: {rec['verdict']}")

        ledger.run(f"screen:{slug(row)}", job)

    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(one, rows))
    records = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]
    write_summary(records, out / "summary.csv")
    kept = sum(r["verdict"] == "candidate" for r in records)
    log(f"screening: {kept} candidates of {len(records)} screened -> {out / 'summary.csv'}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=f"azure:{os.environ.get('LLM_DEPLOYMENT_NAME', 'gpt-4.1-mini')}")
    ap.add_argument("--runs", type=int, default=2)  # plan 2.6: 2 runs per bug and model
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--screen-parallel", type=int, default=4)
    ap.add_argument("--idoft", default="../../idoft/py-data.csv")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--only-screen", action="store_true")
    a = ap.parse_args(argv)
    RUNS.mkdir(exist_ok=True)
    lock = open(RUNS / "run_all.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # held until this process exits, however it exits
    except BlockingIOError:
        sys.exit("another run_all.py is already running in this folder: wait for it, or stop it first")
    if subprocess.run(["docker", "info"], capture_output=True).returncode:
        sys.exit("Docker is not running: start Docker Desktop, then run the same command again")
    ledger = Ledger(RUNS / "ledger.json")
    models = [tuple(m.split(":", 1)) for m in a.models.split(",")]
    tasks = sorted(str(p) for p in Path("tasks").glob("*.json")
                   if "@" not in p.stem and (not a.tasks or p.stem in a.tasks.split(",")))
    log(f"run_all: tasks {[Path(t).stem for t in tasks]}, models {models}, {a.runs} runs, {a.workers} workers, "
        f"{sum(v.get('status') == 'done' for v in ledger.jobs.values())} jobs already done")
    if not a.only_screen:
        missing = [k for k in LLM_ENV if not os.environ.get(k)]
        if missing:
            sys.exit(f"{missing} not set: run with `uv run --env-file .env python run_all.py`")
        experiment(ledger, tasks, models, a.runs, a.workers)
    if not a.no_screen:
        screening(ledger, a.idoft, a.screen_parallel)
    failed = [k for k, v in ledger.jobs.items() if v.get("status") == "failed"]
    log(f"run_all finished; {len(failed)} failed jobs{': ' + ', '.join(failed) if failed else ''} (see {RUNS / 'ledger.json'})")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        log("stopped: run the same command again to continue from the checkpoint")
        os._exit(130)  # worker threads hold containers; exit now, the ledger and block files are already on disk
    except DockerDown as e:
        log(f"Docker stopped responding ({e}). Start Docker Desktop, then run the same command again.")
        os._exit(1)
