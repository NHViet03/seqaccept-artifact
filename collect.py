"""Week 2-3 glue: measure failure rates, record agent streams, turn them into replayable episodes (plan 2.9).

A task is a JSON file describing one planted bug (paths are relative to where you run the command):
  {"repo": "owner/name", "repo_kind": "OD" | "NOD", "bug_kind": "order" | "ordinary",
   "image": "seqaccept/name:1", "relpath": "pkg/mod.py", "tests": ["tests/test_mod.py", ...],
   "report": "2-3 sentence bug report", "buggy": "bugs/name/mod.py", "original": "bugs/name/mod.orig.py",
   "p0": 0.02}   # background rate of the original code, measured with `rate TASK original 400`

Usage:
  uv run --env-file .env python collect.py check [TASK ...] [--llm]   # preflight: run before anything long
  uv run python collect.py rate   TASK buggy|original N [--default-order] [--save-p0] [--workers N]  # p0, bug check
  uv run python collect.py record TASK azure|ollama MODEL RUN              # -> streams/<name>.json
  uv run python collect.py blocks TASK streams/<name>.json [--workers N]   # -> episodes/<name>.json
`record` is kept apart from `blocks` so paid agent calls are saved before any block runs.
Add `--workers N` to `rate` / `blocks` to run N containers at once; `run_all.py` drives everything overnight.
"""
import hashlib
import itertools
import json
import os
import platform
import random
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from agent import run_stream
from build_episode import build_episode
from run_block import BlockError, BlockResult, DockerDown, run_block

AGENT_SEED0 = 10**6  # the agent's own CI block for attempt i uses seed AGENT_SEED0 + i; evidence seeds start at 1
MAX_BLOCK_SECONDS = 15  # plan days 6-7: one block takes <= ~15 s
LLM_ENV = ("LLM_ENDPOINT", "LLM_API_KEY", "LLM_DEPLOYMENT_NAME")
TEMPERATURE = 0.8  # plan 2.7


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def logged(run, label, total=None, every=25):
    """Wrap run(code, seed) to log progress every `every` blocks: count, failures, seconds per block, ETA."""
    state, lock = {"n": 0, "failed": 0, "t0": time.monotonic()}, threading.Lock()

    def wrapped(code, seed):
        res = run(code, seed)
        with lock:
            state["n"] += 1
            state["failed"] += res.failed
            n = state["n"]
        if n % every == 0 or n == total:
            per = (time.monotonic() - state["t0"]) / n
            eta = f", ~{per * (total - n) / 60:.1f} min left" if total else ""
            log(f"  {label}: {n}{f'/{total}' if total else ''} blocks, {state['failed']} failed, {per:.1f} s/block{eta}")
        return res

    return wrapped


def block_size(n_tests):
    """§5.5: blocks of 8-32 tests; a small suite gets at most half its tests so a block is still a sample of it."""
    return max(1, min(32, max(8, n_tests // 4), n_tests // 2))


def subset(task, seed):
    """Def. 4: the block's random subset S of the task's tests, drawn from the block seed (order: pytest-randomly)."""
    ids = task["test_ids"]
    return sorted(random.Random(f"subset-{seed}").sample(ids, task["block_size"]))


def collect_test_ids(task, runner=subprocess.run):
    """Node ids of the task's test selection (deselects applied), collected once on the original code."""
    script = (f"cp /candidate /repo/{task['relpath']} && cd /repo && python -m pytest -q --collect-only "
              "-p no:cacheprovider -p no:randomly " + " ".join(map(shlex.quote, task["tests"])))
    cmd = ["docker", "run", "--rm", "--network", "none", "-v", f"{Path(task['original']).resolve()}:/candidate:ro",
           task["image"], "sh", "-c", script]
    out = runner(cmd, capture_output=True, text=True, timeout=600).stdout
    ids = [line.strip() for line in out.splitlines() if "::" in line and not line.startswith(" ")]
    if not ids:
        raise RuntimeError(f"no tests collected for {task['image']}: {out[-500:]}")
    return ids


def make_run(task, default_order=False, runner=subprocess.run, tries=3):
    """run(code, seed) -> BlockResult: one block of `code` in a fresh container (Def. 4).

    With "test_ids" in the task the block runs a random subset of them (Def. 4); with "inject" the outcomes get the
    task's injected flakiness (§8). default_order runs the whole selection in its own order (Prop. 10 demo).
    An infrastructure error is never evidence (D7): the same block (same seed, new container) is retried, then raised.
    """
    def run(code, seed):
        tests = subset(task, seed) if "test_ids" in task and not default_order else task["tests"]
        with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
            f.write(code)
            f.flush()
            os.chmod(f.name, 0o644)  # readable by a non-root user in the container
            for attempt in range(tries):
                try:
                    return run_block(task["image"], task["relpath"], f.name, tests, seed,
                                     runner=runner, default_order=default_order, inject=task.get("inject"))
                except DockerDown:
                    raise  # retrying cannot help: stop the run
                except BlockError as e:
                    if attempt == tries - 1:
                        raise
                    log(f"  infrastructure error, retrying seed {seed} in a new container: {str(e)[:120]}")
    return run


def cached(run, path):
    """Checkpoint every block of run(code, seed) in a JSONL file keyed by (code hash, seed). A resumed job replays the
    finished blocks from the file (same seeds, so the same experiment) and only runs the missing ones."""
    path, lock, done = Path(path), threading.Lock(), {}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done[(r["code"], r["seed"])] = BlockResult(r["failed"], "", r["seconds"])
            except (ValueError, KeyError):
                pass  # a line cut short when the machine stopped: that block runs again

    def wrapped(code, seed):
        key = (hashlib.sha1(code.encode()).hexdigest()[:12], seed)
        if key in done:
            return done[key]
        res = run(code, seed)
        with lock:
            with path.open("a") as f:
                f.write(json.dumps({"code": key[0], "seed": seed, "failed": res.failed, "seconds": round(res.seconds, 3)}) + "\n")
            done[key] = res
        return res

    wrapped.resumed = len(done)
    return wrapped


def pool_map(workers):
    """map() over `workers` containers at once (order kept); plain map for 1."""
    return map if workers <= 1 else ThreadPoolExecutor(workers).map


def fail_rate(run, code, n, pmap=map):
    return sum(r.failed for r in pmap(lambda seed: run(code, seed), range(1, n + 1))) / n


def record(task, llm, runner=subprocess.run, max_attempts=15, on_attempt=None):
    run, seeds = make_run(task, runner=runner), itertools.count(AGENT_SEED0)
    code = Path(task["buggy"]).read_text()
    return run_stream(llm, lambda c: run(c, next(seeds)), {**task, "code": code}, max_attempts, on_attempt)


def image_id(image, runner=subprocess.run):
    """Container digest saved with the episode (§8 release)."""
    cmd = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    return runner(cmd, capture_output=True, text=True, check=True).stdout.strip()


def check_task(task, runner=subprocess.run, n=10):
    """Preflight one task against the plan's screening rules (days 6-8): installs, runs offline in both orders,
    no missing test dependency, block <= 15 s, p0 measured and <= 0.05, bug detected. Empty "problems" = ready."""
    out = {"problems": [f"missing file {task[k]}" for k in ("buggy", "original") if not Path(task[k]).is_file()]}
    if out["problems"]:
        return out
    try:
        image_id(task["image"], runner)
    except subprocess.CalledProcessError:
        return {"problems": [f"image {task['image']} not built (docker build -t {task['image']} repos/<repo>)"]}
    original, buggy = (Path(task[k]).read_text() for k in ("original", "buggy"))
    run = make_run(task, runner=runner)
    try:
        start = time.monotonic()
        orig = [run(original, seed) for seed in range(1, n + 1)]
        out["seconds_per_block"] = (time.monotonic() - start) / n
        out["buggy_rate"] = fail_rate(run, buggy, n)
        make_run(task, default_order=True, runner=runner)(original, 1)  # Prop. 10 demo path
    except BlockError as e:
        return {"problems": [f"a block cannot run: {e}"]}
    out["p0_estimate"] = p0 = mean(r.failed for r in orig)
    missing = sorted({m for r in orig for m in re.findall(r"No module named '([^']+)'", r.output)})
    if missing:
        out["problems"].append(f"tests import uninstalled modules {missing}: add to the Dockerfile or deselect them")
    if task["p0"] is None:
        out["problems"].append(f"p0 not measured: collect.py rate TASK original 400 --save-p0 (estimate {p0:.2f})")
    elif task["p0"] > 0.05:
        out["problems"].append(f"p0 = {task['p0']:.3f} > 0.05: drop the repo or deselect its flaky tests (plan 2.3)")
    else:
        p0 = task["p0"]
    if out["buggy_rate"] <= p0 + 0.05:
        out["problems"].append(f"bug not detected: buggy fails {out['buggy_rate']:.2f} <= p0 + 0.05 (plan 2.4)")
    if out["seconds_per_block"] > MAX_BLOCK_SECONDS:
        out["problems"].append(f"{out['seconds_per_block']:.0f} s per block > {MAX_BLOCK_SECONDS} s (plan day 6)")
    return out


def check_llm(task, llm, runner=subprocess.run):
    """One real agent attempt on the task: the reply must hold a ```python block about as long as the file."""
    stream = record(task, llm, runner=runner, max_attempts=1)
    if not stream:
        return ["the model's reply had no ```python block: every attempt would be skipped"]
    if len(stream[0]["code"]) < 0.5 * len(Path(task["buggy"]).read_text()):
        return ["the model's reply is under half the file's length: it is truncating the file"]
    return []


def write_json(path, obj, indent=None):
    """Write via a temporary file and an atomic rename: a stop mid-write never leaves a half-written file behind."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent) + "\n")
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hardware(runner=subprocess.run):
    """Hardware profile saved with every episode (§14: wall-clock results are reported with the hardware used)."""
    cmd = ["docker", "info", "--format", "{{json .}}"]
    info = json.loads(runner(cmd, capture_output=True, text=True, check=True).stdout)
    keep = ("NCPU", "MemTotal", "ServerVersion", "OperatingSystem", "Architecture", "KernelVersion")
    return {"host": platform.platform(), "host_cpu": platform.processor(), **{k: info.get(k) for k in keep}}


def blocks(task, stream_doc, runner=subprocess.run, workers=1, checkpoint=None):
    if task.get("p0") is None:
        raise ValueError("task has no p0: measure it first with `collect.py rate TASK original 400`")
    started = now()
    run = make_run(task, runner=runner)
    if checkpoint:
        run = cached(run, checkpoint)
        log(f"  checkpoint {checkpoint}: {run.resumed} blocks already done")
    ep = build_episode(stream_doc["stream"], task["p0"], logged(run, "evidence", every=50), log=log,
                       pmap=pool_map(workers))
    meta = {k: task[k] for k in ("repo", "repo_kind", "bug_kind")}
    meta.update(flakiness=task.get("flakiness") or (f"injected-{task['inject']['mode']}" if task.get("inject") else "natural"),
                inject=task.get("inject"), block_size=task.get("block_size"), n_tests=len(task.get("test_ids", [])))
    return {**ep, **meta, "model": stream_doc["model"], "agent": stream_doc.get("meta"),
            "image": image_id(task["image"], runner), "hardware": {**hardware(runner), "workers": workers},
            "blocks_started": started, "blocks_finished": now()}


def measure(task_path, which, n, save_p0=False, default_order=False, workers=1, checkpoint=None):
    """Failure rate of the task's original or buggy code over n blocks; save_p0 writes it into the task as p0."""
    task = json.loads(Path(task_path).read_text())
    log(f"{task_path}: {n} blocks of the {which} code, {'default' if default_order else 'random'} order, {workers} at once")
    run = make_run(task, default_order=default_order)
    if checkpoint:
        run = cached(run, checkpoint)
        log(f"  checkpoint {checkpoint}: {run.resumed} blocks already done")
    rate = fail_rate(logged(run, which, total=n, every=max(1, n // 20)), Path(task[which]).read_text(), n,
                     pmap=pool_map(workers))
    log(f"{which}: failure rate {rate:.3f} over {n} blocks")
    if save_p0 and which == "original":
        saved = {**task, "p0": rate, "p0_blocks": n, "p0_workers": workers, "p0_measured_at": now()}
        write_json(task_path, saved, indent=2)
        log(f"p0 = {rate:.3f} saved to {task_path}")
    return rate


def stream_name(task_path, model, rep):
    return f"{Path(task_path).stem}-{model.replace(':', '_').replace('/', '_')}-{rep}"


def record_stream(task_path, kind, model, rep, max_attempts=15):
    """Record one agent stream to streams/<name>.json, saved after every attempt ("finished": null until done)."""
    import llm

    task = json.loads(Path(task_path).read_text())
    name = stream_name(task_path, model, rep)
    meta = {"kind": kind, "deployment": model, "api_version": os.environ.get("LLM_API_VERSION"),
            "temperature": TEMPERATURE, "max_attempts": max_attempts, "agent_seed0": AGENT_SEED0, "task": task_path,
            "started": now()}  # §14: exact model identifiers and access dates
    Path("streams").mkdir(exist_ok=True)
    out, got, t0 = Path(f"streams/{name}.json"), [], time.monotonic()

    def progress(i, entry):  # log and save after every attempt, so an interrupted run keeps what it paid for
        if entry is not None:
            got.append(entry)
            write_json(out, {"model": model, "meta": {**meta, "finished": None}, "stream": got})
        what = "no code in reply, skipped" if entry is None else (
            f"CI {'FAILED' if entry['agent_failed'] else 'passed'}, fix {entry['fix_id']}")
        log(f"  attempt {i:2}/{max_attempts}: {what}  ({time.monotonic() - t0:.0f} s since start)")

    log(f"recording {name}: {max_attempts} attempts, each reply rewrites {task['relpath']} (can take a minute each)")
    stream = record(task, getattr(llm, kind)(model, temperature=TEMPERATURE), max_attempts=max_attempts,
                    on_attempt=progress)
    write_json(out, {"model": model, "meta": {**meta, "finished": now()}, "stream": stream})
    log(f"{out}: {len(stream)} attempts, {len({c['fix_id'] for c in stream})} unique fixes")
    return out


def episode_file(task_path, stream_path, workers=1, checkpoint=None, out=None):
    """Evidence blocks for one recorded stream -> `out` (default episodes/<stream name>.json)."""
    task, doc = json.loads(Path(task_path).read_text()), json.loads(Path(stream_path).read_text())
    log(f"{stream_path}: {len(doc['stream'])} attempts, {len({c['fix_id'] for c in doc['stream']})} unique fixes, "
        f"up to 400 blocks each (clearly broken fixes stop at 40), {workers} containers at once")
    ep = blocks(task, doc, workers=workers, checkpoint=checkpoint)
    Path("episodes").mkdir(exist_ok=True)
    out = Path(out or f"episodes/{Path(stream_path).name}")
    write_json(out, ep)
    log(f"{out}: {len(ep['fixes'])} unique fixes")
    return out


def preflight(task_paths, with_llm):
    """`check`: every task, then (with --llm) the LLM settings and one real agent call per task. True if all ready."""
    ok = True
    for path in task_paths:
        log(f"checking {path}: image, 10 original + 10 buggy blocks, 1 default-order block")
        out = check_task(json.loads(Path(path).read_text()))
        stats = ", ".join(f"{k} {v:.2f}" for k, v in out.items() if k != "problems")
        print(f"{'OK  ' if not out['problems'] else 'FAIL'} {path}  {stats}")
        for p in out["problems"]:
            print(f"     - {p}")
        ok &= not out["problems"]
    if with_llm:
        missing = [k for k in LLM_ENV if not os.environ.get(k)]
        if missing:
            print(f"FAIL LLM: {missing} not set (run with `uv run --env-file .env ...`)")
            return False
        try:
            import llm
        except ImportError:
            print("FAIL LLM: openai not installed (uv sync --extra llm)")
            return False
        model = llm.azure(os.environ["LLM_DEPLOYMENT_NAME"])
        for path in task_paths:
            log(f"asking {os.environ['LLM_DEPLOYMENT_NAME']} for one fix of {path} (can take a minute)")
            problems = check_llm(json.loads(Path(path).read_text()), model)
            print(f"{'OK  ' if not problems else 'FAIL'} LLM on {path}" + "".join(f"\n     - {p}" for p in problems))
            ok &= not problems
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "check":
        paths = [a for a in sys.argv[2:] if not a.startswith("--")] or sorted(map(str, Path("tasks").glob("*.json")))
        sys.exit(0 if preflight(paths, "--llm" in sys.argv) else 1)
    cmd, task_path = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    if cmd == "rate":
        measure(task_path, sys.argv[3], int(sys.argv[4]), save_p0="--save-p0" in sys.argv,
                default_order="--default-order" in sys.argv, workers=workers)
    elif cmd == "record":
        record_stream(task_path, sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "blocks":
        episode_file(task_path, sys.argv[3], workers=workers)
    else:
        sys.exit(__doc__)


HIDDEN_SEED0 = 5 * 10**6  # B7 hidden-suite runs, apart from every other seed range
HIDDEN_RUNS = 20  # per distinct fix; B7 replays draw one with replacement, like the block pool (D6)


def hidden_split(ids):
    """B7 (§9.2): a fixed third of the test ids (by hash, so it never changes) is the held-out suite the agent never
    sees; the rest stay visible. Returns (visible, hidden)."""
    hidden = [t for t in ids if zlib.crc32(t.encode()) % 3 == 0]
    return [t for t in ids if t not in set(hidden)], hidden


def hidden_runs(task, stream_path, episode_path, n=HIDDEN_RUNS, workers=1, runner=subprocess.run):
    """B7 evidence: every distinct fix of the episode runs the WHOLE hidden suite n times (fresh container, random
    order); saved as fixes[f]["hidden"] (1 = failed) and ["hidden_seconds"] in the episode file."""
    ep = json.loads(Path(episode_path).read_text())
    codes = {c["fix_id"]: c["code"] for c in json.loads(Path(stream_path).read_text())["stream"]}
    suite = {k: v for k, v in task.items() if k not in ("test_ids", "hidden_ids")}
    run = make_run({**suite, "tests": task["hidden_ids"]}, runner=runner)
    todo = [(f, i) for f in ep["fixes"] if "hidden" not in ep["fixes"][f] for i in range(n)]
    with ThreadPoolExecutor(max(1, workers)) as ex:
        res = list(ex.map(lambda fi: run(codes[fi[0]], HIDDEN_SEED0 + fi[1]), todo))
    for (f, _), r in zip(todo, res):
        ep["fixes"][f].setdefault("hidden", []).append(r.failed)
        ep["fixes"][f].setdefault("hidden_seconds", []).append(r.seconds)
    write_json(episode_path, ep)
    log(f"{episode_path}: hidden suite ({len(task['hidden_ids'])} tests) run {n} times on {len(todo) // n} fixes")
