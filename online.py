"""§12 online subset (D20): SeqAccept in the loop instead of offline replay.

The agent proposes a fix; SeqAccept (rules.seq_accept, unchanged) tests it on fresh blocks and either accepts it, which
ends the episode, or rejects it (or hits the cap), and the agent gets that evidence as feedback and tries again.
Candidate j is only written when SeqAccept moves on to j, so the policy shapes the stream, as in deployment.
The accepted fix is then labelled on 200 fresh truth blocks (§9.1). Episodes run concurrently; a shared semaphore keeps
at most SLOTS containers alive, so this can run beside run_all.py (8 containers) without exhausting Docker's memory.

  uv run --env-file .env python online.py [--models gpt-4.1-mini,gpt-4o,grok-4.7,ollama:llama3.1:8b] [--slots 4] 2>&1 | tee -a runs/online.log
Results: online/<task>-<model>.json ("finished": null until done; an interrupted episode is run again from scratch).
"""
import argparse
import hashlib
import itertools
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent import SYSTEM, extract_code
from collect import TEMPERATURE, log, make_run, now, write_json
from rules import seq_accept
from sim import label

MAX_ATTEMPTS = 15  # same budget of agent replies as the recorded streams
TRUTH_BLOCKS = 200  # §9.1 truth set for the accepted fix
MARGIN = 0.05  # eps = p0 + 0.05 (D4)
ONLINE_SEED0, TRUTH_SEED0 = 3 * 10**6, 4 * 10**6  # apart from the recorded streams' seeds
SLOTS = 4  # containers this script may run at once (run_all.py uses 8; 12 x 2 GB stays inside the Docker VM)
DEFAULT_MODELS = ("gpt-4.1-mini", "gpt-4o")
NO_FREEZEGUN = ("grok-4.7", "ollama:llama3.1:8b")  # freezegun's history exceeds grok's 80k tokens/min and llama's 96k window (D18, D22)


class OutOfAttempts(Exception):
    """The agent used all its replies before SeqAccept accepted anything."""


def online_episode(task, llm, run, max_attempts=MAX_ATTEMPTS, truth_blocks=TRUTH_BLOCKS, on_event=log):
    """One online episode. llm(messages) -> str; run(code, seed) -> BlockResult. Returns the episode record."""
    p0, code0 = task["p0"], task["code"]
    eps = p0 + MARGIN
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Bug report:\n{task['report']}\n\nFile {task['relpath']}:\n```python\n{code0}```"},
    ]
    cands, replies, attempts, seeds = [], [], 0, itertools.count(ONLINE_SEED0)

    def propose():
        nonlocal attempts
        if cands:  # the previous fix was rejected or hit the cap: its evidence is the agent's feedback
            c = cands[-1]
            out = c["last_fail_output"] or "(no failing run)"
            messages.extend([{"role": "assistant", "content": replies[-1]},
                             {"role": "user", "content": f"The tests FAILED in {c['fails']} of {c['blocks']} runs, so the "
                                                         f"fix was not accepted. Output of a failing run:\n{out}\n\nTry again."}])
        while attempts < max_attempts:
            attempts += 1
            reply = llm(messages)
            code = extract_code(reply)
            if code is None:
                on_event(f"  attempt {attempts:2}/{max_attempts}: no code in reply, skipped")
                continue
            replies.append(reply)
            cands.append({"fix_id": hashlib.sha1(code.encode()).hexdigest()[:12], "code": code, "attempt": attempts,
                          "blocks": 0, "fails": 0, "seconds": 0.0, "last_fail_output": ""})
            on_event(f"  attempt {attempts:2}/{max_attempts}: candidate {len(cands)} ({cands[-1]['fix_id']})")
            return
        raise OutOfAttempts

    def block(j):
        while j >= len(cands):
            propose()
        c = cands[j]
        res = run(c["code"], next(seeds))
        c["blocks"] += 1
        c["fails"] += res.failed
        c["seconds"] += res.seconds
        if res.failed:
            c["last_fail_output"] = res.output
        return res.failed

    try:
        accepted = seq_accept(max_attempts, block, eps=eps, refund="uncapped").accepted  # the rule the recorded online runs used (D20, D24)
    except OutOfAttempts:
        accepted = None
    out = {"p0": p0, "eps": eps, "attempts": attempts, "accepted": accepted, "label": None, "truth_rate": None,
           "candidates": [{k: v for k, v in c.items() if k != "last_fail_output"} for c in cands]}
    if accepted is not None:
        code = cands[accepted]["code"]
        truth = [run(code, TRUTH_SEED0 + i).failed for i in range(truth_blocks)]
        out["truth_rate"] = sum(truth) / len(truth)
        out["label"] = label(out["truth_rate"], p0)
    return out


def online_tasks():
    """The online subset: every main-table task (10 natural, 7 Bernoulli, 7 Markov), one run per model."""
    return sorted(str(p) for p in Path("tasks").glob("*.json")
                  if "_b" not in p.stem and not p.stem.endswith(("@bernoulli10", "@bernoulli60", "@b7")))


def main(argv=None):
    import llm as llm_mod

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--slots", type=int, default=SLOTS)
    a = ap.parse_args(argv)
    slots = threading.Semaphore(a.slots)
    Path("online").mkdir(exist_ok=True)

    def one(task_path, model):
        name = f"{Path(task_path).stem}-{model.replace('ollama:', '').replace(':', '_')}"
        path = Path(f"online/{name}.json")
        if path.exists() and json.loads(path.read_text()).get("finished"):
            log(f"skip online:{name}: done")
            return
        task = json.loads(Path(task_path).read_text())
        task["code"] = Path(task["buggy"]).read_text()
        raw = make_run(task)

        def run(code, seed):
            with slots:
                return raw(code, seed)

        meta = {"task": task_path, "model": model, "temperature": TEMPERATURE, "max_attempts": MAX_ATTEMPTS,
                "flakiness": task.get("flakiness") or ("injected-" + task["inject"]["mode"] if task.get("inject") else "natural"),
                "repo": task["repo"], "started": now()}
        write_json(path, {**meta, "finished": None})
        log(f"online:{name}: start")
        client = llm_mod.ollama(model[7:], temperature=TEMPERATURE) if model.startswith("ollama:") else llm_mod.azure(model, temperature=TEMPERATURE)
        ep = online_episode(task, client, run,
                            on_event=lambda m: log(f"{name}{m}"))
        write_json(path, {**meta, **ep, "finished": now()})
        log(f"online:{name}: done, accepted {ep['accepted']} ({ep['label']}), "
            f"{sum(c['blocks'] for c in ep['candidates'])} blocks, {ep['attempts']} attempts")

    jobs = [(t, m) for m in a.models.split(",") for t in online_tasks()
            if not (m in NO_FREEZEGUN and Path(t).stem.split("@")[0] == "freezegun")]
    log(f"online: {len(jobs)} episodes, {a.slots} containers at once")
    with ThreadPoolExecutor(max_workers=2 * a.slots) as ex:
        for f in [ex.submit(one, t, m) for t, m in jobs]:
            try:
                f.result()
            except Exception as e:  # one failed episode must not stop the others; rerun picks it up
                log(f"online episode failed: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
