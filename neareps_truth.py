"""Firmer labels near eps (EASE round 8, E8-B): extra truth blocks for every near-eps candidate within 3 points of eps.

No LLM calls; Docker must be running. On 200 truth blocks a candidate failing at 6% cannot be told from one at
eps = 5%, so the near-eps FAR of B5 rests on uncertain labels. This re-runs each such candidate (same task variant,
image and block size as episodes_neareps/) for --extra more blocks on fresh seeds (from 1,000,001, unused by the arm),
and writes results/neareps_truth.json: per task and candidate, the extra blocks' outcomes. The pools the rules draw
from are not touched; near_eps_checks.py relabels with the 200 + extra truth blocks. Blocks are checkpointed in
runs/blocks/<task>-truth.jsonl, so an interrupted run resumes, and a larger --extra later only runs the new seeds.

  uv run python neareps_truth.py --plan
  uv run python neareps_truth.py [--extra 2000] [--workers 8] 2>&1 | tee -a runs/neareps_truth.log
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import replay
from collect import cached, log, logged, make_run, pool_map, write_json
from flaky_arm import jobs

BAND, SEED0 = 0.03, 1_000_001
OUT = Path("results/neareps_truth.json")


def near_candidates():
    """{task: {fix_id: (code, truth rate, mean block seconds)}} for candidates within BAND of eps."""
    code = {}
    for bug, stream, name in jobs():
        for c in json.loads(stream.read_text())["stream"]:
            code.setdefault((bug, c["fix_id"]), c["code"])
    out = {}
    for p in sorted(Path("episodes_neareps").glob("*.json")):
        ep, task = json.loads(p.read_text()), p.stem.split("-")[0]
        eps = ep["p0"] + replay.MARGIN
        for f in ep["stream"]:
            fx = ep["fixes"][f]
            r = mean(fx["truth"])
            if abs(r - eps) <= BAND:
                out.setdefault(task, {})[f] = (code[(task.split("@")[0], f)], r, mean(fx["seconds"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=int, default=2000, help="extra truth blocks per candidate")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plan", action="store_true", help="print the candidates and the expected time; run nothing")
    a = ap.parse_args()
    cands = near_candidates()
    n = sum(map(len, cands.values()))
    secs = sum(s for d in cands.values() for _, _, s in d.values()) * a.extra / a.workers
    log(f"near-eps truth: {n} candidates ({', '.join(f'{t} {len(d)}' for t, d in cands.items())}), {a.extra} extra "
        f"blocks each = {n * a.extra} blocks, ~{secs / 3600:.1f} h at {a.workers} containers (recorded block seconds)")
    if a.plan:
        raise SystemExit(0)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    seeds = list(range(SEED0, SEED0 + a.extra))
    for task, d in cands.items():
        t = json.loads(Path(f"tasks/{task}.json").read_text())
        run = logged(cached(make_run(t), Path(f"runs/blocks/{task}-truth.jsonl")), "truth", total=len(d) * a.extra, every=500)
        for i, (f, (code, r, _)) in enumerate(d.items(), 1):
            if len(out.get(task, {}).get(f, {}).get("extra", [])) >= a.extra:
                continue
            res = list(pool_map(a.workers)(lambda s: run(code, s), seeds))
            fails = [x.failed for x in res]
            out.setdefault(task, {})[f] = {"truth200": r, "extra": fails, "seed0": SEED0,
                                          "seconds": round(mean(x.seconds for x in res), 3)}
            write_json(OUT, out)
            log(f"{task} {i}/{len(d)} {f}: 200-block rate {r:.3f}, {a.extra}-block rate {mean(fails):.4f}")
    log("near-eps truth: done")
