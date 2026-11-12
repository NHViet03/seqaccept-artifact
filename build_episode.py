"""Turn an agent's recorded stream into a replayable episode (see replay.py for the format).

`run(code, seed) -> BlockResult` runs ONE block (fresh container, random order seeded by `seed`).
Each unique fix is run n_truth + n_pool times: the first n_truth grade it, the rest are what the rules draw from.
A fix that fails >= 75% of its first 40 blocks is clearly broken: stop there and reuse those blocks for both.
Every block's seed and wall-clock seconds are saved with the fix (§8 release: per-block outcomes with seeds), in run
order, and the parameters with the episode.
"""
import itertools


def build_episode(stream, p0, run, n_truth=200, n_pool=200, probe=40, probe_frac=0.75, log=None, pmap=map, first_seed=1):
    """log(msg), if given, reports each fix as it is probed, extended and graded. pmap runs a fix's blocks (e.g. several
    containers at once); results stay in seed order."""
    log = log or (lambda msg: None)
    seeds = itertools.count(first_seed)  # first_seed > 1 gives a loop its own seeds (flaky_seeds.py)
    unique = list(dict.fromkeys(c["fix_id"] for c in stream))
    probe = min(probe, n_truth + n_pool)  # never run more blocks than were asked for
    fixes = {}
    for cand in stream:
        fid = cand["fix_id"]
        if fid in fixes:
            continue
        tag = f"fix {unique.index(fid) + 1}/{len(unique)} {fid}"
        log(f"{tag}: probing {probe} blocks")
        used = [next(seeds) for _ in range(probe)]
        code = cand["code"]
        res = list(pmap(lambda s: run(code, s), used))
        clearly_broken = sum(r.failed for r in res) >= probe_frac * probe
        if clearly_broken:
            log(f"{tag}: clearly broken ({sum(r.failed for r in res)}/{probe} failed), stopping at {probe} blocks")
        else:
            log(f"{tag}: {sum(r.failed for r in res)}/{probe} failed, running {n_truth + n_pool - probe} more")
            more = [next(seeds) for _ in range(n_truth + n_pool - probe)]
            used += more
            res += list(pmap(lambda s: run(code, s), more))
        blocks = [r.failed for r in res]
        fixes[fid] = {
            "truth": blocks if clearly_broken else blocks[:n_truth],
            "pool": blocks if clearly_broken else blocks[n_truth:],
            "seeds": used,
            "seconds": [round(r.seconds, 3) for r in res],
        }
        truth = fixes[fid]["truth"]
        log(f"{tag}: done, truth failure rate {sum(truth) / len(truth):.3f} (p0 {p0:.3f})")
    params = {"n_truth": n_truth, "n_pool": n_pool, "probe": probe, "probe_frac": probe_frac}
    return {"p0": p0, "stream": [c["fix_id"] for c in stream], "fixes": fixes, "params": params}
