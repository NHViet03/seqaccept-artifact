"""Cap ablation (2026-11-18): the shared derived cap T is sized for SeqAccept's first slice gamma * alpha. Rules whose
slices shrink (B6, harmonic and Bonferroni spending) need more blocks to reach 1/alpha_j, so under the shared cap they
lose late candidates for want of blocks. This reruns them, and SeqAccept, with each candidate's cap scaled by
log(1/alpha_j) / log(1 / (gamma * alpha)) (never below T; rules.cap_for with SLICE_CAPS), on the recorded natural loops,
the near-eps arm and the late streams, next to the shared cap.

  uv run python slice_caps.py      -> results/slice_caps.json
"""
import json
from functools import partial
from pathlib import Path

import replay
import rules
from extra_analyses import evaluate_rules
from rules import bonferroni, fixed_spending, harmonic, naive_cs, seq_accept

RULES = ("SeqAccept", "B5", "B6", "Harmonic", "Bonferroni-15", "Bonferroni-1000")


def rules_of(ep):
    e = ep["p0"] + replay.MARGIN
    return dict(zip(RULES, (partial(seq_accept, eps=e), partial(naive_cs, eps=e), partial(seq_accept, eps=e, refund=False),
                            partial(fixed_spending, slices=harmonic(), eps=e),
                            partial(fixed_spending, slices=bonferroni(15), eps=e),
                            partial(fixed_spending, slices=bonferroni(1000), eps=e))))


def summary(eps_list):
    out = {}
    for n, rs in evaluate_rules(eps_list, rules_of).items():
        s = replay.summarize(rs, eps_list)
        out[n] = {"far": s["far"], "frr": s["frr"], "blocks_per_accept": s["blocks_per_accept"],
                  "blocks_per_loop": sum(r["blocks"] for r in rs) / sum(r["reps"] for r in rs),
                  "correct_accept": sum(r["correct_accept"] for r in rs) / len(rs)}
    return out


if __name__ == "__main__":
    episodes = [replay.load_episode(p) for p in sorted(Path("episodes").glob("*.json")) if not replay.excluded(p)]
    natural = [ep for ep in episodes if ep.get("flakiness", "natural") == "natural"]
    near = [replay.load_episode(p) for p in sorted(Path("episodes_neareps").glob("*.json"))]
    out = {}
    for flag, name in ((False, "shared"), (True, "slice")):
        rules.SLICE_CAPS = flag
        out[name] = {"natural": summary(natural), "neareps": summary(near), "late": {}}
        for k in (10, 40):
            late = replay.late_streams(episodes, k)
            out[name]["late"][k] = {n: v["correct_accept"] for n, v in summary(late).items()}
        print(name, json.dumps({n: [round(v["blocks_per_loop"]) for v in (out[name]["natural"][n], out[name]["neareps"][n])]
                                + [round(100 * out[name]["late"][40][n], 1)] for n in RULES}))
    rules.SLICE_CAPS = False
    Path("results/slice_caps.json").write_text(json.dumps(out, indent=1))
    print("saved results/slice_caps.json")
