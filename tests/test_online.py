"""D20: the online loop asks the agent for a new fix only when SeqAccept rejects the current one."""
import random

import online
from run_block import BlockResult


def fake_llm(codes):
    replies = iter(codes)
    calls = []

    def llm(messages):
        calls.append([m["content"] for m in messages])
        return next(replies)
    return llm, calls


def test_a_rejected_fix_is_fed_back_and_the_next_one_is_accepted():
    llm, calls = fake_llm(["```python\nbroken\n```", "no code here", "```python\ngood\n```"])
    rng = random.Random(0)
    run = lambda code, seed: BlockResult(int(code == "broken\n" or rng.random() < 0.0), "boom" if code == "broken\n" else "")
    task = {"p0": 0.0, "code": "buggy\n", "report": "r", "relpath": "a.py"}
    ep = online.online_episode(task, llm, run, truth_blocks=20, on_event=lambda m: None)
    assert ep["accepted"] == 1 and ep["label"] == "correct" and ep["attempts"] == 3
    assert ep["candidates"][0]["fails"] == ep["candidates"][0]["blocks"] > 0
    assert "FAILED in" in calls[1][-1] and "boom" in calls[1][-1]  # the rejection is the agent's feedback
    assert len(calls[2]) == len(calls[1])  # a reply with no code adds nothing to the history


def test_an_agent_out_of_replies_ends_the_episode_without_an_accept():
    llm, _ = fake_llm(["```python\nbroken\n```"] * 3)
    ep = online.online_episode({"p0": 0.0, "code": "x\n", "report": "r", "relpath": "a.py"}, llm,
                               lambda code, seed: BlockResult(1, "boom"), max_attempts=3, on_event=lambda m: None)
    assert ep["accepted"] is None and ep["label"] is None and ep["attempts"] == 3 and len(ep["candidates"]) == 3
