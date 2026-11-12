"""Mini repair agent in RECORD mode: keep proposing fixes and never stop at the first green run.

The stopping rule is what we study, so it is applied later, offline, to the stream saved here.
`llm(messages) -> str` and `block(code) -> BlockResult` are injected (see llm.py, run_block.py).
"""
import hashlib
import re

SYSTEM = (
    "You fix bugs in one Python file. Reply with the complete corrected file in a single ```python block."
)


def extract_code(text):
    m = re.search(r"```(?:python|py)\n(.*?)```", text, re.S)
    return m.group(1) if m else None


def run_stream(llm, block, task, max_attempts=15, on_attempt=None):
    """Returns [{"fix_id", "code", "agent_failed"}, ...] in the order the agent produced them.

    The agent's only feedback is ONE block per attempt (its "CI"); that run is NOT used as evidence later.
    on_attempt(i, entry) is called after attempt i (1-based); entry is None when the reply had no code.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Bug report:\n{task['report']}\n\nFile {task['relpath']}:\n```python\n{task['code']}```"},
    ]
    stream = []
    for i in range(1, max_attempts + 1):
        reply = llm(messages)
        code = extract_code(reply)
        if code is None:
            if on_attempt:
                on_attempt(i, None)
            continue
        res = block(code)
        stream.append({"fix_id": hashlib.sha1(code.encode()).hexdigest()[:12], "code": code, "agent_failed": res.failed})
        if on_attempt:
            on_attempt(i, stream[-1])
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"The tests {'FAILED' if res.failed else 'passed'}. Output:\n{res.output}\n\nTry again."},
        ]
    return stream
