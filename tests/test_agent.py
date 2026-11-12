from agent import extract_code, run_stream
from run_block import BlockResult

TASK = {"report": "off-by-one in add()", "relpath": "pkg/mod.py", "code": "def add(a, b):\n    return a + b + 1\n"}


def reply(code):
    return f"Here is the fix:\n```python\n{code}```\nDone."


def scripted_llm(*codes, seen=None):
    queue = list(codes)

    def llm(messages):
        if seen is not None:
            seen.append(list(messages))  # snapshot: the agent keeps appending to this list
        return queue.pop(0)

    return llm


def test_extract_code_returns_the_first_python_block():
    assert extract_code("text\n```python\nx = 1\n```\nmore\n```python\ny = 2\n```") == "x = 1\n"
    assert extract_code("no code here") is None


def test_recording_does_not_stop_at_the_first_passing_candidate():
    llm = scripted_llm(*[reply(f"v = {i}\n") for i in range(4)])
    stream = run_stream(llm, lambda code: BlockResult(0, "all green"), TASK, max_attempts=4)
    assert [c["agent_failed"] for c in stream] == [0, 0, 0, 0]  # kept going although every run was green


def test_the_agent_sees_the_bug_report_first_and_then_the_previous_test_output():
    seen = []
    llm = scripted_llm(reply("a = 1\n"), reply("a = 2\n"), seen=seen)
    run_stream(llm, lambda code: BlockResult(1, "AssertionError: 3 != 2"), TASK, max_attempts=2)
    assert "off-by-one in add()" in seen[0][-1]["content"]
    assert "AssertionError: 3 != 2" in seen[1][-1]["content"]


def test_identical_code_gets_the_same_fix_id():
    llm = scripted_llm(reply("a = 1\n"), reply("a = 2\n"), reply("a = 1\n"))
    stream = run_stream(llm, lambda code: BlockResult(1, ""), TASK, max_attempts=3)
    ids = [c["fix_id"] for c in stream]
    assert ids[0] == ids[2] != ids[1]


def test_a_reply_without_code_is_skipped_and_not_recorded():
    llm = scripted_llm("Sorry, I cannot do that.", reply("a = 1\n"))
    stream = run_stream(llm, lambda code: BlockResult(1, ""), TASK, max_attempts=2)
    assert [c["code"] for c in stream] == ["a = 1\n"]


def test_progress_is_reported_after_every_attempt_including_skipped_ones():
    seen = []
    llm = scripted_llm("no code", reply("a = 1\n"))
    run_stream(llm, lambda code: BlockResult(0, ""), TASK, max_attempts=2, on_attempt=lambda i, e: seen.append((i, e)))
    assert seen[0] == (1, None) and seen[1][0] == 2 and seen[1][1]["code"] == "a = 1\n"
