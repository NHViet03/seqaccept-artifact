import pytest

from llm import _wait_out_rate_limit


class RateLimitError(Exception):  # same class name as openai's, which is how the helper recognises it
    pass


def flaky(fails, exc=RateLimitError):
    calls = []

    def call():
        calls.append(1)
        if len(calls) <= fails:
            raise exc("429")
        return "reply"

    return call, calls


def test_a_rate_limited_call_waits_and_then_succeeds():
    call, calls = flaky(2)
    waits = []
    assert _wait_out_rate_limit(call, wait=65, sleep=waits.append, log=lambda m: None) == "reply"
    assert waits == [65, 65] and len(calls) == 3


def test_it_gives_up_after_the_last_try_and_other_errors_are_not_retried():
    call, calls = flaky(99)
    with pytest.raises(RateLimitError):
        _wait_out_rate_limit(call, tries=3, sleep=lambda s: None, log=lambda m: None)
    assert len(calls) == 3
    call, calls = flaky(1, exc=ValueError)
    with pytest.raises(ValueError):
        _wait_out_rate_limit(call, sleep=lambda s: None, log=lambda m: None)
    assert len(calls) == 1


def test_ollama_sends_its_context_size_and_fails_loudly_when_the_history_fills_it(monkeypatch):
    import io, json, llm

    sent = []

    def fake(prompt_tokens):
        def urlopen(req, timeout):
            sent.append(json.loads(req.data))
            return io.BytesIO(json.dumps({"message": {"content": "ok"}, "prompt_eval_count": prompt_tokens}).encode())
        return urlopen

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake(1000))
    assert llm.ollama("m", num_ctx=4096)([{"role": "user", "content": "x"}]) == "ok"
    assert sent[0]["options"]["num_ctx"] == 4096
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake(4000))
    with pytest.raises(RuntimeError, match="context window full"):
        llm.ollama("m", num_ctx=4096)([{"role": "user", "content": "x"}])
