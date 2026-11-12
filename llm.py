"""LLM clients for agent.run_stream: `llm(messages) -> str`. Not unit-tested (network); smoke test:
    uv run --env-file .env python llm.py azure gpt-4.1-mini     |     uv run python llm.py ollama qwen2.5-coder:32b
Azure needs LLM_ENDPOINT and LLM_API_KEY (optional LLM_API_VERSION) from .env; `model` is your DEPLOYMENT name.
"""
import json
import os
import sys
import time
import urllib.request


def _wait_out_rate_limit(call, tries=30, wait=65, sleep=time.sleep, log=print):
    """Call `call()`; on a 429 that the SDK's own retries did not clear, wait out the tokens-per-minute window and try again.
    The agent resends its whole history, so late attempts on a large file exceed the deployment's per-minute token limit."""
    for attempt in range(tries):
        try:
            return call()
        except Exception as e:
            if type(e).__name__ != "RateLimitError" or attempt == tries - 1:
                raise
            log(f"  rate limited (429), waiting {wait} s ({attempt + 1}/{tries})")
            sleep(wait)


def azure(deployment, temperature=0.8):
    from openai import AzureOpenAI  # pip extra: uv sync --extra llm

    client = AzureOpenAI(
        azure_endpoint=os.environ["LLM_ENDPOINT"],
        api_key=os.environ["LLM_API_KEY"],
        api_version=os.environ.get("LLM_API_VERSION", "2024-10-21"),
        max_retries=8,  # the SDK retries rate limits, timeouts and 5xx with backoff, so a long record is not lost
        timeout=1800,  # one grok-4.7 reply on an 800-line file took ~17 min
    )
    def llm(messages):
        r = _wait_out_rate_limit(lambda: client.chat.completions.create(
            model=deployment, messages=messages, temperature=temperature))
        return (r.choices[0].message.content or "") if r.choices else ""  # empty reply (e.g. filtered): the agent skips it like any reply without code

    return llm


NUM_CTX = 98304  # Ollama defaults to 4096 and silently drops the oldest turns; 96k tokens of KV cache is ~12 GB for an 8B model (fits every task but freezegun)


def ollama(model, temperature=0.8, host="http://localhost:11434", num_ctx=NUM_CTX):
    def llm(messages):
        body = json.dumps({"model": model, "messages": messages, "stream": False, "keep_alive": "24h",
                           "options": {"temperature": temperature, "num_ctx": num_ctx}})
        req = urllib.request.Request(f"{host}/api/chat", body.encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=1800))  # a local 8B reply that rewrites a 800-line file takes minutes
        if d.get("prompt_eval_count", 0) >= num_ctx - 256:  # the history no longer fits: Ollama would drop the bug report, so fail loudly
            raise RuntimeError(f"context window full ({d['prompt_eval_count']} of {num_ctx} tokens)")
        return d["message"]["content"]

    return llm


if __name__ == "__main__":
    kind, model = sys.argv[1], sys.argv[2]
    ask = {"azure": azure, "ollama": ollama}[kind](model)
    print(ask([{"role": "user", "content": "Reply with exactly: ok"}]))
