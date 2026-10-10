"""Lab 1.2: call a model API, stream the answer, and handle error responses.

Run:  python -m labs.s01.first_call      (replay mode; no key needed)
"""
from __future__ import annotations

import time
from typing import Callable

from skyops.model.client import ModelClient, get_client
from skyops.model.errors import ModelError
from skyops.model.types import ChatRequest, Message, ModelResponse

SYSTEM = "You are a concise assistant for airline operations engineers. Answer in one or two sentences."
QUESTION = "In one sentence, what does an AIDX message carry?"


def ask(client: ModelClient, question: str = QUESTION) -> ModelResponse:
    """Step 1: one request, one response. Return the whole ModelResponse, not just the text:
    the stop reason and the usage are part of the answer."""
    req = ChatRequest(model=client.model, system=SYSTEM, messages=[Message.user(question)], max_tokens=200)
    return client.complete(req)


def stream_answer(client: ModelClient, question: str, on_text: Callable[[str], None]) -> ModelResponse:
    """Step 2: stream. Call on_text for every chunk as it arrives; return the final response."""
    req = ChatRequest(model=client.model, system=SYSTEM, messages=[Message.user(question)], max_tokens=200)
    for chunk in client.stream(req):
        on_text(chunk)
    return client.last_response


def call_with_retry(client: ModelClient, req: ChatRequest, max_attempts: int = 3,
                    sleep: Callable[[float], None] = time.sleep, base_delay_s: float = 1.0) -> ModelResponse:
    """Step 3: handle errors by kind, not by message text.

    - retryable (429, 529, 5xx): wait, then try again; honour retry-after when the provider sends it,
      otherwise back off exponentially;
    - not retryable (400, 401, budget): raise immediately; retrying a bad request only costs time.
    Section 15 adds jitter, retry budgets and circuit breakers.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return client.complete(req)
        except ModelError as e:
            if not e.retryable or attempt == max_attempts:
                raise
            sleep(e.retry_after if e.retry_after is not None else base_delay_s * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


def main() -> None:
    client = get_client(fixture="fixtures/s01/first_call.jsonl", match="sequence")
    r = ask(client)
    print(f"answer: {r.text}\nstop: {r.stop_reason} (provider said {r.raw_stop_reason})\n"
          f"usage: {r.usage.input_tokens} in, {r.usage.output_tokens} out")
    client = get_client(fixture="fixtures/s01/stream.jsonl", match="sequence")
    print("streamed: ", end="")
    final = stream_answer(client, QUESTION, lambda t: print(t, end="", flush=True))
    print(f"\nstream stop: {final.stop_reason}, {final.usage.output_tokens} output tokens")
    client = get_client(fixture="fixtures/s01/errors.jsonl", match="sequence")
    waits: list[float] = []
    req = ChatRequest(model=client.model, system=SYSTEM, messages=[Message.user(QUESTION)], max_tokens=200)
    r = call_with_retry(client, req, sleep=waits.append)
    print(f"after errors: {client.calls} calls, waited {waits} seconds, answer starts: {r.text[:40]!r}")


if __name__ == "__main__":
    main()
