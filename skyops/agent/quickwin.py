"""The quick-win agent (lecture 0.3). Deliberately small, and deliberately incomplete.

It loops: the model proposes tool calls, this code validates and runs them, the results go back,
until the model answers or the step budget runs out. It has a step budget and a spend cap.
It does NOT yet have: idempotency, timeouts, retries, approval, injection defences or evals.
Lecture 3.3 breaks it in five ways; the rest of the course fixes each one properly.

Run:  make quickwin            (replay mode, no API key, no cost)
"""
from __future__ import annotations

import sys

import dataclasses

from ..cost import load_price

from ..model.client import ModelClient, get_client
from ..model.errors import ModelError
from ..model.types import ChatRequest, Message, ToolResultBlock
from ..settings import ROOT, Settings
from ..sim.flights import DATE
from ..tools import flights
from ..trace import Trace

SYSTEM = (f"You are the SkyOps operations assistant at the HUB airport. Today is {DATE}. "
          "Answer questions about departures using the tools. Use only facts the tools return. "
          "If a tool returns an error, say what you could not find.")
QUESTION = "Is the evening flight to Singapore on time, and which gate is it leaving from?"
MAX_STEPS = 6


def run(question: str, client: ModelClient, max_steps: int = MAX_STEPS,
        tool_runner=None, tool_specs=None, system: str = SYSTEM) -> Trace:
    """tool_runner and tool_specs default to the read-only flight tools; Section 2 swaps in a
    recording tool runner, and Section 3 adds a write tool to break the agent on purpose."""
    tool_runner = tool_runner or flights.run_tool
    tool_specs = tool_specs if tool_specs is not None else flights.specs()
    trace = Trace(question=question, price=load_price())
    client.trace = trace
    messages = [Message.user(question)]
    for _ in range(max_steps):
        resp = client.complete(ChatRequest(model=client.model, system=system, messages=messages,
                                           tools=tool_specs, max_tokens=512))
        messages.append(resp.as_message())
        if resp.stop_reason != "tool_use":
            trace.answer(resp.text)
            return trace
        results = []
        for call in resp.tool_calls:            # the model proposed these; our code decides
            ok, output = tool_runner(call.name, call.input)
            trace.tool_call(call.name, call.input, ok, output[:80] + ("…" if len(output) > 80 else ""))
            results.append(ToolResultBlock(tool_use_id=call.id, content=output, is_error=not ok))
        messages.append(Message(role="user", content=results))
    trace.answer(f"[stopped: step budget of {max_steps} used up]")
    return trace


def main() -> int:
    s = Settings.from_env()
    if s.fixture is None:
        s = dataclasses.replace(s, fixture=ROOT / "fixtures/s00/quickwin.jsonl")
    try:
        trace = run(QUESTION, get_client(s))
    except ModelError as e:
        print(f"stopped: {e}")
        return 1
    print(trace.render())
    print(f"trace saved to {trace.save(ROOT / 'runs').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
