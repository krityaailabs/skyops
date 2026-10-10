"""The SkyOps agent loop (Section 4): the quick-win loop rebuilt properly, in plain Python.

    result = run_agent(question, client, ToolBox.flight_tools())

What it adds over the quick win (skyops/agent/quickwin.py, which stays as it was for Sections 0–3):
- every proposed call is validated against the tool's contract before anything runs (lecture 4.3);
- step, token and wall-clock budgets, with a graceful stop that says which budget ran out (4.4);
- tool exceptions and bad arguments come back to the model as structured data, never as a crash (4.5);
- a runaway guard that stops the same call being made over and over (4.6).
Section 8 adds an optional context layer (skyops/context/manager.py): pass context=ContextManager(...) and
every request goes through it (pinned rules, compaction, scratchpad, a token budget per component).
Without it, requests are exactly as in Section 4.

The model proposes; this code decides. Nothing here knows which provider is in use.
One client per run: run_agent attaches its trace to the client, and the spend guard counts per client.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from pydantic import BaseModel, ValidationError

from ..cost import load_price
from ..logs import get_logger, redact
from ..model.client import ModelClient
from ..model.errors import LabNotDone
from ..model.types import ChatRequest, Message, ToolResultBlock, ToolSpec, ToolUseBlock
from ..sim.flights import DATE
from ..trace import Trace

if TYPE_CHECKING:
    from ..context.manager import ContextManager

SYSTEM = (f"You are the SkyOps operations assistant at the HUB airport. Today is {DATE}. "
          "Answer questions about departures using the tools. Use only facts the tools return. "
          "If a tool returns an error, say what you could not find.")

Stop = Literal["answered", "refused", "context_exceeded", "step_budget", "token_budget", "time_budget",
               "repeated_call", "pinned_missing"]


# ---------- tools ----------

@dataclass(frozen=True)
class Tool:
    name: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], dict[str, Any]]
    description: str


@dataclass
class ToolBox:
    """The tools this agent may call. Anything not in the box doesn't exist, whatever the model proposes."""
    tools: dict[str, Tool] = field(default_factory=dict)

    @classmethod
    def flight_tools(cls) -> "ToolBox":
        from ..tools import flights
        return cls({n: Tool(n, m, fn, d) for n, (m, fn, d) in flights.TOOLS.items()})

    def add(self, tool: Tool) -> "ToolBox":
        self.tools[tool.name] = tool
        return self

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name=t.name, description=t.description, input_schema=t.args_model.model_json_schema())
                for t in self.tools.values()]


# ---------- budgets ----------

@dataclass(frozen=True)
class Budgets:
    max_steps: int = 8              # model calls per run
    max_tokens: int = 20_000        # input + output tokens per run, as reported by the provider
    max_seconds: float = 60.0       # wall clock per run, including tool time
    max_repeats: int = 3            # the same call (name and arguments) this many times = runaway


# ---------- results ----------

@dataclass
class RunResult:
    answer: str
    stop: Stop
    trace: Trace
    steps: int
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)      # the full transcript (Section 8)

    @property
    def tools_used(self) -> list[str]:
        return [n for n, _ in self.tool_calls]


log = get_logger("skyops.agent")

STOP_MESSAGES = {
    "refused": "The model declined to answer this request.",
    "context_exceeded": "I stopped: the conversation no longer fits in the model's context window.",
    "step_budget": "I stopped before finishing: the step budget ran out.",
    "token_budget": "I stopped before finishing: the token budget ran out.",
    "time_budget": "I stopped before finishing: the time budget ran out.",
    "repeated_call": "I stopped: I was repeating the same lookup without making progress.",
    "pinned_missing": "I stopped: the safety rules were missing from the request, so it was not sent.",
}


def error_result(call_id: str, kind: str, message: str, retryable: bool) -> ToolResultBlock:
    """An error the model can read and act on, in one stable JSON shape (lecture 4.5)."""
    return ToolResultBlock(tool_use_id=call_id, is_error=True,
                           content=json.dumps({"error": kind, "message": message, "retryable": retryable}))


def validate_call(box: ToolBox, call: ToolUseBlock) -> tuple[BaseModel | None, ToolResultBlock | None]:
    """Lecture 4.3: check a proposed call against the tool's contract before anything runs.

    Returns (validated arguments, None) or (None, an error result for the model).
    Unknown tool -> error "unknown_tool". Arguments that fail the Pydantic model -> "invalid_arguments",
    with the field name and the validator's message. Neither is retryable as-is: the model must change the call.
    """
    tool = box.tools.get(call.name)
    if tool is None:
        return None, error_result(call.id, "unknown_tool", f"No tool named {call.name}. "
                                  f"Available: {', '.join(box.tools)}.", retryable=False)
    try:
        return tool.args_model.model_validate(call.input), None
    except ValidationError as e:
        err = e.errors(include_url=False)[0]
        where = ".".join(str(p) for p in err["loc"]) or "arguments"
        return None, error_result(call.id, "invalid_arguments", f"{where}: {err['msg']}", retryable=False)


def execute(box: ToolBox, call: ToolUseBlock) -> ToolResultBlock:
    """Lecture 4.5: validate, run, and turn every failure into data. Never raises to the loop.

    - validation failures come straight from validate_call;
    - a result dict with an "error" key is a business answer (e.g. not_found): is_error=True, not retryable;
    - a TimeoutError or ConnectionError, or any exception with a true `retryable` attribute, is retryable
      ("tool_unavailable");
    - any other exception is "tool_failed", not retryable;
    - the model sees the exception's type name only; the detail goes to the server log, redacted;
    - a tool that returns something that isn't a JSON-serialisable dict is "tool_failed" too.
    """
    args, err = validate_call(box, call)
    if err is not None:
        return err
    try:
        out = box.tools[call.name].fn(args)
        if not isinstance(out, dict):
            raise TypeError(f"tool returned {type(out).__name__}, not dict")
        out = dict(out)                                  # never modify the tool's own object
        if "error" in out:
            out.setdefault("retryable", False)
            return ToolResultBlock(tool_use_id=call.id, content=json.dumps(out), is_error=True)
        return ToolResultBlock(tool_use_id=call.id, content=json.dumps(out))
    except Exception as e:  # noqa: BLE001 - the loop must survive any tool bug
        if isinstance(e, LabNotDone):       # an unfinished lab must fail loudly, not look like a tool error
            raise
        log.error("tool failed", extra={"tool": call.name, "error_type": type(e).__name__, "detail": redact(str(e))})
        if isinstance(e, (TimeoutError, ConnectionError)) or getattr(e, "retryable", False):
            return error_result(call.id, "tool_unavailable", f"{call.name} did not respond: {type(e).__name__}",
                                retryable=True)
        return error_result(call.id, "tool_failed", f"{call.name} failed: {type(e).__name__}", retryable=False)


def check_budgets(b: Budgets, steps: int, tokens: int, elapsed_s: float) -> Stop | None:
    """Lecture 4.4: which budget, if any, is used up. Checked before every model call.

    Order matters when several are exhausted at once: time, then tokens, then steps, so the
    reported reason is the most expensive one.
    """
    if elapsed_s >= b.max_seconds:
        return "time_budget"
    if tokens >= b.max_tokens:
        return "token_budget"
    if steps >= b.max_steps:
        return "step_budget"
    return None


def detect_runaway(history: list[tuple[str, dict[str, Any]]], max_repeats: int) -> bool:
    """Lecture 4.6: True when any (tool, arguments) pair has been proposed max_repeats times or more.
    Arguments compare as canonical JSON, so key order doesn't hide a repeat; repeats needn't be consecutive."""
    seen: dict[str, int] = {}
    for name, args in history:
        k = json.dumps([name, args], sort_keys=True)
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= max_repeats:
            return True
    return False


def run_agent(question: str, client: ModelClient, box: ToolBox, budgets: Budgets = Budgets(),
              clock: Callable[[], float] = time.monotonic, system: str = SYSTEM,
              max_tokens_per_call: int = 512, context: "ContextManager | None" = None,
              transcript: list[Message] | None = None) -> RunResult:
    """Lecture 4.2: the loop. Ask the model; if it proposes tools, validate and run them in our code,
    append the results, and ask again; stop on an answer or when a budget or the runaway guard says so.

    Section 8: `transcript` continues an earlier conversation (pass the previous result's messages), and
    `context` builds every request through the context layer; a request that can't fit stops the run with
    "context_exceeded", and one without its pinned rules is never sent ("pinned_missing")."""
    trace = Trace(question=question, price=load_price())
    client.trace = trace
    messages = list(transcript or []) + [Message.user(question)]
    history: list[tuple[str, dict[str, Any]]] = []
    start, steps = clock(), 0
    while True:
        stop = check_budgets(budgets, steps, trace.usage.total, clock() - start)
        if stop is None and detect_runaway(history, budgets.max_repeats):
            stop = "repeated_call"
        req = ChatRequest(model=client.model, system=system, messages=messages, tools=box.specs(),
                          max_tokens=max_tokens_per_call)
        if stop is None and context is not None:
            from ..context.window import ContextOverflow, PinnedRulesMissing
            try:
                req = context.prepare(req, trace)
            except ContextOverflow:
                stop = "context_exceeded"
            except PinnedRulesMissing:
                stop = "pinned_missing"
        if stop is not None:
            trace.answer(f"[{stop}] {STOP_MESSAGES[stop]}")
            return RunResult(STOP_MESSAGES[stop], stop, trace, steps, history, messages)
        resp = client.complete(req)
        steps += 1
        messages.append(resp.as_message())
        if resp.stop_reason in ("refusal", "context_exceeded"):
            stop = "refused" if resp.stop_reason == "refusal" else "context_exceeded"
            trace.answer(f"[{stop}] {STOP_MESSAGES[stop]}")
            return RunResult(STOP_MESSAGES[stop], stop, trace, steps, history, messages)
        if resp.stop_reason != "tool_use":         # end_turn, max_tokens (short answers only), other
            trace.answer(resp.text)
            return RunResult(resp.text, "answered", trace, steps, history, messages)
        results = []
        for call in resp.tool_calls:            # proposals; our code decides what runs
            history.append((call.name, call.input))
            res = execute(box, call)
            trace.tool_call(call.name, call.input, not res.is_error,
                            res.content[:80] + ("…" if len(res.content) > 80 else ""))
            if context is not None:
                context.observe(call, res)
            results.append(res)
        messages.append(Message(role="user", content=results))
