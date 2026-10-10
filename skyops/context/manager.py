"""Lectures 8.3 and 8.8: the SkyOps context layer, in the Section 4 loop.

    ctx = ContextManager(window=Window(8_000), retrieved=SOP, working=WorkingMemory())
    result = run_agent(question, client, box.add(ctx.working.tool()), context=ctx)

Before every model call the loop hands its full transcript to ctx.prepare(), which builds the request the
model actually sees:

    system  = pinned rules (verbatim, hashed) + persona + <retrieved>/<memory> + <summary> + <scratchpad>
    messages = the turns not yet folded into the summary

It compacts when the request passes a share of the window, fits what's left (history only), checks the
pinned rules on the final request, and records where every token went in the trace. The loop's own
transcript is never changed: it is the record (and, in Section 15, the checkpoint). The prompt is a view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..model.types import ChatRequest, ToolResultBlock, ToolUseBlock
from .budget import Window, account
from .compaction import compact
from .pinned import SAFETY_RULES, PinnedBlock
from .window import PinnedRulesMissing, fit_to_window, pinned_missing

PERSONA = ("You are the SkyOps operations assistant at the HUB airport, working with the duty manager. Today is "
           "2026-10-09. Answer from tool results only, and say what you could not find.")

FitFn = Callable[[ChatRequest, Window, list[PinnedBlock]], tuple[ChatRequest, int]]


@dataclass
class ContextManager:
    window: Window
    persona: str = PERSONA
    pinned: list[PinnedBlock] = field(default_factory=lambda: [SAFETY_RULES])
    retrieved: str = ""                  # documents chosen for this task (Section 9 does this properly)
    memory: str = ""                     # a rendered <memory> block (skyops.context.memory.render_memories)
    working: Any = None                  # WorkingMemory, FreeTextNotes, or None
    compaction: bool = True
    compact_at_share: float = 0.7        # compact when a request passes this share of window.input_limit
    keep_turns: int = 2
    enforce_pinned: bool = True          # False ONLY to reproduce Incident 08: detect, but still send
    rules_first: bool = True
    fit: FitFn = fit_to_window
    summary: str = ""
    cut: int = 0                         # how many transcript messages are folded into the summary
    events: list[dict[str, Any]] = field(default_factory=list)
    sent: list[ChatRequest] = field(default_factory=list)

    def system_prompt(self) -> str:
        pinned = [b.render() for b in self.pinned]
        body = [self.persona]
        if self.retrieved:
            body.append(f"<retrieved>\n{self.retrieved}\n</retrieved>")
        if self.memory:
            body.append(self.memory)
        if self.summary:
            body.append(f"<summary>\nEarlier in this conversation, compacted:\n{self.summary}\n</summary>")
        pad = self.working.render() if self.working is not None else ""
        if pad:
            body.append(pad)
        return "\n\n".join(pinned + body if self.rules_first else body + pinned)

    def prepare(self, req: ChatRequest, trace=None) -> ChatRequest:
        """Build the request the model will see, from the loop's request (full transcript).

        1. candidate = req.model_copy(update={"system": self.system_prompt(), "messages": req.messages[self.cut:]}).
        2. If self.compaction and account(candidate).total > self.compact_at_share * self.window.input_limit:
           c = compact(candidate.messages, self.summary, self.keep_turns). If c.dropped_messages > 0, set
           self.summary = c.summary, add c.dropped_messages to self.cut, and rebuild the candidate as in
           step 1 (the system prompt changed: it now carries the summary).
        3. sent, dropped = self.fit(candidate, self.window, self.pinned). Let ContextOverflow and
           PinnedRulesMissing propagate: the loop turns them into a stop.
        4. missing = pinned_missing(sent, self.pinned). If missing and self.enforce_pinned, raise
           PinnedRulesMissing(missing).
        5. Record an event dict with exactly these keys: "kind": "context", "step": len(self.events) + 1,
           "tokens": account(sent).tokens, "total": account(sent).total, "compacted_turns": the number of
           turns compacted in step 2 (0 if none), "fit_dropped": dropped, "pinned": "ok" if nothing is
           missing, else "MISSING " + ", ".join(missing), and "scratchpad": self.working.status() if
           self.working is not None, else "". Append it to self.events and, if trace is not None, call
           trace.context(event).
        6. Append sent to self.sent and return it.
        """
        def build() -> ChatRequest:
            return req.model_copy(update={"system": self.system_prompt(), "messages": req.messages[self.cut:]})

        candidate, compacted = build(), 0
        if self.compaction and account(candidate).total > self.compact_at_share * self.window.input_limit:
            c = compact(candidate.messages, self.summary, self.keep_turns)
            if c.dropped_messages > 0:
                self.summary, self.cut, compacted = c.summary, self.cut + c.dropped_messages, c.dropped_turns
                candidate = build()
        sent, dropped = self.fit(candidate, self.window, self.pinned)
        missing = pinned_missing(sent, self.pinned)
        if missing and self.enforce_pinned:
            raise PinnedRulesMissing(missing)
        b = account(sent)
        event = {"kind": "context", "step": len(self.events) + 1, "tokens": b.tokens, "total": b.total,
                 "compacted_turns": compacted, "fit_dropped": dropped,
                 "pinned": "ok" if not missing else "MISSING " + ", ".join(missing),
                 "scratchpad": self.working.status() if self.working is not None else ""}
        self.events.append(event)
        if trace is not None:
            trace.context(event)
        self.sent.append(sent)
        return sent

    def observe(self, call: ToolUseBlock, result: ToolResultBlock) -> None:
        """After each tool call: let working memory see successful results, so facts can cite them."""
        if self.working is not None:
            self.working.observe(call.id, call.name, result.content, not result.is_error)
