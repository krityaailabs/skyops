"""Lecture 8.8: the retention test, as a function. Run it in CI on every replayed conversation, and in
production on a sample of the requests that actually left (Section 20 turns its output into a metric).

    problems = retention_problems(ctx.sent, ctx.pinned, ctx.window)   # [] = nothing lost, nothing broken
"""
from __future__ import annotations

from ..model.errors import LabNotDone
from ..model.types import ChatRequest
from .budget import Window, account
from .compaction import orphaned_results
from .pinned import PinnedBlock
from .window import pinned_missing


def retention_problems(sent: list[ChatRequest], blocks: list[PinnedBlock], window: Window) -> list[str]:
    """Lecture 8.8: what every request must keep, checked on every request. Returns problem strings.

    For each request, numbered from 1 in the order given, append in this order:
    1. "request {i}: pinned {id} missing" for each id in pinned_missing(request, blocks);
    2. "request {i}: pinned {id} appears {n} times" for each block whose marker() occurs n > 1 times in
       request.system (a duplicated block wastes the budget and hides which copy the model obeys);
    3. "request {i}: orphaned tool result {tool_use_id}" for each id in orphaned_results(request.messages);
    4. "request {i}: {total} tokens over the input limit {limit}" if account(request).total is greater
       than window.input_limit (total and limit as plain integers, no separators).
    An empty list means every request kept its rules, its tool pairs and its budget.
    """
    problems: list[str] = []
    for i, r in enumerate(sent, 1):
        problems += [f"request {i}: pinned {m} missing" for m in pinned_missing(r, blocks)]
        for b in blocks:
            n = (r.system or "").count(b.marker())
            if n > 1:
                problems.append(f"request {i}: pinned {b.id} appears {n} times")
        problems += [f"request {i}: orphaned tool result {t}" for t in orphaned_results(r.messages)]
        total = account(r).total
        if total > window.input_limit:
            problems.append(f"request {i}: {total} tokens over the input limit {window.input_limit}")
    return problems
