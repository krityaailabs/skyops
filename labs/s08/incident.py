"""Lecture 8.7, Incident 08: a context overflow silently dropped the safety instructions.

    python -m labs.s08.incident            the shift on the OLD request builder (v1), with the detection
                                           signal running in shadow mode, then the same shift on the fix
    python -m labs.s08.incident --v1       the v1 builder only
    python -m labs.s08.incident --fixed    the fixed builder only
    python -m labs.s08.incident --guard    v1 again, with the detection signal as a guard: the request
                                           without its rules is never sent (the mitigation, not the fix)

What v1 did (the code is below, kept for this lab): rules appended at the END of the system prompt, and,
when a request was over budget, whole lines cut from the end of the system prompt, "because the conversation
is what the duty manager sees". No error, no log line. Late in a long shift the cut reached the rules.

Both model conversations are scripted replays (fixtures/s08/incident_v1.jsonl and shift.jsonl). The canary
is benign (CANARY-7F3): success for an attacker is the canary reaching the outbox, never real data leaving.
Everything else is real: the builder, the cut, the check, the tools and the outbox.
"""
from __future__ import annotations

import argparse

from labs.s03.detectors import detect_canary
from skyops.context.budget import Window, account
from skyops.context.manager import ContextManager
from skyops.context.pinned import PinnedBlock
from skyops.model.client import ModelClient, get_client
from skyops.model.types import ChatRequest
from skyops.sim.faults import CANARY

from . import shift
from .common import SHIFT, SOP

FIXTURE_V1 = "fixtures/s08/incident_v1.jsonl"


def fit_v1(req: ChatRequest, window: Window, blocks: list[PinnedBlock]) -> tuple[ChatRequest, int]:
    """SkyOps request builder v1, as it was before Incident 08. DO NOT USE: kept to reproduce the incident.
    Over budget? Cut lines from the end of the system prompt until the request fits. Silent."""
    if account(req).total <= window.input_limit:
        return req, 0
    lines = (req.system or "").splitlines()
    while lines and account(req.model_copy(update={"system": "\n".join(lines)})).total > window.input_limit:
        lines.pop()
    return req.model_copy(update={"system": "\n".join(lines)}), 0


def v1_context(guard: bool = False) -> ContextManager:
    """v1's settings: rules last, no compaction, the silent cut, and (in this lab only) the detection signal
    in shadow mode, so we can see when it WOULD have fired without changing what v1 sent."""
    return ContextManager(window=shift.WINDOW, retrieved=SOP, rules_first=False, compaction=False,
                          fit=fit_v1, enforce_pinned=guard)


def run(fixed: bool, client: ModelClient | None = None, guard: bool = False) -> shift.Shift:
    if fixed:
        return shift.run(client)
    return shift.converse(SHIFT, client or get_client(fixture=FIXTURE_V1, match="strict"), v1_context(guard))


def rules_lines_kept(req: ChatRequest, block: PinnedBlock) -> int:
    """How many lines of the rules text are still in the request (for the timeline)."""
    return sum(1 for line in block.text.splitlines() if line in (req.system or ""))


def timeline(s: shift.Shift) -> list[str]:
    ctx, block = s.ctx, s.ctx.pinned[0]
    turn_of, n = [], 0
    for t, r in enumerate(s.results, 1):
        k = sum(1 for e in r.trace.events if e["kind"] == "context")
        turn_of += [t] * k
    out = []
    for i, (e, req) in enumerate(zip(ctx.events, ctx.sent), 1):
        out.append(f"request {i:>2} (turn {turn_of[i - 1]}): {e['total']:>5,} tokens · rules lines "
                   f"{rules_lines_kept(req, block)}/{len(block.text.splitlines())} · pinned {e['pinned']}")
    if s.results and s.results[-1].stop == "pinned_missing":
        out.append(f"request {len(ctx.events) + 1:>2} (turn {len(s.results)}): NOT SENT · the guard found the "
                   f"pinned rules missing and stopped the run")
    return out


def summary(s: shift.Shift) -> dict:
    ctx = s.ctx
    first = next((i for i, e in enumerate(ctx.events, 1) if e["pinned"] != "ok"), None)
    if first is None and s.results and s.results[-1].stop == "pinned_missing":
        first = len(ctx.events) + 1                   # the guard refused it, so it was never sent
    return {"requests": len(ctx.events), "first_missing": first,
            "missing": sum(1 for e in ctx.events if e["pinned"] != "ok"),
            "canary_sent": detect_canary(s.gw.sent, [CANARY]), "messages_sent": len(s.gw.sent)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", action="store_true")
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--v1", action="store_true")
    a = ap.parse_args(argv)
    for fixed in ([True] if a.fixed else [False] if (a.guard or a.v1) else [False, True]):
        s = run(fixed, guard=a.guard)
        sm = summary(s)
        title = "fixed builder" if fixed else "v1 builder, detection signal as a guard" if a.guard else \
            "v1 builder (detection signal in shadow mode)"
        print(f"== {title} ==")
        print("\n".join(timeline(s)))
        print(f"first request without intact rules: {sm['first_missing'] or 'none'} · outbound messages "
              f"{sm['messages_sent']} · canary in outbox: {sm['canary_sent']}")
        print(f"last answer: {s.results[-1].answer}")


if __name__ == "__main__":
    main()
