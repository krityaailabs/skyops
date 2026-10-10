"""Lectures 8.3 and 8.8: one long duty-manager shift through the SkyOps context layer.

Six questions in one conversation, on a deliberately small context window, so compaction has to happen in
minutes rather than hours. Every request carries the pinned rules; old turns become a rolling summary; the
trace shows where every token went. Run:  python -m labs.s08.shift

The model's turns are a scripted replay (fixtures/s08/shift.jsonl). Everything else is real: the loop, the
Section 5 tools, compaction, the pinned check and the token accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from skyops.agent.loop import RunResult, run_agent
from skyops.context.budget import Window
from skyops.context.manager import ContextManager
from skyops.model.client import ModelClient, get_client
from skyops.model.types import Message
from skyops.tools.catalog import Gateways

from .common import LONG_RUN, SHIFT, SOP, shift_box

WINDOW = Window(size_tokens=3_400)
FIXTURE = "fixtures/s08/shift.jsonl"


@dataclass
class Shift:
    results: list[RunResult] = field(default_factory=list)
    ctx: ContextManager | None = None
    gw: Gateways | None = None

    @property
    def transcript(self) -> list[Message]:
        return self.results[-1].messages if self.results else []


def converse(questions: list[str], client: ModelClient, ctx: ContextManager, gw: Gateways | None = None) -> Shift:
    """Ask each question in turn, in one conversation: each run continues the previous transcript."""
    box, gw = shift_box(gw)
    out = Shift(ctx=ctx, gw=gw)
    transcript: list[Message] = []
    for q in questions:
        r = run_agent(q, client, box, budgets=LONG_RUN, context=ctx, transcript=transcript)
        out.results.append(r)
        transcript = r.messages
        if r.stop != "answered":
            break
    return out


def fixed_context() -> ContextManager:
    return ContextManager(window=WINDOW, retrieved=SOP)


def run(client: ModelClient | None = None) -> Shift:
    return converse(SHIFT, client or get_client(fixture=FIXTURE, match="strict"), fixed_context())


def report(s: Shift) -> str:
    lines = []
    for i, r in enumerate(s.results, 1):
        ev = [e for e in r.trace.events if e["kind"] == "context"]
        last = ev[-1] if ev else {}
        comp = sum(e["compacted_turns"] for e in ev)
        lines.append(f"turn {i}: {r.steps} model calls · last request {last.get('total', 0):,} tokens · "
                     f"pinned {last.get('pinned', '-')}" + (f" · compacted {comp} turn{'' if comp == 1 else 's'}" if comp else "")
                     + f" · {r.stop}")
    ctx = s.ctx
    ok = sum(1 for e in ctx.events if e["pinned"] == "ok")
    lines.append(f"requests: {len(ctx.events)} · pinned rules intact in {ok} · summary {len(ctx.summary.splitlines())} lines "
                 f"· window input limit {ctx.window.input_limit:,} tokens")
    lines.append(f"answer to turn {len(s.results)}: {s.results[-1].answer}")
    return "\n".join(lines)


def main() -> None:
    s = run()
    print(report(s))


if __name__ == "__main__":
    main()
