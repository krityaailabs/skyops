"""Lecture 8.4: the same task with a free-text scratchpad and with typed working memory, side by side.

    python -m labs.s08.scratchpad            both traces
    python -m labs.s08.scratchpad --compare  one summary line each, and both answers

Free text (write_notes): the notes are rewritten every step, grow, and accept a guess as if it were a fact.
Typed (update_scratchpad): plan, facts and open questions; a fact without a source is rejected and the guess
becomes an open question, which the model then checks.

Both model conversations are scripted replays (fixtures/s08/notes_free.jsonl, notes_typed.jsonl), written to
show the two shapes a scratchpad allows. What's real is what your code accepts, rejects and sends.
"""
from __future__ import annotations

from skyops.agent.loop import Budgets, RunResult, run_agent
from skyops.context.budget import Window
from skyops.context.manager import ContextManager
from skyops.context.scratchpad import FreeTextNotes, WorkingMemory
from skyops.model.client import ModelClient, get_client
from skyops.tools.catalog import toolbox

QUESTION = "Are any of tonight's departures after 21:00 disrupted, and does anything need a passenger notice?"
WINDOW = Window(size_tokens=16_000)
BUDGETS = Budgets(max_steps=12, max_tokens=200_000)


def run(kind: str, client: ModelClient | None = None) -> tuple[RunResult, ContextManager]:
    working = FreeTextNotes() if kind == "free" else WorkingMemory()
    ctx = ContextManager(window=WINDOW, working=working, compaction=False)
    box, _, _ = toolbox(scope=f"notes-{kind}-2026-10-09", dry_run=True)
    box.add(working.tool())
    fixture = f"fixtures/s08/notes_{kind}.jsonl"
    r = run_agent(QUESTION, client or get_client(fixture=fixture, match="strict"), box, budgets=BUDGETS, context=ctx)
    return r, ctx


def main(argv: list[str] | None = None) -> None:
    import sys
    compare = "--compare" in (sys.argv[1:] if argv is None else argv)
    for kind in ("free", "typed"):
        r, ctx = run(kind)
        pad = [e["tokens"]["scratchpad"] for e in ctx.events]
        if compare:
            print(f"{kind:5}: {r.steps} model calls · scratchpad tokens per step {' '.join(str(t) for t in pad)} · "
                  f"{ctx.events[-1]['scratchpad']}")
            print(f"       answer: {r.answer}")
            continue
        print(f"== {kind} ==")
        print(r.trace.render())
        print(f"scratchpad tokens per step: {' '.join(str(t) for t in pad)}")


if __name__ == "__main__":
    main()
