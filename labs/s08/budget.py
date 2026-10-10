"""Lecture 8.1: measure the token budget of a ten-step run by component, and decide what to cut first.

One question, ten model calls. Every call re-sends everything before it, so the request grows step by step:
the rules and tool schemas are a fixed cost per call, history and tool results grow, and the run's total
input is the sum of all ten requests. Run:  python -m labs.s08.budget

The model's turns are a scripted replay (fixtures/s08/budget_run.jsonl). The accounting is real, and an
estimate (skyops/context/tokens.py): no provider tokenizer runs offline.
"""
from __future__ import annotations

from skyops.agent.loop import Budgets, RunResult, run_agent
from skyops.context.budget import COMPONENTS, Window, account
from skyops.context.compaction import clear_old_results
from skyops.context.manager import ContextManager
from skyops.context.scratchpad import WorkingMemory
from skyops.model.client import ModelClient, get_client
from skyops.tools.catalog import toolbox

from .common import SOP

QUESTION = ("Before the evening bank: check every departure from 16:00 onwards and tell me which ones need a "
            "passenger notice under DP-4. Don't send anything.")
WINDOW = Window(size_tokens=16_000)
FIXTURE = "fixtures/s08/budget_run.jsonl"
BUDGETS = Budgets(max_steps=12, max_tokens=200_000)


def run(client: ModelClient | None = None) -> tuple[RunResult, ContextManager]:
    ctx = ContextManager(window=WINDOW, retrieved=SOP, working=WorkingMemory(), compaction=False)
    box, _, _ = toolbox(scope="review-2026-10-09-evening", dry_run=True)
    box.add(ctx.working.tool())
    r = run_agent(QUESTION, client or get_client(fixture=FIXTURE, match="strict"), box, budgets=BUDGETS, context=ctx)
    return r, ctx


def cut_options(ctx: ContextManager) -> list[tuple[str, int]]:
    """What each cut would save on the LAST request, measured, in the order you should consider them."""
    last = ctx.sent[-1]
    base = account(last).total
    cleared = last.model_copy(update={"messages": clear_old_results(last.messages, keep_last=2)})
    no_sop = last.model_copy(update={"system": last.system.replace(f"<retrieved>\n{SOP}\n</retrieved>", "")})
    return [("clear tool results older than the last two (facts are in the scratchpad)", base - account(cleared).total),
            ("drop the retrieved procedure (only if the task no longer needs it)", base - account(no_sop).total),
            ("compact old turns (nothing to compact: the whole run is one turn)", 0),
            ("cut the pinned rules", 0)]


def main() -> None:
    r, ctx = run()
    head = f"{'step':>4} " + " ".join(f"{c[:8]:>9}" for c in COMPONENTS) + f" {'total':>7}"
    print(head)
    for e in ctx.events:
        print(f"{e['step']:>4} " + " ".join(f"{e['tokens'][c]:>9,}" for c in COMPONENTS) + f" {e['total']:>7,}")
    cumulative = sum(e["total"] for e in ctx.events)
    last = ctx.events[-1]
    print(f"run: {r.steps} model calls · input re-sent across the run {cumulative:,} tokens (estimate) · "
          f"last request {last['total'] / WINDOW.input_limit:.0%} of the input limit ({WINDOW.input_limit:,})")
    print("cut first (saving on the last request):")
    for what, saved in cut_options(ctx):
        print(f"  {saved:>6,}  {what}" if what != "cut the pinned rules" else f"  {'never':>6}  {what}")


if __name__ == "__main__":
    main()
