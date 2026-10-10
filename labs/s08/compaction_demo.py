"""Lecture 8.3: compaction on a real transcript, offline. What it saves, what it keeps, and what breaks.

    python -m labs.s08.compaction_demo

Takes the full transcript of the six-question shift (labs/s08/shift.py), builds the request a context layer
WITHOUT compaction would send for one more question, then compacts it: old turns become a rolling summary,
the last two turns stay verbatim, and the pinned rules are checked on the result. Finally, a naive cut
("keep the last seven messages") shows the bug that cutting by message count causes.
"""
from __future__ import annotations

from skyops.context.budget import account
from skyops.context.compaction import compact, orphaned_results
from skyops.context.manager import ContextManager
from skyops.context.window import pinned_missing
from skyops.model.types import ChatRequest, Message

from . import shift
from .common import SOP

NEXT = "Which of today's flights still need passenger notices?"


def requests():
    s = shift.run()
    box, _ = shift.shift_box()
    transcript = s.transcript + [Message.user(NEXT)]
    plain = ContextManager(window=shift.WINDOW, retrieved=SOP)
    full = ChatRequest(model="claude-haiku-5-5", system=plain.system_prompt(), messages=transcript,
                       tools=box.specs(), max_tokens=512)
    c = compact(transcript, "", keep_turns=2)
    plain.summary = c.summary
    compacted = full.model_copy(update={"system": plain.system_prompt(), "messages": c.kept})
    naive = full.model_copy(update={"messages": transcript[-7:]})
    return s, full, c, compacted, naive


def main() -> None:
    s, full, c, compacted, naive = requests()
    rules = shift.fixed_context().pinned
    b0, b1 = account(full), account(compacted)
    print(f"transcript: {len(full.messages)} messages in {len(s.results) + 1} turns")
    print(f"without compaction: {b0.total:,} tokens ({b0.line()})")
    print(f"compacted:          {b1.total:,} tokens ({b1.line()})")
    print(f"folded {c.dropped_turns} turns ({c.dropped_messages} messages) into {len(c.summary.splitlines())} summary lines; "
          f"kept {len(c.kept)} messages verbatim")
    print(f"pinned rules after compaction: {'intact' if not pinned_missing(compacted, rules) else 'MISSING'} · "
          f"orphaned tool results: {len(orphaned_results(c.kept))}")
    print(f"naive cut (last 7 messages): orphaned tool results: {len(orphaned_results(naive.messages))} "
          f"{orphaned_results(naive.messages)}")


if __name__ == "__main__":
    main()
