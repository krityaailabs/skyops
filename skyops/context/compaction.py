"""Lectures 8.3 and 8.8: compaction. Replace old turns with a rolling summary; never touch pinned text.

    c = compact(messages, previous_summary, keep_turns=2)
    c.summary   -> previous summary + one line per dropped event, rendered later as <summary>…</summary>
    c.kept      -> the last two turns, verbatim (the same Message objects)

Compaction only ever sees the history. The pinned rules live in the system prompt, which compaction has
no access to, so it can't summarise them away; the retention test proves that on every request.

The summariser here is extractive and deterministic: it keeps who asked what, which tools ran with which
arguments, the key fields of each result (values cut to 60 characters), and the answers. It never copies a
free-text field (remarks, messages) or a result that isn't a JSON object: those are replaced by a note of
their size. And because the summary lands in the system prompt, every line built from a tool call, a result
or an answer is screened with the memory screen's instruction pattern, and withheld if it matches. In live
mode you could ask a model to write the summary instead, but test it the same way: what must survive is
decided by you, not by the summariser.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..model.errors import LabNotDone
from ..model.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from .memory import INSTRUCTION
from .window import turn_starts

KEY_FIELDS = ("flight_number", "status", "estimated_departure", "gate", "delay_reason", "recipients",
              "replayed", "voucher_id", "error", "total", "next_cursor")   # no free-text fields
MAX_VALUE = 60
MAX_SUMMARY_LINES = 40


@dataclass
class Compacted:
    summary: str
    kept: list[Message] = field(default_factory=list)
    dropped_turns: int = 0
    dropped_messages: int = 0


def _short(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _result_line(content: str) -> str:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return f"(non-JSON result, {len(content)} chars, omitted)"
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        rows = [" ".join(_short(str(r.get(k)), MAX_VALUE) for k in ("flight_number", "scheduled_departure", "status")
                         if r.get(k)) for r in data["items"] if isinstance(r, dict)]
        return "; ".join(rows)
    if isinstance(data, dict):
        return ", ".join(f"{k}={_short(str(data[k]), MAX_VALUE)}" for k in KEY_FIELDS if data.get(k) not in (None, ""))
    return f"(JSON {type(data).__name__} result, {len(content)} chars, omitted)"


def _screened(line: str) -> str:
    """Withhold a line that reads like an instruction: the summary sits in the system prompt."""
    if INSTRUCTION.search(line):
        return line[:line.index(":") + 1] + " (withheld: instruction-like text)" if ":" in line else \
            "- (withheld: instruction-like text)"
    return line


def summarise(messages: list[Message]) -> list[str]:
    """One line per event, in order: requests, tool calls, results (key fields only) and answers.
    Every line except the duty manager's own request is screened for instruction-like text."""
    lines: list[str] = []
    for m in messages:
        for b in m.content:
            if isinstance(b, TextBlock) and m.role == "user":
                lines.append(f"- Duty manager asked: {_short(b.text, 140)}")
            elif isinstance(b, TextBlock):
                lines.append(_screened(f"- Assistant said: {_short(b.text, 160)}"))
            elif isinstance(b, ToolUseBlock):
                args = ", ".join(f"{k}={_short(str(v), MAX_VALUE)}" for k, v in b.input.items() if k != "message")
                lines.append(_screened(f"- Called {b.name}({args})"))
            elif isinstance(b, ToolResultBlock):
                lines.append(_screened(f"  {'error' if b.is_error else 'result'}: {_result_line(b.content)}"))
    return lines


def orphaned_results(messages: list[Message]) -> list[str]:
    """Tool results whose tool call isn't in the message just before them. Providers reject these, so any
    cut that produces one is a bug (lecture 8.3's "what breaks")."""
    out = []
    for i, m in enumerate(messages):
        for b in m.content:
            if isinstance(b, ToolResultBlock):
                prev = messages[i - 1] if i > 0 else None
                ids = {x.id for x in prev.content if isinstance(x, ToolUseBlock)} if prev and prev.role == "assistant" else set()
                if b.tool_use_id not in ids:
                    out.append(b.tool_use_id)
    return out


CLEARED = '{"cleared": "older result removed to save context; call the tool again if needed"}'


def clear_old_results(messages: list[Message], keep_last: int = 2) -> list[Message]:
    """Lecture 8.1: the cheapest cut inside ONE long turn. Replace the content of every tool result except
    the last `keep_last` with a short stub. The tool call and its result block both stay, so the pairing
    the provider requires is intact; only the bulky content goes. Returns new messages; the input is unchanged."""
    positions = [(i, j) for i, m in enumerate(messages) for j, b in enumerate(m.content) if isinstance(b, ToolResultBlock)]
    clear = set(positions[:max(0, len(positions) - keep_last)])
    out = []
    for i, m in enumerate(messages):
        if any((i, j) in clear for j in range(len(m.content))):
            blocks = [b.model_copy(update={"content": CLEARED}) if (i, j) in clear else b for j, b in enumerate(m.content)]
            out.append(m.model_copy(update={"content": blocks}))
        else:
            out.append(m)
    return out


def compact(messages: list[Message], previous_summary: str, keep_turns: int = 2,
            max_lines: int = MAX_SUMMARY_LINES) -> Compacted:
    """Lecture 8.8: fold every turn except the last `keep_turns` into a rolling summary.

    1. starts = turn_starts(messages). If len(starts) <= keep_turns, nothing is old enough: return
       Compacted(previous_summary, list(messages), 0, 0).
    2. cut = starts[-keep_turns]. Everything before the cut is dropped; messages[cut:] are kept verbatim.
       Cutting only at turn starts means a tool call and its result are never separated.
    3. The new summary is ROLLING: the lines of previous_summary (if any), then summarise(dropped).
       Never rebuild it from scratch: the dropped messages are gone after this call.
    4. Keep at most max_lines lines: drop the OLDEST lines first and put one line at the top that reads
       exactly "- (N earlier lines omitted)" with N the total omitted so far, including any number an
       earlier omission line already carried (that line itself is not counted). The omission line counts
       toward max_lines: with it, the summary is still at most max_lines lines in total.
    5. Return Compacted("\\n".join(lines), messages[cut:], dropped_turns=number of starts before the cut,
       dropped_messages=cut).
    Compaction never sees the system prompt, so pinned rules can't be summarised away.
    """
    starts = turn_starts(messages)
    if len(starts) <= keep_turns:
        return Compacted(previous_summary, list(messages), 0, 0)
    cut = starts[-keep_turns]
    lines = [l for l in previous_summary.splitlines() if l.strip()]
    omitted = 0
    if lines and lines[0].startswith("- (") and lines[0].endswith(" earlier lines omitted)"):
        omitted = int(lines[0][3:].split()[0])
        lines = lines[1:]
    lines += summarise(messages[:cut])
    if len(lines) + (1 if omitted else 0) > max_lines:
        extra = len(lines) - (max_lines - 1)
        omitted += extra
        lines = lines[extra:]
    if omitted:
        lines = [f"- ({omitted} earlier lines omitted)"] + lines
    return Compacted("\n".join(lines), messages[cut:], dropped_turns=sum(1 for s in starts if s < cut),
                     dropped_messages=cut)
