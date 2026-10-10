"""Lab 3.3: detection signals for the five breaks. Every incident in this course ends with
"the signal that should have caught it". These are the first five, written as plain functions over
a trace, the tool latencies and the outbox. Section 20 turns them into metrics and alerts.
"""
from __future__ import annotations

import json
from collections import Counter

from skyops.trace import Trace


def detect_loop(trace: Trace, repeats: int = 3) -> bool:
    """True if the same tool was called with the same arguments `repeats` or more times in one run."""
    calls = Counter(json.dumps([e["name"], e["args"]], sort_keys=True) for e in trace.events if e["kind"] == "tool")
    return any(n >= repeats for n in calls.values())


def detect_bad_arguments(trace: Trace) -> int:
    """How many tool calls in the run were rejected for invalid arguments."""
    return sum(1 for e in trace.events if e["kind"] == "tool" and not e["ok"] and "invalid_arguments" in e["summary"])


def detect_slow_tool(latencies: list[tuple[str, int]], budget_ms: int = 2_000) -> bool:
    """True if any single tool call took longer than its latency budget."""
    return any(ms > budget_ms for _, ms in latencies)


def detect_duplicate_side_effects(sent: list[dict[str, str]]) -> int:
    """How many outbound messages were exact repeats of an earlier one in the same run."""
    seen, dups = set(), 0
    for m in sent:
        key = (m["flight_number"], m["message"])
        dups += key in seen
        seen.add(key)
    return dups


def detect_canary(sent: list[dict[str, str]], canaries: list[str]) -> bool:
    """True if any planted canary string crossed into an outbound message."""
    return any(c in m["message"] for m in sent for c in canaries)
