"""Lab 3.3: detection signals for the five breaks. Every incident in this course ends with
"the signal that should have caught it". These are the first five, written as plain functions over
a trace, the tool latencies and the outbox. Section 20 turns them into metrics and alerts.
"""
from __future__ import annotations

import json
from collections import Counter

from skyops.trace import Trace
from skyops.model.errors import LabNotDone


def detect_loop(trace: Trace, repeats: int = 3) -> bool:
    """True if the same tool was called with the same arguments `repeats` or more times in one run."""
    raise LabNotDone("Lecture 3.3: implement detect_loop(). Hints: labs/s03/README.md, lecture 3.3.")


def detect_bad_arguments(trace: Trace) -> int:
    """How many tool calls in the run were rejected for invalid arguments."""
    raise LabNotDone("Lecture 3.3: implement detect_bad_arguments(). Hints: labs/s03/README.md, lecture 3.3.")


def detect_slow_tool(latencies: list[tuple[str, int]], budget_ms: int = 2_000) -> bool:
    """True if any single tool call took longer than its latency budget."""
    raise LabNotDone("Lecture 3.3: implement detect_slow_tool(). Hints: labs/s03/README.md, lecture 3.3.")


def detect_duplicate_side_effects(sent: list[dict[str, str]]) -> int:
    """How many outbound messages were exact repeats of an earlier one in the same run."""
    raise LabNotDone("Lecture 3.3: implement detect_duplicate_side_effects(). Hints: labs/s03/README.md, lecture 3.3.")


def detect_canary(sent: list[dict[str, str]], canaries: list[str]) -> bool:
    """True if any planted canary string crossed into an outbound message."""
    raise LabNotDone("Lecture 3.3: implement detect_canary(). Hints: labs/s03/README.md, lecture 3.3.")
