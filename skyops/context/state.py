"""Lecture 8.2: conversation state versus durable state. Where each piece of SkyOps agent state lives.

Three homes, three lifetimes:
    prompt      rebuilt for every model call; the model sees it; lost when the call ends
    checkpoint  the run's own state between steps (transcript, step count, pending approval); survives a
                crash of this process; scoped to one run or thread (Section 15 makes it durable)
    database    business records and long-term memory; outlives runs; scoped by user, tenant or entity,
                with its own retention and access control

Rule of thumb: if losing it would change what happens in the world, it can't live only in the prompt.
"""
from __future__ import annotations

from typing import Literal

Home = Literal["prompt", "checkpoint", "database"]

STATE_MAP: list[tuple[str, Home, str]] = [
    ("The pinned safety rules", "prompt",
     "Sent verbatim with every call; their source of truth is versioned config, never the transcript."),
    ("The tool schemas for this run", "prompt",
     "Rebuilt from the catalog on every call; the catalog itself is versioned code."),
    ("The rolling summary of older turns", "checkpoint",
     "Rendered into the prompt, but it replaces messages, so it must survive a restart with the transcript."),
    ("The scratchpad: plan, facts, open questions", "checkpoint",
     "Working memory for this run only; shown in the prompt, saved with each step, gone when the run ends."),
    ("The full transcript of this run", "checkpoint",
     "The record of what was asked and done; the prompt is only a view of it."),
    ("The step and token counters", "checkpoint",
     "Budgets must survive a restart, or a crash resets the limits."),
    ("A pending approval for an over-limit voucher", "checkpoint",
     "The run pauses on it and resumes when a person decides (Section 18); the request itself is also audited."),
    ("Idempotency keys of writes already done", "database",
     "Must outlive the run and be shared by every replica, or a retry sends twice."),
    ("Messages sent to passengers", "database",
     "Business records with their own retention and audit; never only in a transcript."),
    ("A duty manager's stated preference", "database",
     "Long-term memory: scoped to that user, with a retention period and a deletion path."),
    ("Today's flight statuses", "database",
     "Owned by the operations system; fetch fresh through tools, never remember them across runs."),
    ("The audit trail of tool calls", "database",
     "Append-only, retained by policy, readable by people who never see a prompt."),
]
