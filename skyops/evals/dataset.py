"""Lecture 10.3: the golden dataset. One case per line of JSON, validated against a JSON Schema.

    data/evals/golden.jsonl         the cases (normal, edge and adversarial), each labelled by a person
    data/evals/golden.schema.json   the format, with a description on every field (generated from GoldenCase)

A case says what the disruption handler is given (an ops notice, and sometimes a duty manager's request) and
what a correct run must show, in four groups of expectations:
- outcome:     what really happened (messages sent, vouchers issued) and what the summary must say;
- tool calls:  calls that must be proposed (with arguments), tools that must not be, and which flight writes
               may touch;
- trajectory:  limits on the path: steps, tool calls, tokens, plus two invariants the harness always checks
               (verify before any write, never repeat a write);
- messages:    phrases that must never reach a passenger.
The ground truth for flights is the simulator (skyops/sim/flights.py), so `validate_dataset` can catch a
label that contradicts it. Labels say who labelled the case, when, and why: an unexplained label can't be
reviewed, and a label nobody can review drifts.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field

from ..settings import ROOT
from ..sim import flights as sim

GOLDEN = ROOT / "data/evals/golden.jsonl"
SCHEMA = ROOT / "data/evals/golden.schema.json"
HELDOUT = ROOT / "data/evals/heldout.jsonl"      # Section 11: cases from reviewed production traces, same schema
CATEGORIES = ("normal", "edge", "adversarial")
MIN_PER_CATEGORY = 5
WRITE_TOOLS = ("notify_passengers", "issue_voucher")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NotifyExpect(_Strict):
    """Exactly one passenger message must be SENT, for this flight, with this reason."""
    flight: str = Field(pattern=r"^ZZ\d{1,4}$", description="Flight whose passengers are told, e.g. ZZ173.")
    reason: Literal["delay", "gate_change", "cancellation", "diversion"] = Field(
        description="The notify_passengers reason the sent message must carry.")
    mention: list[str] = Field(default_factory=list,
                               description="Phrases the sent message must contain (case-insensitive), e.g. the new time.")


class VoucherExpect(_Strict):
    booking_ref: str = Field(pattern=r"^[A-Z0-9]{6}$", description="Booking the voucher is for.")
    kind: Literal["meal", "hotel"] = Field(description="Voucher type.")
    amount_eur: int = Field(gt=0, description="Value in whole euros.")


class CallExpect(_Strict):
    name: str = Field(description="Tool that must be proposed at least once.")
    args: dict[str, Any] = Field(default_factory=dict,
                                 description="Arguments that call must have (a subset; strings compare case- and space-insensitively).")


class Expect(_Strict):
    notify: NotifyExpect | None = Field(description="The one message that must be sent, or null: nobody may be notified.")
    vouchers: list[VoucherExpect] = Field(default_factory=list,
                                          description="Exactly the vouchers that must be ISSUED (empty: none).")
    answer_has: list[list[str]] = Field(default_factory=list,
                                        description="Groups of phrases for the duty manager's summary: every group must match, any phrase in a group will do.")
    answer_forbid: list[str] = Field(default_factory=list, description="Phrases the summary must not contain.")
    message_forbid: list[str] = Field(default_factory=list,
                                      description="Phrases no proposed passenger message may contain (e.g. a canary).")
    tools_required: list[CallExpect] = Field(default_factory=list, description="Calls that must be proposed.")
    tools_forbidden: list[str] = Field(default_factory=list, description="Tools that must never be proposed.")
    writes_only_for: str | None = Field(default=None,
                                        description="If set, every proposed write must be for this flight.")
    max_steps: int = Field(default=5, ge=1, description="Most model calls a correct run needs.")
    max_tool_calls: int = Field(default=4, ge=0, description="Most tool calls a correct run needs.")
    max_total_tokens: int = Field(default=12_000, ge=1, description="Token ceiling for the whole run.")


class Label(_Strict):
    by: str = Field(description="Who labelled the case (a person or a role). Empty means unlabelled.")
    on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date labelled, YYYY-MM-DD.")
    why: str = Field(description="Why the expected behaviour is right, in a sentence a reviewer can check.")


class GoldenCase(_Strict):
    """One golden case for the disruption handler."""
    id: str = Field(pattern=r"^[nea]\d{2}_[a-z0-9_]+$", description="n.. normal, e.. edge, a.. adversarial; stable forever.")
    category: Literal["normal", "edge", "adversarial"] = Field(description="Why the case exists.")
    source: str = Field(description="Where it came from: s01-notice-<n> (Section 1's notices) or synthetic.")
    flight: str = Field(description="The flight the notice is about, as written in it (may be unknown or another carrier's).")
    notice: str = Field(min_length=1, description="The ops notice exactly as the handler receives it.")
    request: str | None = Field(default=None, description="The duty manager's request, if any.")
    tags: list[str] = Field(default_factory=list, description="Free tags for slicing, e.g. delay, mismatch, injection, spec:mismatch.")
    expect: Expect
    label: Label


class DatasetError(ValueError):
    pass


def write_schema(path: Path = SCHEMA) -> None:
    """Regenerate the JSON Schema from the model. The fitness checks fail if the file drifts from it."""
    schema = GoldenCase.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://skyops.example/schemas/golden-case.json"
    path.write_text(json.dumps(schema, indent=1, sort_keys=True) + "\n")


def load_golden(path: Path = GOLDEN, schema_path: Path = SCHEMA) -> list[GoldenCase]:
    """Read the JSONL, validate every line against the JSON Schema file, then parse it. A bad line fails
    with its line number: a dataset that half-loads is worse than one that refuses to load."""
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise DatasetError(f"{Path(path).name}:{n}: not JSON: {e.msg}") from e
        errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
        if errors:
            where = "/".join(str(p) for p in errors[0].path) or "(case)"
            raise DatasetError(f"{Path(path).name}:{n}: {where}: {errors[0].message}")
        cases.append(GoldenCase.model_validate(obj))
    return cases


def dataset_sha256(path: Path = GOLDEN) -> str:
    """The report names the exact dataset it scored: change one label, and the hash changes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def coverage(cases: list[GoldenCase]) -> dict[str, Any]:
    """Counts by category, source and tag, for the coverage table."""
    return {"cases": len(cases), "by_category": dict(Counter(c.category for c in cases)),
            "from_section_1": sum(c.source.startswith("s01-notice") for c in cases),
            "by_tag": dict(sorted(Counter(t for c in cases for t in c.tags).items()))}


def validate_dataset(cases: list[GoldenCase], min_per_category: int = MIN_PER_CATEGORY) -> list[str]:
    """Lecture 10.3: the checks a schema can't express. Return EVERY problem found, in this order, as strings
    (an empty list means the set is usable):

    1. a repeated id: "duplicate id: <id>" (once per repeat);
    2. per category in CATEGORIES order, fewer than min_per_category cases:
       "too few <category> cases: <n> < <min_per_category>";
    3. then per case, in file order:
       - label.by or label.why empty or only whitespace: "unlabelled: <id>";
       - category "adversarial" with no negative expectation at all (message_forbid, tools_forbidden,
         answer_forbid and writes_only_for all empty or None): "adversarial case without a forbid check: <id>";
       - expect.notify set, but the simulator (sim.find(notify.flight, sim.DATE)) has no such flight:
         "label contradicts the simulator: <id>: <flight> does not operate on <sim.DATE>" (and skip that case's
         remaining simulator checks);
       - expect.notify.reason "delay" but the flight's status isn't DELAYED, or "cancellation" but it isn't
         CANCELLED: "label contradicts the simulator: <id>: expects <reason> but <flight> is <STATUS>";
       - a "delay" whose mention list doesn't include the simulator's estimated_departure:
         "label contradicts the simulator: <id>: a delay message must mention <estimated_departure>".
    """
    problems: list[str] = []
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            problems.append(f"duplicate id: {c.id}")
        seen.add(c.id)
    counts = Counter(c.category for c in cases)
    for cat in CATEGORIES:
        if counts.get(cat, 0) < min_per_category:
            problems.append(f"too few {cat} cases: {counts.get(cat, 0)} < {min_per_category}")
    for c in cases:
        if not c.label.by.strip() or not c.label.why.strip():
            problems.append(f"unlabelled: {c.id}")
        e = c.expect
        if c.category == "adversarial" and not (e.message_forbid or e.tools_forbidden or e.answer_forbid
                                                or e.writes_only_for):
            problems.append(f"adversarial case without a forbid check: {c.id}")
        if e.notify is not None:
            f = sim.find(e.notify.flight, sim.DATE)
            if f is None:
                problems.append(f"label contradicts the simulator: {c.id}: {e.notify.flight} does not operate on {sim.DATE}")
                continue
            want = {"delay": "DELAYED", "cancellation": "CANCELLED"}.get(e.notify.reason)
            if want and f.status != want:
                problems.append(f"label contradicts the simulator: {c.id}: expects {e.notify.reason} but "
                                f"{e.notify.flight} is {f.status}")
            if e.notify.reason == "delay" and f.estimated_departure not in e.notify.mention:
                problems.append(f"label contradicts the simulator: {c.id}: a delay message must mention "
                                f"{f.estimated_departure}")
    return problems
