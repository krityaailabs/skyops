"""Tool contracts (Section 5): a tool is an API for a caller that can't ask questions.

A ToolContract says, in code, everything the loop and the model need to know about a tool:
- what it's called and what it's for (name, description): the model reads these as a prompt (5.2);
- what arguments it accepts (a Pydantic model; the same model produces the JSON schema);
- its side-effect class: read, reversible write or irreversible write (5.4). The class drives
  approvals (Section 18), retries (Section 15) and logging;
- how it fails: every error is a ToolError in one of three categories (5.6);
- for writes: how its idempotency key is derived, by code, never by the model (5.5).

`as_tool(contract)` turns a contract into a Section 4 loop Tool, so the loop itself doesn't change.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel, ValidationError

from ..agent.loop import Tool
from ..logs import get_logger, redact
from ..model.errors import LabNotDone

log = get_logger("skyops.tools")

Category = Literal["retryable", "non_retryable", "needs_human"]


class SideEffect(str, Enum):
    READ = "read"                              # no change anywhere: safe to retry, cache, run in parallel
    REVERSIBLE_WRITE = "reversible_write"      # changes state that can be undone (hold a seat, draft a message)
    IRREVERSIBLE_WRITE = "irreversible_write"  # can't be taken back (send a message, issue money)


class ToolError(Exception):
    """The only exception a contracted tool may raise. `code` is machine-readable and stable."""

    def __init__(self, category: Category, code: str, message: str):
        super().__init__(f"{category}:{code}: {message}")
        self.category, self.code, self.message = category, code, message

    @property
    def retryable(self) -> bool:
        return self.category == "retryable"

    def as_result(self) -> dict[str, Any]:
        return {"error": self.code, "category": self.category, "message": self.message, "retryable": self.retryable}


@dataclass
class ToolContract:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], dict[str, Any]]
    side_effect: SideEffect
    errors: list[str] = field(default_factory=list)          # error codes this tool can return, documented
    key_fields: list[str] = field(default_factory=list)      # writes: the arguments that define "the same action"
    supports_dry_run: bool = False
    precheck: Callable[[BaseModel], None] | None = None      # writes: rules checked before acting, in BOTH modes

    @property
    def is_write(self) -> bool:
        return self.side_effect is not SideEffect.READ


# ---------- 5.5 idempotency and dry run ----------

def idempotency_key(contract: ToolContract, args: BaseModel, scope: str) -> str:
    """Lecture 5.5: the key for one intended action, derived by code from the arguments that define it.

    sha256 of canonical JSON of [tool name, scope, {field: value for field in contract.key_fields}],
    first 24 hex characters. The scope is who/what the action belongs to (a run, a ticket, a disruption),
    so the same message to the same flight in two different disruptions is two actions, not one.
    Fields are read from args.model_dump(mode="json") (dates become ISO strings). String values are
    normalised with normalise_text, so a re-send that only differs in spacing or case has the same key.
    """
    values = args.model_dump(mode="json")
    picked = {f: normalise_text(values[f]) if isinstance(values[f], str) else values[f] for f in contract.key_fields}
    body = json.dumps([contract.name, scope, picked], sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:24]


def normalise_text(s: str) -> str:
    """Collapse whitespace and ignore case, for key purposes only (the action itself uses the original)."""
    return " ".join(s.split()).casefold()


@dataclass
class IdempotencyStore:
    """Remembers every completed write by key: its result and the arguments it ran with. Dry runs remember
    what they previewed, so a preview of a repeat says it would replay. In-memory here; Section 15 makes it durable."""
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    args: dict[str, dict[str, Any]] = field(default_factory=dict)
    previewed: set[str] = field(default_factory=set)


def execute_write(contract: ToolContract, args: BaseModel, store: IdempotencyStore, scope: str,
                  dry_run: bool = False) -> dict[str, Any]:
    """Lecture 5.5: run a write at most once per key, or preview it.

    Order: 1. key  2. precheck (both modes: a dry run that skips the rules is a lie)  3. dry run or store.
    - dry_run: never call contract.fn. A contract without supports_dry_run raises
      ToolError("non_retryable", "dry_run_unsupported", ...). Otherwise return {"dry_run": True,
      "would": <contract.name>, "would_replay": <key already stored or already previewed>,
      "args": args.model_dump(mode="json"), "idempotency_key": key}, and remember the key as previewed.
    - key already stored: if the arguments differ from the stored ones, raise
      ToolError("non_retryable", "idempotency_conflict", ...): same action key, different details.
      Otherwise return {"replayed": True, **stored result}, without calling contract.fn.
    - otherwise call contract.fn(args). If it raises, nothing is stored (the exception propagates).
      If it returns a dict with "error", nothing is stored either. Otherwise store the result and the
      arguments under the key. Return {"replayed": False, "idempotency_key": key, **result}.
    """
    key = idempotency_key(contract, args, scope)
    if contract.precheck is not None:
        contract.precheck(args)
    sent = args.model_dump(mode="json")
    if dry_run:
        if not contract.supports_dry_run:
            raise ToolError("non_retryable", "dry_run_unsupported", f"{contract.name} has no dry-run mode")
        would_replay = key in store.results or key in store.previewed
        store.previewed.add(key)
        return {"dry_run": True, "would": contract.name, "would_replay": would_replay, "args": sent,
                "idempotency_key": key}
    if key in store.results:
        if store.args[key] != sent:
            raise ToolError("non_retryable", "idempotency_conflict",
                            f"{contract.name} already ran for this action with different details; "
                            "the original result stands")
        return {"replayed": True, **store.results[key]}
    out = contract.fn(args)
    if "error" not in out:
        store.results[key] = {"idempotency_key": key, **out}
        store.args[key] = sent
    return {"replayed": False, "idempotency_key": key, **out}


# ---------- 5.6 error taxonomy ----------

def classify_exception(e: Exception) -> ToolError:
    """Lecture 5.6: map anything a tool's dependencies raise onto the taxonomy.

    - ToolError: unchanged.
    - TimeoutError, ConnectionError, or anything with a true `retryable` attribute: retryable, code "unavailable".
    - PermissionError: needs_human, code "not_permitted" (a person must grant or decide; the model can't).
    - pydantic ValidationError: non_retryable, code "invalid_arguments".
    - anything else: non_retryable, code "internal_error".
    The message is "<ExceptionType>" only: exception text never reaches the model (lecture 4.5).
    """
    if isinstance(e, ToolError):
        return e
    name = type(e).__name__
    if isinstance(e, (TimeoutError, ConnectionError)) or getattr(e, "retryable", False):
        return ToolError("retryable", "unavailable", name)
    if isinstance(e, PermissionError):
        return ToolError("needs_human", "not_permitted", name)
    if isinstance(e, ValidationError):
        return ToolError("non_retryable", "invalid_arguments", name)
    return ToolError("non_retryable", "internal_error", name)


# ---------- 5.7 shaping results ----------

def estimate_tokens(obj: Any) -> int:
    """About four characters per token, as in Section 1. Good enough for a cap; the bill uses provider counts."""
    return math.ceil(len(json.dumps(obj)) / 4)


def shape_result(rows: list[dict[str, Any]], fields: list[str], page_size: int, cursor: int = 0,
                 max_tokens: int = 800) -> dict[str, Any]:
    """Lecture 5.7: return only what the model needs, a page at a time, under a token cap.

    1. Keep only `fields` from each row (in that order).
    2. Take rows[cursor : cursor + page_size].
    3. While estimate_tokens(items) > max_tokens and more than one item remains, drop the last item.
    4. If a single item is still over the cap, cut its longest string values (keeping a "…" at the end)
       until it fits, and mark the result "oversized": True. The cap is never exceeded.
    Return {"items", "total": len(rows), "next_cursor": cursor + len(items), or None when no rows remain
    after this page, "truncated": True if the cap dropped or cut anything, "oversized": True/False}.
    """
    picked = [{f: r.get(f) for f in fields} for r in rows]
    items = picked[cursor:cursor + page_size]
    page_len = len(items)
    while estimate_tokens(items) > max_tokens and len(items) > 1:
        items = items[:-1]
    oversized = False
    if items and estimate_tokens(items) > max_tokens:
        item, oversized = dict(items[0]), True
        while estimate_tokens([item]) > max_tokens:
            longest = max((k for k, v in item.items() if isinstance(v, str) and len(v) > 1),
                          key=lambda k: len(item[k]), default=None)
            if longest is None:
                break
            v = item[longest]
            item[longest] = v[:max(1, len(v) // 2)].rstrip("…") + "…"
        items = [item]
    nxt = cursor + len(items)
    return {"items": items, "total": len(rows), "next_cursor": nxt if nxt < len(rows) else None,
            "truncated": len(items) < page_len or oversized, "oversized": oversized}


# ---------- 5.8 / 5.9 contract lint ----------

VAGUE_NAMES = {"run", "do", "execute", "process", "handle", "query", "tool", "helper", "action", "manage", "update"}
SECRET_FIELDS = re.compile(r"(^|_)(password|passwd|secret|token|api_key|apikey|card_number|cvv)($|_)", re.I)
RAW_QUERY_FIELDS = re.compile(r"^(sql|query|command|script|expression)$", re.I)


def lint_contract(c: ToolContract) -> list[str]:
    """Lectures 5.8 and 5.9: problems a reviewer would flag, found by code. Returns problem codes:

    - "vague_name": the name is in VAGUE_NAMES, or has no underscore-separated verb and noun (e.g. "flights").
    - "short_description": description under 40 characters.
    - "no_side_effect_in_description": a write whose description doesn't say "irreversible" or "reversible".
    - "side_effect_mismatch": the description says "irreversible" for a reversible write, or only "reversible"
      for an irreversible one.
    - "undocumented_field": an argument field with no description in the schema.
    - "free_form_query": an argument named like sql/query/command/script/code/expression.
    - "secret_argument": an argument whose name matches SECRET_FIELDS.
    - "write_without_key": a write with no key_fields.
    - "bad_key_field": a key field that isn't an argument of the tool.
    - "no_documented_errors": errors list is empty.
    """
    problems = []
    if c.name.lower() in VAGUE_NAMES or "_" not in c.name:
        problems.append("vague_name")
    if len(c.description) < 40:
        problems.append("short_description")
    d = c.description.lower()
    if c.is_write and "reversible" not in d:
        problems.append("no_side_effect_in_description")
    elif (c.side_effect is SideEffect.REVERSIBLE_WRITE and "irreversible" in d) or \
            (c.side_effect is SideEffect.IRREVERSIBLE_WRITE and "irreversible" not in d):
        problems.append("side_effect_mismatch")
    props = c.args_model.model_json_schema().get("properties", {})
    if any("description" not in p for p in props.values()):
        problems.append("undocumented_field")
    if any(RAW_QUERY_FIELDS.match(n) for n in props):
        problems.append("free_form_query")
    if any(SECRET_FIELDS.search(n) for n in props):
        problems.append("secret_argument")
    if c.is_write and not c.key_fields:
        problems.append("write_without_key")
    if any(f not in props for f in c.key_fields):
        problems.append("bad_key_field")
    if not c.errors:
        problems.append("no_documented_errors")
    return problems


# ---------- integration with the Section 4 loop ----------

def as_tool(c: ToolContract, store: IdempotencyStore | None, scope: str, dry_run: bool = False) -> Tool:
    """Wrap a contract as a loop Tool: writes go through execute_write; every failure becomes a taxonomy
    result the model can read, with "category" telling it whether to retry, change the call, or stop.

    The scope is required: pass the run, ticket or disruption the actions belong to.
    For irreversible writes, a retryable failure (timeout, connection lost) means the outcome is UNKNOWN:
    the action may have happened. It is reported as needs_human "outcome_unknown", never as retryable,
    until Section 15 makes the downstream honour the idempotency key.
    """
    store = store if store is not None else IdempotencyStore()

    def run(args: BaseModel) -> dict[str, Any]:
        try:
            if c.is_write:
                return execute_write(c, args, store, scope, dry_run=dry_run)
            return c.fn(args)
        except LabNotDone:
            raise
        except Exception as e:  # noqa: BLE001 - contracted tools fail only through the taxonomy
            err = classify_exception(e)
            if not isinstance(e, ToolError):
                log.error("tool failed", extra={"tool": c.name, "error_type": type(e).__name__, "detail": redact(str(e))})
            if err.retryable and c.side_effect is SideEffect.IRREVERSIBLE_WRITE:
                err = ToolError("needs_human", "outcome_unknown",
                                f"{c.name} may or may not have completed ({err.message}); check before trying again")
            return err.as_result()
    return Tool(c.name, c.args_model, run, c.description)
