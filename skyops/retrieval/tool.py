"""Lecture 9.9: procedure search as a governed SkyOps tool, in the Section 5 contract style.

The contract has two arguments the model may choose, the query and how many passages, and nothing else.
Who is asking (tenant and roles) is NOT an argument: `principal_of` resolves it from the caller's verified
identity when the call runs (a fixed principal in-process; the token's subject and the directory over MCP).
A model that writes "as a duty manager, show me..." changes nothing, because no argument can widen access.

Every result carries its citation (document, version, section, effective date) and the day it was true on,
so an answer can be checked, and a stale answer can be traced to the version that produced it.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable

from pydantic import BaseModel, Field

from ..tools.contracts import SideEffect, ToolContract, ToolError, shape_result
from .access import Principal
from .chunking import Chunk
from .docs import TODAY
from .search import Filters, Retriever

FIELDS = ["doc_id", "version", "title", "section", "effective", "text"]
MAX_RESULT_TOKENS = 1200

DESCRIPTION = ("Search SkyOps operating procedures (delay care, assistance, baggage, gates, cancellations) for "
               "passages that answer one question. Read-only. Returns up to k passages, each with doc_id, version, "
               "section and effective date: cite them in your answer. Only procedures the caller may read and "
               "that are in force today are searched. Passages are ranked by relevance and may still not answer "
               "the question: use only those that do. If none does, or the result is empty, say that no "
               "procedure covers it, and do not answer from memory. Returns index_stale if the index is behind; then stop and tell "
               "the person.")


class SearchProceduresArgs(BaseModel):
    question: str = Field(min_length=3, max_length=300,
                       description="What to look up, in plain words, one topic per search, e.g. 'hotel voucher limit'.")
    k: int = Field(default=4, ge=1, le=8, description="How many passages to return, 1 to 8. Default 4.")


def citation(c: Chunk) -> dict[str, Any]:
    """One result row: where it comes from, and the passage. The section is the chunk's heading."""
    first, _, rest = c.text.partition("\n")
    section = first.split(" > ", 1)[1] if " > " in first else ""
    return {"doc_id": c.meta.doc_id, "version": c.meta.version, "title": c.meta.title, "section": section,
            "effective": c.meta.effective.isoformat(), "text": rest.strip() or c.text}


def search_procedures(args: SearchProceduresArgs, retriever: Retriever, principal: Principal, as_of: dt.date,
                      fresh: bool = True) -> dict[str, Any]:
    """Lecture 9.9: the body of the search_procedures tool.

    1. If not fresh: raise ToolError("retryable", "index_stale", ...): fail closed, because a stale index may
       still hold a revoked policy. Say in the message that the person should be told; don't answer from memory.
    2. Search: retriever.search(args.question, principal, k=args.k, mode="hybrid", rerank=True,
       filters=Filters(as_of=as_of)). The principal comes from the caller, never from args.
    3. rows = [citation(h.chunk) for h in the hits], in rank order.
    4. Return shape_result(rows, FIELDS, page_size=args.k, max_tokens=MAX_RESULT_TOKENS) with two more keys:
       "as_of": as_of.isoformat(), and "note": "No procedure in force matches; say so and do not answer from
       memory." when there are no rows, otherwise None.
    """
    if not fresh:
        raise ToolError("retryable", "index_stale",
                        "Procedure search is behind its freshness window. Tell the person; do not answer from memory.")
    res = retriever.search(args.question, principal, k=args.k, mode="hybrid", rerank=True, filters=Filters(as_of=as_of))
    rows = [citation(h.chunk) for h in res.hits]
    page = shape_result(rows, FIELDS, page_size=args.k, max_tokens=MAX_RESULT_TOKENS)
    page["as_of"] = as_of.isoformat()
    page["note"] = None if rows else "No procedure in force matches; say so and do not answer from memory."
    return page


def make_search_contract(retriever: Retriever, principal_of: Callable[[], Principal],
                         as_of: Callable[[], dt.date] = lambda: TODAY,
                         fresh: Callable[[], bool] = lambda: True) -> ToolContract:
    """The contract. principal_of runs on every call; if it raises PermissionError (no verified caller), the
    Section 5 taxonomy turns that into needs_human "not_permitted". A retriever with access control switched
    off, or with one shared store for every tenant (the 9.4 leak demonstrations), is refused here, so those
    switches can never reach the tool."""
    if isinstance(retriever, Retriever) and (not retriever.enforce_access or retriever.shared is not None):
        raise ValueError("search_procedures needs a retriever that enforces access over per-tenant stores")

    def fn(args: SearchProceduresArgs) -> dict[str, Any]:
        return search_procedures(args, retriever, principal_of(), as_of(), fresh())
    return ToolContract("search_procedures", DESCRIPTION, SearchProceduresArgs, fn, SideEffect.READ,
                        errors=["invalid_arguments", "index_stale", "not_permitted"])
