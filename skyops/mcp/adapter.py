"""Lecture 7.2: Section 5 contracts, served over MCP without losing anything on the way.

A ToolContract already holds everything an MCP tool needs: a name, a description the model reads, an
argument model (its JSON Schema becomes inputSchema), a side-effect class, documented errors and key fields.
This module turns one into an MCPServer tool:

    - inputSchema       from the argument model, field by field (descriptions, formats, enums, patterns)
    - annotations       from the side-effect class (annotations_for): hints for hosts, never security
    - execution         through Section 5's as_tool, so writes keep their idempotency keys and every
                        failure is a taxonomy result
    - results           to_call_result: success -> structuredContent + a text copy; taxonomy error ->
                        isError true, with the error object as structuredContent so a client can act on
                        "category" without parsing prose
    - scope             the idempotency scope comes from the caller (the token's subject) and the day,
                        not from a global: two duty managers are two scopes
"""
from __future__ import annotations

import inspect
import json
from typing import Annotated, Any, Callable

from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp_types import CallToolResult, TextContent, ToolAnnotations

from ..sim.flights import DATE as OPERATING_DAY
from ..tools.contracts import IdempotencyStore, SideEffect, ToolContract, as_tool


def annotations_for(contract: ToolContract) -> ToolAnnotations:
    """Lecture 7.2: the MCP tool annotations a contract implies.

    READ               read_only_hint=True,  destructive_hint=False, idempotent_hint=True,  open_world_hint=False
    REVERSIBLE_WRITE   read_only_hint=False, destructive_hint=False, idempotent_hint=<has key_fields>, open_world_hint=False
    IRREVERSIBLE_WRITE read_only_hint=False, destructive_hint=True,  idempotent_hint=<has key_fields>, open_world_hint=True
    (An irreversible write reaches outside SkyOps: messages to people, vouchers they can spend.)
    Hosts may use these to decide what needs a person's approval; clients treat them as untrusted hints
    unless they trust this server. They are never this server's security.
    """
    se = contract.side_effect
    if se is SideEffect.READ:
        return ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
    keyed = bool(contract.key_fields)
    if se is SideEffect.REVERSIBLE_WRITE:
        return ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=keyed, open_world_hint=False)
    return ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=keyed, open_world_hint=True)


def to_call_result(result: dict[str, Any]) -> CallToolResult:
    """Lecture 7.2: a contracted tool's dict result as an MCP CallToolResult.

    - A taxonomy error (a dict with an "error" key, from ToolError.as_result): is_error=True,
      structured_content=the dict, and one text block "<error code>: <message>" for clients that read text.
    - Anything else: is_error=False, structured_content=the dict, and one text block with the same dict as
      compact JSON (sorted keys), for clients that don't read structured content.
    Dates and other non-JSON values become strings (json.dumps default=str).
    """
    data = json.loads(json.dumps(result, default=str))
    if "error" in data:
        text = f"{data['error']}: {data.get('message', '')}"
        return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=data, is_error=True)
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=data, is_error=False)


def caller_scope() -> str:
    """Idempotency scope for writes made over MCP: who is acting, and on which operating day at the hub."""
    tok = get_access_token()
    who = f"{tok.client_id}:{tok.subject}" if tok else "anonymous"
    return f"{who}:{OPERATING_DAY}"


def contract_tool(c: ToolContract, store: IdempotencyStore, scope_of: Callable[[], str] = caller_scope) -> Callable:
    """Build the function MCPServer registers: keyword parameters copied from the argument model, so the
    tool's inputSchema is the contract's schema, and a body that runs the contract through as_tool."""
    fields = c.args_model.model_fields
    params = [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[f.annotation, f],
                                default=inspect.Parameter.empty if f.is_required() else f.default)
              for n, f in fields.items()]

    def tool(**kwargs: Any) -> CallToolResult:
        try:
            args = c.args_model(**kwargs)              # validators run here (e.g. "zz 214" -> "ZZ214")
        except Exception as e:                          # noqa: BLE001 - becomes a taxonomy result
            msg = e.errors(include_url=False)[0]["msg"] if hasattr(e, "errors") else type(e).__name__
            return to_call_result({"error": "invalid_arguments", "category": "non_retryable", "message": msg,
                                   "retryable": False})
        return to_call_result(as_tool(c, store, scope_of()).fn(args))

    tool.__signature__ = inspect.Signature(params, return_annotation=CallToolResult)
    tool.__annotations__ = {p.name: p.annotation for p in params} | {"return": CallToolResult}
    tool.__name__ = c.name
    return tool


def register(srv: MCPServer, contracts: list[ToolContract], store: IdempotencyStore | None = None,
             scope_of: Callable[[], str] = caller_scope) -> IdempotencyStore:
    """Add every contract to the server as a tool. One idempotency store is shared by all of them."""
    store = store if store is not None else IdempotencyStore()
    for c in contracts:
        srv.add_tool(contract_tool(c, store, scope_of), name=c.name, description=c.description,
                     annotations=annotations_for(c))
    return store
