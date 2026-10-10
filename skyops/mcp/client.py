"""Lecture 7.5: an MCP client for the SkyOps host, and the bridge into the Section 4 loop.

    with connect({"flight-ops": target_a, "crew-ops": target_b}) as host:
        box = host.toolbox()                  # every server's tools, as one Section 4 ToolBox
        run_agent(question, model_client, box)

A target is either an MCPServer (in-process, no HTTP: for tests and labs) or an HttpTarget (Streamable HTTP
with a bearer token: the real thing). The loop stays synchronous, as in Section 4; the MCP clients run on
an event loop in a background thread (an anyio blocking portal), and each tool call crosses over to it.

What the host owns, and the server can't: which servers it connects, which of their tools the model sees
(allow-list, name collisions), and how every result and error looks to the model (map_call_result).
"""
from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx2
from anyio.from_thread import BlockingPortal, start_blocking_portal
from jsonschema import Draft202012Validator
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server import MCPServer
from mcp.shared.exceptions import MCPError
from mcp_types import CallToolResult, Implementation
from pydantic import BaseModel, ConfigDict, model_validator

from ..agent.loop import Tool, ToolBox
from skyops.model.errors import LabNotDone

CLIENT_INFO = Implementation(name="skyops-host", version="0.7.0")


@dataclass(frozen=True)
class HttpTarget:
    url: str                                   # e.g. https://mcp.skyops.example/mcp
    token: str | None = None                   # a bearer token issued FOR this server (audience = its URI)
    app: Any = None                            # tests and labs: an ASGI app served in memory (no network)
    verify: Any = True                         # TLS: True, or a CA bundle path (lecture 7.10)


def map_call_result(result: CallToolResult) -> dict[str, Any]:
    """Lecture 7.5: what the model sees for one MCP tool result, in the Section 5 shapes.

    - is_error with structured_content containing "error": return structured_content unchanged (a taxonomy
      error from a contracted server: the model can act on "category").
    - is_error without that: {"error": "tool_error", "category": "non_retryable", "message": <the text
      content, joined with spaces>, "retryable": False} (a server that only sends prose).
    - success with structured_content that is a dict: return it.
    - success otherwise: {"text": <the text content, joined with newlines>}.
    Never pass a server's raw exception text through anything else: it's untrusted text (Section 19).
    """
    raise LabNotDone("Lecture 7.5: implement map_call_result(). Hints: labs/s07/README.md, lecture 7.5.")


def protocol_error(e: Exception, challenge: dict[str, str] | None = None) -> dict[str, Any]:
    """A failure of the call itself (refused, unauthorised, transport down), not of the tool.
    `challenge` is the HTTP 401/403 the transport saw for this call, if any (the SDK client reports those
    as a generic error, so the host watches the HTTP responses itself)."""
    if challenge is not None:
        need = challenge.get("scope")
        return {"error": "not_authorized", "category": "needs_human", "retryable": False,
                "message": (f"This call needs the scope {need}; a person must grant it." if need else
                            f"The server refused this call ({challenge['status']}); a person must grant access.")}
    if isinstance(e, MCPError):
        return {"error": "protocol_error", "category": "non_retryable", "retryable": False,
                "message": f"MCP error {e.error.code}"}
    if isinstance(e, (TimeoutError, ConnectionError, httpx2.TransportError)):
        return {"error": "unavailable", "category": "retryable", "retryable": True, "message": type(e).__name__}
    return {"error": "internal_error", "category": "non_retryable", "retryable": False, "message": type(e).__name__}


def schema_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """A Pydantic model that validates arguments against a server's JSON Schema (2020-12) and reports that
    same schema to the model, unchanged. The loop's validate_call works on MCP tools exactly as on local ones."""
    validator = Draft202012Validator(schema)

    class Args(BaseModel):
        model_config = ConfigDict(extra="allow")

        @model_validator(mode="before")
        @classmethod
        def _check(cls, data: Any) -> Any:
            errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
            if errors:
                e = errors[0]
                raise ValueError(f"{'.'.join(str(p) for p in e.path) or 'arguments'}: {e.message}")
            return data

        @classmethod
        def model_json_schema(cls, *a: Any, **k: Any) -> dict[str, Any]:
            return schema

    Args.__name__ = Args.__qualname__ = f"{name}_args"
    return Args


@dataclass
class Host:
    portal: BlockingPortal
    clients: dict[str, Client]
    allow: dict[str, set[str]] | None = None   # server -> tools the model may see (None: all)
    calls: list[dict[str, Any]] = field(default_factory=list)
    challenges: list[dict[str, str]] = field(default_factory=list)   # HTTP 401/403 seen, newest last

    def list_tools(self) -> dict[str, list]:
        return {s: self.portal.call(c.list_tools).tools for s, c in self.clients.items()}

    def toolbox(self) -> ToolBox:
        """One ToolBox from every connected server. A name offered by two servers is a configuration
        error, not something to resolve silently: the host must not guess which server the model meant."""
        box, owner = ToolBox(), {}
        for server, tools in self.list_tools().items():
            for t in tools:
                if self.allow is not None and t.name not in self.allow.get(server, set()):
                    continue
                if t.name in owner:
                    raise ValueError(f"tool {t.name} is offered by {owner[t.name]} and {server}; prefix one")
                owner[t.name] = server
                box.add(Tool(t.name, schema_model(t.name, t.input_schema), self._caller(server, t.name),
                             t.description or ""))
        return box

    def _caller(self, server: str, name: str):
        def call(args: BaseModel) -> dict[str, Any]:
            payload = args.model_dump(mode="json", exclude_none=True)
            seen = len(self.challenges)
            try:
                out = map_call_result(self.portal.call(self.clients[server].call_tool, name, payload))
            except Exception as e:                       # noqa: BLE001 - the model gets a shape, never a traceback
                mine = [c for c in self.challenges[seen:] if c.get("name") == name]
                out = protocol_error(e, mine[-1] if mine else None)
            self.calls.append({"server": server, "tool": name, "ok": "error" not in out})
            return out
        return call


def _challenge(header: str) -> dict[str, str]:
    """Parse a WWW-Authenticate Bearer challenge: error, scope, resource_metadata."""
    import re
    return dict(re.findall(r'(\w+)="([^"]*)"', header))


def _client(target: MCPServer | HttpTarget, challenges: list[dict[str, str]]) -> Client:
    if isinstance(target, MCPServer):
        return Client(target, client_info=CLIENT_INFO)
    headers = {"authorization": f"Bearer {target.token}"} if target.token else {}
    transport = httpx2.ASGITransport(app=target.app) if target.app is not None else None

    async def watch(r: httpx2.Response) -> None:
        if r.status_code in (401, 403):
            challenges.append({"status": str(r.status_code), "name": r.request.headers.get("mcp-name", ""),
                               **_challenge(r.headers.get("www-authenticate", ""))})

    http = httpx2.AsyncClient(transport=transport, headers=headers, verify=target.verify, timeout=30,
                              event_hooks={"response": [watch]})
    return Client(streamable_http_client(target.url, http_client=http), client_info=CLIENT_INFO)


@contextmanager
def connect(targets: dict[str, MCPServer | HttpTarget], allow: dict[str, set[str]] | None = None) -> Iterator[Host]:
    with start_blocking_portal() as portal, ExitStack() as stack:
        for t in targets.values():
            if isinstance(t, HttpTarget) and t.app is not None:     # in-memory app: run its lifespan too
                stack.enter_context(portal.wrap_async_context_manager(t.app.router.lifespan_context(t.app)))
        challenges: list[dict[str, str]] = []
        clients = {name: stack.enter_context(portal.wrap_async_context_manager(_client(t, challenges)))
                   for name, t in targets.items()}
        yield Host(portal, clients, allow, challenges=challenges)


def summarise(host: Host) -> str:
    return json.dumps(host.calls)
