"""Lecture 7.9: an MCP gateway in front of several servers: header routing, central auth, policy, audit.

    host ──(token for the GATEWAY)──> gateway ──(a new token for EACH backend)──> flight-ops / crew-ops

What it does on every request, in order:
  1. authenticate: a bearer token whose audience is the gateway (else 401 + resource-metadata pointer)
  2. decide (decide()): from headers only - protocol version, Mcp-Method, Mcp-Name - and the token's scopes:
     forward to one backend, fan out (lists), answer itself (discover), or refuse with the right error
  3. forward: the request body untouched; the client's token is NEVER forwarded (that would be token
     passthrough). The gateway gets a token minted for that backend, carrying the same subject and the
     same client, with the gateway named as the actor (`act`): the backend's idempotency scope is then the
     same whether a call came direct or through the gateway.
  4. audit: one line per request - who, what method and name, which backend, what happened. No arguments
     and no results: they can hold personal data, and the backend logs its own business events. A
     resource read is recorded by its pattern (skyops://bookings/*), never by the booking reference.
Lists are the one place the gateway reads response bodies: it merges the backends' tools and keeps the
strictest cache hints. A backend that refuses the caller (401/403) is left out of the merge and the audit
line says so; a backend that fails (5xx, bad JSON, unreachable) fails the whole listing with 502, because
a host that silently loses half its tools makes worse decisions than one that sees an error.
Discovery it answers itself.

The backends here run in memory (ASGI); in production they're URLs. Token exchange is simulated by minting
with the course test key; a real gateway asks the authorization server (OAuth token exchange, RFC 8693).
"""
from __future__ import annotations

import base64
import json
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

import httpx2

from . import auth as A
from skyops.model.errors import LabNotDone

GATEWAY = "https://gateway.skyops.example/mcp"
CURRENT = "2026-07-28"
_METADATA = f'{GATEWAY.rsplit("/mcp", 1)[0]}/.well-known/oauth-protected-resource/mcp'
LISTS = {"tools/list": "tools", "prompts/list": "prompts", "resources/templates/list": "resourceTemplates",
         "resources/list": "resources"}


@dataclass(frozen=True)
class Backend:
    name: str
    url: str                     # the backend's canonical URI: the audience of the tokens minted for it
    owns: tuple[str, ...]        # Mcp-Name patterns: tool and prompt names, resource URIs
    baseline: tuple[str, ...] = ("flights:read",)       # scopes every request to this backend needs


@dataclass(frozen=True)
class Decision:
    action: str                  # "forward" | "fan_out" | "discover" | "refuse"
    backend: str | None = None
    status: int = 200
    code: int | None = None      # JSON-RPC error code, when refusing
    reason: str = ""
    scope: list[str] = field(default_factory=list)      # scopes to challenge for, on a 403


BACKENDS = (
    Backend("flight-ops", A.RESOURCE, ("list_departures", "get_flight_status", "notify_passengers",
                                       "issue_voucher", "disruption_brief", "skyops://bookings/*")),
    Backend("crew-ops", A.CREW_RESOURCE, ("get_crew_status",), ("crew:read",)),
)


def decide(method: str | None, name: str | None, version: str | None, scopes: list[str],
           backends: tuple[Backend, ...] = BACKENDS) -> Decision:
    """Lecture 7.9: one routing and policy decision, from headers and the token's scopes only.

    In this order:
    1. version != CURRENT (missing, older or unknown): refuse, status 400, code -32022, reason
       "unsupported protocol version" (this gateway serves the current revision only; lecture 6.6).
    2. method missing: refuse, 400, code -32020, reason "Mcp-Method missing".
    3. "server/discover": Decision("discover"). A method in LISTS: Decision("fan_out").
    4. method is tools/call, resources/read or prompts/get:
       - name missing: refuse, 400, -32020, "Mcp-Name missing"
       - no backend owns the name (fnmatch against Backend.owns): refuse, 400, -32602, "unknown name"
       - needed = the backend's baseline, then A.required_scopes(method, name), without duplicates;
         not all in scopes: refuse, 403, code None, reason "insufficient_scope", scope=needed
         (every scope the operation needs, in one challenge)
       - otherwise Decision("forward", backend=<its name>)
    5. anything else: refuse, 404, -32601, "method not found".
    """
    raise LabNotDone("Lecture 7.9: implement decide(). Hints: labs/s07/README.md, lecture 7.9.")


class BackendFailed(Exception):
    pass


def _audit_name(method: str | None, name: str | None, backends) -> str | None:
    """What the audit records as the name: resource URIs by the pattern that matched, never the URI itself."""
    if method == "resources/read" and name:
        return next((p for b in backends for p in b.owns if "*" in p and fnmatch(name, p)), "<unknown resource>")
    return name


def _decode(v: str | None) -> str | None:
    if v and v.startswith("=?base64?") and v.endswith("?="):
        return base64.b64decode(v[9:-2]).decode()
    return v


class Gateway:
    """The ASGI app. `apps` maps backend name -> an in-memory ASGI app (tests, labs) or None (real URL)."""

    def __init__(self, apps: dict[str, Any] | None = None, backends: tuple[Backend, ...] = BACKENDS):
        self.backends = {b.name: b for b in backends}
        self.apps = apps or {}
        self.verifier = A.JwtVerifier(audience=GATEWAY)
        self.audit: list[dict[str, Any]] = []
        self._http: dict[str, httpx2.AsyncClient] = {}

    def http(self, name: str) -> httpx2.AsyncClient:
        if name not in self._http:
            app = self.apps.get(name)
            self._http[name] = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app) if app else None, timeout=30)
        return self._http[name]

    async def _send(self, send, status: int, body: dict | bytes, headers: list[tuple[bytes, bytes]] | None = None,
                    ctype: str = "application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", ctype.encode()), *(headers or [])]})
        await send({"type": "http.response.body", "body": raw})

    async def _body(self, receive) -> bytes:
        chunks, more = [], True
        while more:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        return b"".join(chunks)

    def _forward_headers(self, h: dict[str, str], backend: Backend, sub: str, scopes: list[str],
                         client_id: str) -> dict[str, str]:
        keep = {k: v for k, v in h.items() if k in ("accept", "content-type", "mcp-protocol-version", "mcp-method",
                                                    "mcp-name") or k.startswith("mcp-param-")}
        token = A.mint(sub, scopes, aud=backend.url, client_id=client_id, act="skyops-gateway")   # exchange, not passthrough
        return {**keep, "authorization": f"Bearer {token}"}

    @property
    def router(self):                                   # the shape client.connect and uvicorn expect
        return self

    @asynccontextmanager
    async def lifespan_context(self, _app=None):
        """Start the in-memory backends with the gateway; close the forwarding clients on the way out."""
        async with AsyncExitStack() as stack:
            for app in self.apps.values():
                await stack.enter_async_context(app.router.lifespan_context(app))
            try:
                yield
            finally:
                for c in self._http.values():
                    await c.aclose()
                self._http.clear()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":                 # running under uvicorn
            msg = await receive()
            async with self.lifespan_context():
                await send({"type": "lifespan.startup.complete"})
                await receive()
            return await send({"type": "lifespan.shutdown.complete"})
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/mcp":
            return await self._send(send, 404, {"error": "not found"})
        h = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        body = await self._body(receive)
        bearer = h.get("authorization", "")
        tok = await self.verifier.verify_token(bearer[7:]) if bearer.lower().startswith("bearer ") else None
        method, name = h.get("mcp-method"), _decode(h.get("mcp-name"))
        logged = _audit_name(method, name, self.backends.values())
        if tok is None:
            self.audit.append({"sub": None, "method": method, "name": logged, "backend": None, "outcome": 401})
            return await self._send(send, 401, {"error": "invalid_token"}, [(b"www-authenticate",
                                    f'Bearer resource_metadata="{_METADATA}"'.encode())])
        d = decide(method, name, h.get("mcp-protocol-version"), tok.scopes, tuple(self.backends.values()))
        entry = {"sub": tok.subject, "method": method, "name": logged, "backend": d.backend, "outcome": None}
        self.audit.append(entry)
        if d.action == "refuse":
            entry["outcome"] = f"{d.status} {d.reason}"
            if d.status == 403:
                return await self._send(send, 403, {"error": "insufficient_scope", "scope": " ".join(d.scope)},
                                        [(b"www-authenticate", (f'Bearer error="insufficient_scope", scope="{" ".join(d.scope)}", '
                                          f'resource_metadata="{_METADATA}"').encode())])
            req_id = _json_id(body)
            return await self._send(send, d.status, {"jsonrpc": "2.0", "id": req_id,
                                                     "error": {"code": d.code, "message": d.reason}})
        if d.action == "discover":
            entry["outcome"] = "200 answered by gateway"
            return await self._send(send, 200, _discover(_json_id(body)))
        if d.action == "fan_out":
            entry["backend"] = ",".join(self.backends)
            try:
                merged, left_out = await self._fan_out(method, body, h, tok)
            except BackendFailed as e:
                entry["outcome"] = f"502 {e} failed"
                return await self._send(send, 502, {"jsonrpc": "2.0", "id": _json_id(body), "error": {
                    "code": -32603, "message": f"backend {e} unavailable"}})
            entry["outcome"] = "200 merged" + (f" ({', '.join(left_out)} refused, left out)" if left_out else "")
            return await self._send(send, 200, merged)
        b = self.backends[d.backend]
        r = await self.http(b.name).post(b.url, content=body,
                                         headers=self._forward_headers(h, b, tok.subject, tok.scopes, tok.client_id))
        entry["outcome"] = f"{r.status_code}"
        await self._send(send, r.status_code, r.content, ctype=r.headers.get("content-type", "application/json"))

    async def _fan_out(self, method: str, body: bytes, h: dict[str, str], tok) -> tuple[dict[str, Any], list[str]]:
        key, items, ttl, scopes, left_out = LISTS[method], [], None, set(), []
        for b in self.backends.values():
            try:
                r = await self.http(b.name).post(b.url, content=body,
                                                 headers=self._forward_headers(h, b, tok.subject, tok.scopes, tok.client_id))
            except httpx2.HTTPError:
                raise BackendFailed(b.name) from None
            if r.status_code in (401, 403):             # this caller may not see that server: leave it out, say so
                left_out.append(b.name)
                continue
            try:
                res = (r.json() or {}).get("result")
            except ValueError:
                res = None
            if r.status_code != 200 or not isinstance(res, dict):
                raise BackendFailed(b.name)
            items += res.get(key, [])
            ttl = res.get("ttlMs") if ttl is None else min(ttl, res.get("ttlMs", 0))
            scopes.add(res.get("cacheScope", "private"))
        return ({"jsonrpc": "2.0", "id": _json_id(body),
                 "result": {"resultType": "complete", key: sorted(items, key=lambda t: t.get("name") or t.get("uri", "")),
                            "ttlMs": ttl or 0, "cacheScope": "private" if "private" in scopes or left_out else "public"}},
                left_out)


def _json_id(body: bytes) -> Any:
    """The JSON-RPC id, for error replies. The one field of a request body the gateway reads."""
    try:
        return json.loads(body).get("id")
    except (ValueError, AttributeError):
        return None


def _discover(req_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id,
            "result": {"resultType": "complete", "supportedVersions": [CURRENT],
                       "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
                       "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "skyops-gateway", "version": "0.7.0"}},
                       "ttlMs": 3_600_000, "cacheScope": "public"}}
