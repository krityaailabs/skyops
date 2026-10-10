"""The SkyOps MCP servers (Section 7). Official Python SDK v2 (`mcp` 2.3.0), protocol revision 2026-07-28.

    flight-ops   the Section 5 contracts as tools (7.2), one resource and one prompt (7.3)
    crew-ops     the legacy crew service behind a facade (7.6)

Both can run with authorisation (7.4): the SDK checks the bearer token on every request (401 if missing or
not valid for this server); ScopeGuard checks the operation's scopes (403 insufficient_scope).
`python -m skyops.mcp.serve` runs flight-ops over HTTPS (7.10).
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any

from mcp.server import MCPServer
from mcp.server.auth.settings import AuthSettings
from mcp.server.caching import CacheHint
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.server.request_state import RequestStateSecurity
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS, MODERN_PROTOCOL_VERSIONS

from ..sim import bookings
from ..tools.catalog import Gateways, make_catalog
from ..tools.contracts import IdempotencyStore
from . import auth as A
from .adapter import register
from .facade import crew_contract
from skyops.model.errors import LabNotDone

VERSION = "0.7.0"
HINTS = {"server/discover": CacheHint(3_600_000, "public"), "tools/list": CacheHint(300_000, "public"),
         "prompts/list": CacheHint(300_000, "public"), "resources/templates/list": CacheHint(300_000, "public"),
         "resources/list": CacheHint(300_000, "public"), "resources/read": CacheHint(60_000, "private")}


def state_security() -> RequestStateSecurity:
    """One sealing key for every replica (lecture 6.4), from the environment; a per-process key otherwise."""
    key = os.environ.get("SKYOPS_STATE_KEY")
    return RequestStateSecurity(keys=[key.encode()], ttl=300) if key else RequestStateSecurity.ephemeral(ttl=300)


def _auth(on: bool, resource: str = A.RESOURCE, baseline: list[str] = A.BASELINE,
          verifier: A.JwtVerifier | None = None) -> dict[str, Any]:
    if not on:
        return {}
    return {"token_verifier": verifier or A.JwtVerifier(audience=resource),
            "auth": AuthSettings(issuer_url=A.ISSUER, resource_server_url=resource, required_scopes=baseline,
                                 validate_token_resource=True)}


def booking_resource(booking_ref: str) -> str:
    """Lecture 7.3: one booking, as JSON text, for the host to attach as context.

    Look it up with bookings.get(booking_ref). Unknown reference: raise ResourceNotFoundError(f"No booking
    {booking_ref}") (from mcp.server.mcpserver.exceptions); the SDK answers it with -32602, resource not found.
    Any other exception is a crash: the client gets -32603. Return json.dumps(booking, sort_keys=True).
    Who may read it is decided before this runs: the bookings:read scope (ScopeGuard), and cacheScope
    "private", so no shared cache serves one caller's booking to another.
    """
    raise LabNotDone("Lecture 7.3: implement booking_resource(). Hints: labs/s07/README.md, lecture 7.3.")


def disruption_brief(flight_number: str) -> str:
    """Lecture 7.3: a prompt template a duty manager picks from a menu. Returns the instruction text.

    It must: name the flight; tell the model to look it up with get_flight_status first, and to check crew
    with get_crew_status when the flight is delayed; ask for exactly three lines (status, cause, the next
    decision needed); and say not to message passengers (a brief informs a person; it doesn't act).
    The check looks for these exact phrases: the flight number, "get_flight_status", "get_crew_status",
    "three lines" and "not message passengers" (case doesn't matter).
    """
    raise LabNotDone("Lecture 7.3: implement disruption_brief(). Hints: labs/s07/README.md, lecture 7.3.")


def build_flight_server(*, auth: bool = False, gw: Gateways | None = None,
                        store: IdempotencyStore | None = None, verifier: A.JwtVerifier | None = None) -> MCPServer:
    srv = MCPServer("skyops-flight-ops", version=VERSION, cache_hints=HINTS,
                    instructions="SkyOps flight operations at the hub. Look flights up before telling anyone anything.",
                    request_state_security=state_security(), **_auth(auth, verifier=verifier))
    contracts, _ = make_catalog(gw)
    register(srv, contracts, store)
    srv.resource("skyops://bookings/{booking_ref}", name="booking", mime_type="application/json",
                 description="One booking: flight, cabin, party size, contact preference, status. Private.")(booking_resource)
    srv.prompt(name="disruption_brief",
               description="Draft a three-line disruption brief for one flight, for the duty manager.")(disruption_brief)
    return srv


def build_crew_server(*, auth: bool = False) -> MCPServer:
    srv = MCPServer("skyops-crew-ops", version=VERSION, cache_hints=HINTS, request_state_security=state_security(),
                    **_auth(auth, A.CREW_RESOURCE, ["crew:read"]))
    register(srv, [crew_contract()])
    return srv


async def _refuse_version(send, pv: str | None) -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": None, "error": {
        "code": -32022, "message": "Unsupported protocol version",
        "data": {"supported": list(MODERN_PROTOCOL_VERSIONS), "requested": pv}}}).encode()
    await send({"type": "http.response.start", "status": 400, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


class ScopeGuard:
    """ASGI middleware: per-operation scopes, from the request headers (Mcp-Method, Mcp-Name) and the token.

    A valid token without the scopes the operation needs gets 403 and a WWW-Authenticate challenge naming
    ALL the scopes that operation needs (spec: one challenge, not one scope at a time). Missing or invalid
    tokens pass through, so the SDK answers them with its 401 and the resource-metadata pointer.
    Trusting Mcp-Name here is safe only because the SDK then rejects any current-revision request whose
    headers and body disagree (HeaderMismatch). Older revisions get no such check, so the guard refuses
    them outright (400, -32022): a request with no MCP-Protocol-Version header, or a pre-2026-07-28 one,
    never reaches the server, where it could otherwise pass this check as a read and run as a write.
    The scopes it names are the server's baseline plus the operation's own, so one challenge is enough.
    """

    def __init__(self, app, verifier: A.JwtVerifier, metadata_url: str, baseline: list[str] = A.BASELINE):
        self.app, self.verifier, self.metadata_url, self.baseline = app, verifier, metadata_url, baseline
        self.denied: list[dict[str, Any]] = []

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        h = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        pv = h.get("mcp-protocol-version")
        if pv is None or pv in HANDSHAKE_PROTOCOL_VERSIONS:
            return await _refuse_version(send, pv)
        bearer = h.get("authorization", "")
        tok = await self.verifier.verify_token(bearer[7:]) if bearer.lower().startswith("bearer ") else None
        if tok is not None:
            name = h.get("mcp-name")
            if name and name.startswith("=?base64?") and name.endswith("?="):
                name = base64.b64decode(name[9:-2]).decode()
            need = list(dict.fromkeys([*self.baseline, *A.required_scopes(h.get("mcp-method", ""), name, self.baseline)]))
            if not set(need) <= set(tok.scopes):
                self.denied.append({"sub": tok.subject, "method": h.get("mcp-method"), "name": name, "need": need})
                challenge = (f'Bearer error="insufficient_scope", scope="{" ".join(need)}", '
                             f'resource_metadata="{self.metadata_url}"')
                body = json.dumps({"error": "insufficient_scope", "scope": " ".join(need)}).encode()
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", challenge.encode())]})
                return await send({"type": "http.response.body", "body": body})
        return await self.app(scope, receive, send)


ALLOWED_HOSTS = ["mcp.skyops.example", "crew.skyops.example", "127.0.0.1:*", "localhost:*"]


def http_app(srv: MCPServer, *, auth: bool = False, hosts: list[str] | None = None, resource: str = A.RESOURCE,
             baseline: list[str] | None = None, verifier: A.JwtVerifier | None = None):
    """The Streamable HTTP app for a server, with the scope guard in front when auth is on.
    Host checking (DNS-rebinding protection, lecture 6.6) stays on: only the names we serve are accepted.
    No browser origins are allowed: SkyOps hosts are services, not web pages."""
    sec = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts or ALLOWED_HOSTS,
                                    allowed_origins=[])
    app = srv.streamable_http_app(transport_security=sec)
    if not auth:
        return app
    baseline = baseline or (["crew:read"] if resource == A.CREW_RESOURCE else A.BASELINE)
    guard = ScopeGuard(app, verifier or A.JwtVerifier(audience=resource),
                       resource.rsplit("/mcp", 1)[0] + "/.well-known/oauth-protected-resource/mcp", baseline)
    guard.router = app.router            # keep lifespan reachable for in-process tests
    return guard
