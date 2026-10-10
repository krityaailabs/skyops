"""Lecture 6.6: header routing rules for a gateway in front of two MCP servers. Routing never reads a request body.

Two backends sit behind one gateway: flight-ops (three replicas, the server recorded in session.jsonl) and
crew-ops (one replica, Section 7 builds it). Every ROUTING decision below uses only HTTP headers:
MCP-Protocol-Version, Mcp-Method, Mcp-Name and Mcp-Param-*. That is what revision 2026-07-28 makes possible, and
it's only safe because the server behind the gateway checks that the headers match the body (HeaderMismatch).

Two jobs do read bodies, of RESPONSES: caching (ttlMs and cacheScope are fields in the result) and merging lists
from several backends. Requests are never parsed.

Legacy traffic (an initialize handshake, a session ID, an older MCP-Protocol-Version) isn't refused: it goes to a
legacy path that passes it through to the dual-era backends without header-based policy, and is counted, so you
can watch it fall during a migration (lecture 6.7).

    python -m labs.s06.gateway fixtures/mcp/session.jsonl --with-crew
    python -m labs.s06.gateway fixtures/mcp/legacy.jsonl

Read only in Section 6; Section 7.9 turns these rules into a running gateway with auth and audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fnmatch import fnmatch

from .wire import CURRENT, LEGACY, REMOVED, decode_header, load

LIST_METHODS = {"tools/list", "prompts/list", "resources/list", "resources/templates/list"}
NAMED_METHODS = {"tools/call", "resources/read", "prompts/get"}
OTHER_METHODS = {"completion/complete": "flight-ops"}       # no Mcp-Name in this revision: send to the owning backend


@dataclass(frozen=True)
class Backend:
    name: str
    replicas: int
    names: tuple[str, ...]          # Mcp-Name patterns this backend owns: tool names, prompt names, resource URIs


@dataclass(frozen=True)
class Policy:
    kind: str                       # read | write
    limit_per_min: int              # per caller
    audit: bool


BACKENDS = (
    Backend("flight-ops", 3, ("get_flight_status", "notify_passengers", "disruption_brief", "skyops://bookings/*")),
    Backend("crew-ops", 1, ("get_crew_roster", "skyops://crew/*")),
)
POLICY = {
    "get_flight_status": Policy("read", 600, False),
    "get_crew_roster": Policy("read", 120, True),        # crew data is personal: audit reads too
    "notify_passengers": Policy("write", 10, True),
}
DEFAULT_POLICY = Policy("read", 300, False)
SEEN = {"legacy": 0}                                     # legacy requests counted, for the migration dashboard


def rpc_error(status: int, code: int, message: str, data: object | None = None) -> str:
    """Refusals carry a JSON-RPC error body, so a dual-era client can tell a modern refusal from a legacy server."""
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return f"refuse ({status}) {json.dumps({'error': err})}"


def backend_for(name: str) -> Backend | None:
    return next((b for b in BACKENDS if any(fnmatch(name, p) for p in b.names)), None)


def replica(b: Backend, key: str) -> int:
    """Any replica can serve any request (no sessions). A stable key only adds cache locality."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % b.replicas + 1


def route(headers: dict[str, str]) -> str:
    """One routing decision from request headers alone."""
    h = {k.lower(): v for k, v in headers.items()}
    version, method = h.get("mcp-protocol-version"), h.get("mcp-method")
    name = decode_header(h["mcp-name"]) if "mcp-name" in h else None
    if "mcp-session-id" in h or version is None or version in LEGACY:
        SEEN["legacy"] += 1
        return "legacy path: pass through to dual-era backends (by session ID if set); no header policy; counted"
    if version != CURRENT:
        return rpc_error(400, -32022, "Unsupported protocol version", {"supported": [CURRENT], "requested": version})
    if method is None:
        return rpc_error(400, -32020, "Header mismatch: Mcp-Method is missing")
    if method in REMOVED:
        return rpc_error(404, -32601, "Method not found", method)
    if method == "server/discover":
        return "gateway answers itself: one discover built from the backends' answers"
    if method in LIST_METHODS:
        return f"fan out to {', '.join(b.name for b in BACKENDS)} and merge one list"
    if method == "subscriptions/listen":
        return "fan out a long-lived stream to every backend; merge their notifications"
    if method in OTHER_METHODS:
        return f"{OTHER_METHODS[method]} (no Mcp-Name for {method}: the owning backend)"
    if method not in NAMED_METHODS:
        return rpc_error(404, -32601, "Method not found", method)
    if name is None:
        return rpc_error(400, -32020, f"Header mismatch: Mcp-Name is required for {method}")
    b = backend_for(name)
    if b is None:
        return rpc_error(400, -32602, f"Unknown tool, prompt or resource: {name}")
    shard = decode_header(h["mcp-param-flight"]) if "mcp-param-flight" in h else None
    where = f"{b.name} replica {replica(b, shard)} of {b.replicas} (by Flight={shard})" if shard else (
        f"{b.name} (any of {b.replicas} replicas)" if b.replicas > 1 else b.name)
    p = POLICY.get(name, DEFAULT_POLICY)
    rules = f"{p.kind}, {p.limit_per_min}/min per caller" + (", audit" if p.audit else "")
    return f"{where} · {rules}"


def cache_rule(method: str, result: dict | None) -> str:
    """Reads the RESPONSE body: cache hints live in the result, not in headers."""
    if method == "server/discover" or not result or "cacheScope" not in result:
        return ""
    if result["cacheScope"] == "public":
        return f"   response: public, {result.get('ttlMs', 0) // 1000} s → shared cache for every caller"
    return f"   response: private, {result.get('ttlMs', 0) // 1000} s → cache per caller's token, never shared"


CREW_EXAMPLES = [
    ("crew roster read", {"MCP-Protocol-Version": CURRENT, "Mcp-Method": "tools/call", "Mcp-Name": "get_crew_roster"}),
    ("unknown tool", {"MCP-Protocol-Version": CURRENT, "Mcp-Method": "tools/call", "Mcp-Name": "export_contacts"}),
    ("encoded name", {"MCP-Protocol-Version": CURRENT, "Mcp-Method": "resources/read",
                      "Mcp-Name": "=?base64?c2t5b3BzOi8vY3Jldy9aWjIxNA==?="}),
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--with-crew", action="store_true", help="also route three hand-written requests")
    a = ap.parse_args(argv)
    for ex in load(a.path):
        if not ex.body:
            continue
        target = f" {ex.target}" if ex.target else ""
        print(f"#{ex.n:<2} {ex.method}{target}")
        print(f"     → {route(ex.req_headers)}")
        rule = cache_rule(ex.method, ex.result)
        if rule:
            print(rule)
    if a.with_crew:
        for label, h in CREW_EXAMPLES:
            print(f"[{label}] {h['Mcp-Method']} {decode_header(h['Mcp-Name'])}")
            print(f"     → {route(h)}")
    print(f"request bytes parsed for routing: 0 · legacy requests: {SEEN['legacy']}")


if __name__ == "__main__":
    main()
