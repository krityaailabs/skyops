"""Record REAL MCP wire traffic for the Section 6 inspector (instructor; learners may re-run it).

Everything here runs in-process, with no network and no keys: the official SDK's server (ops_server.py) is
mounted as an ASGI app, and the SDK's own client talks to it over Streamable HTTP through httpx. Every HTTP
request and response is written to fixtures/mcp/*.jsonl, exactly as sent, minus transport noise headers.

    session.jsonl      the SDK client: discover, a header-mismatch recovery, tools/list, a tool call,
                       a Multi Round-Trip Request and its retry, a private resource, a prompt
    errors.jsonl       hand-built requests (httpx), each with one deliberate fault, and the server's real answer
    legacy.jsonl       hand-built 2025-11-25 requests (handshake, session ID) and the same server's real answers
    third_party.jsonl  discover and tools/list from flight-tracker-plus (lecture 6.9)

Run: python -m labs.s06.record      (re-recording changes the sealed requestState values, nothing else)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import httpx2
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import ElicitResult, Implementation

from . import ops_server, third_party

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fixtures" / "mcp"
BASE = "http://127.0.0.1:8000"
RECORDED = "2026-10-10"
KEEP_REQ = {"accept", "content-type", "mcp-protocol-version", "mcp-method", "mcp-name", "mcp-session-id",
            "last-event-id"}
KEEP_RESP = {"content-type", "mcp-session-id"}
V = "2026-07-28"


def _sdk() -> str:
    from importlib.metadata import version
    return f"mcp {version('mcp')}"


def _body(text: str, ctype: str) -> Any:
    if not text:
        return None
    if ctype.startswith("text/event-stream"):
        data = [l[5:].strip() for l in text.splitlines() if l.startswith("data:")]
        return json.loads(data[-1]) if data else None
    return json.loads(text)


class Recorder:
    def __init__(self, client: str = "mcp SDK client") -> None:
        self.client = client          # who built the requests: the SDK's own client, or hand-built with httpx
        self.records: list[dict] = []
        self.pending: dict[int, dict] = {}

    async def on_request(self, r: httpx2.Request) -> None:
        await r.aread()
        hdrs = {k: v for k, v in r.headers.items() if k in KEEP_REQ or k.startswith("mcp-param-")}
        self.pending[id(r)] = {"method": r.method, "path": r.url.path, "headers": hdrs,
                               "body": json.loads(r.content) if r.content else None}

    async def on_response(self, r: httpx2.Response) -> None:
        await r.aread()
        ctype = r.headers.get("content-type", "")
        self.records.append({
            "n": len(self.records) + 1, "label": "",
            "request": self.pending.pop(id(r.request)),
            "response": {"status": r.status_code,
                         "headers": {k: v for k, v in r.headers.items() if k in KEEP_RESP},
                         "body": _body(r.text, ctype)},
            "meta": {"synthetic": False, "recorded": RECORDED, "sdk": _sdk(), "server": "", "note": "",
                     "client": self.client}})

    def http(self, app) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE,
                                  event_hooks={"request": [self.on_request], "response": [self.on_response]})

    def label(self, label: str, note: str = "", server: str = ops_server.NAME) -> None:
        """Label every record added since the last call."""
        for rec in self.records:
            if not rec["label"]:
                rec["label"], rec["meta"]["note"], rec["meta"]["server"] = label, note, server


def _meta(version: str = V, caps: dict | None = None, drop: str | None = None) -> dict:
    m = {"io.modelcontextprotocol/protocolVersion": version,
         "io.modelcontextprotocol/clientInfo": {"name": "skyops-host", "version": "0.6.0"},
         "io.modelcontextprotocol/clientCapabilities": caps if caps is not None else {}}
    if drop:
        m.pop(drop)
    return m


def _headers(method: str, name: str | None = None, version: str = V, **params: str) -> dict:
    h = {"accept": "application/json, text/event-stream", "content-type": "application/json",
         "mcp-protocol-version": version, "mcp-method": method}
    if name:
        h["mcp-name"] = name
    h.update({f"mcp-param-{k}": v for k, v in params.items()})
    return h


async def _confirm(_ctx, _params) -> ElicitResult:
    return ElicitResult(action="accept", content={"confirm": True})


MSG = "ZZ247 tonight is cancelled. We will text your rebooking options within the hour."


async def record_session(rec: Recorder) -> None:
    srv = ops_server.build()
    app = srv.streamable_http_app()
    async with app.router.lifespan_context(app):
        transport = streamable_http_client(f"{BASE}/mcp", http_client=rec.http(app))
        async with Client(transport, elicitation_callback=_confirm,
                          client_info=Implementation(name="skyops-host", version="0.6.0")) as c:
            rec.label("discover", "the SDK client probes the server once: versions, capabilities, identity")
            await c.call_tool("get_flight_status", {"flight_number": "ZZ214", "date": "2026-10-09"})
            rec.label("call before list", "first call without the tool's schema: the client didn't know to mirror "
                      "flight_number, the server rejects, the client lists tools and retries")
            await c.call_tool("notify_passengers", {"flight_number": "ZZ247", "message": MSG})
            rec.label("multi round-trip", "the server needs a person's confirmation: input_required, then a retry "
                      "with a new id, the answer and the echoed requestState")
            await c.read_resource("skyops://bookings/Q7K2LM")
            rec.label("private resource", "a booking: cacheScope private, so no shared cache may keep it")
            await c.list_prompts()
            await c.get_prompt("disruption_brief", {"flight_number": "ZZ247"})
            rec.label("prompts", "prompts are user-picked templates; the host shows them in a menu")


async def record_errors(rec: Recorder) -> None:
    srv = ops_server.build()
    app = srv.streamable_http_app()
    caps = {"elicitation": {"form": {}}}
    async with app.router.lifespan_context(app):
        c = rec.http(app)

        async def post(headers: dict, body: dict, label: str, note: str) -> httpx2.Response:
            r = await c.post("/mcp", headers=headers, json=body)
            rec.label(label, note)
            return r

        tl = lambda i, v=V: {"jsonrpc": "2.0", "id": i, "method": "tools/list", "params": {"_meta": _meta(v)}}
        await post(_headers("tools/list", version="2099-01-01"), tl(1, "2099-01-01"), "unsupported version",
                   "FAULT: asks for a revision the server doesn't implement (2099-01-01 is deliberately fictional)")
        await post(_headers("tools/list"), tl(2, "2025-11-25"), "version header disagrees",
                   "FAULT: MCP-Protocol-Version header says 2026-07-28, the body's _meta says 2025-11-25")
        await post(_headers("tools/list"), {"jsonrpc": "2.0", "id": 3, "method": "tools/list",
                                            "params": {"_meta": _meta(drop="io.modelcontextprotocol/clientCapabilities")}},
                   "missing capabilities", "FAULT: _meta has no clientCapabilities, which every request must carry")
        call = lambda i, name, args, caps_=None, **extra: {
            "jsonrpc": "2.0", "id": i, "method": "tools/call",
            "params": {"name": name, "arguments": args, **extra, "_meta": _meta(caps=caps_)}}
        await post(_headers("tools/call", "notify_passengers"), call(4, "notify_passengers",
                   {"flight_number": "ZZ247", "message": MSG}), "capability not declared",
                   "FAULT: the tool needs to ask a person, but this client declared no elicitation capability")
        await post(_headers("tools/call", "get_flight_status", Flight="ZZ214"),
                   call(5, "notify_passengers", {"flight_number": "ZZ214", "message": MSG}, caps),
                   "name header disagrees",
                   "FAULT: Mcp-Name says get_flight_status; the body calls notify_passengers. A gateway routing on "
                   "the header would have applied the wrong policy")
        await post(_headers("tools/call", "get_flight_status", Flight="ZZ173"),
                   call(6, "get_flight_status", {"flight_number": "ZZ214", "date": "2026-10-09"}, caps),
                   "param header disagrees", "FAULT: Mcp-Param-Flight says ZZ173; the arguments say ZZ214")
        await post(_headers("ping"), {"jsonrpc": "2.0", "id": 7, "method": "ping", "params": {"_meta": _meta()}},
                   "removed method", "FAULT: ping was removed in 2026-07-28; a leftover from an older client")
        r = await post(_headers("tools/call", "notify_passengers"),
                       call(8, "notify_passengers", {"flight_number": "ZZ247", "message": MSG}, caps),
                       "first round", "correct: the server asks for confirmation and seals its state")
        res = r.json()["result"]
        key, state = next(iter(res["inputRequests"])), res["requestState"]
        answer = {key: {"action": "accept", "content": {"confirm": True}}}
        flipped = state[:-6] + ("A" if state[-6] != "A" else "B") + state[-5:]
        await post(_headers("tools/call", "notify_passengers"),
                   call(9, "notify_passengers", {"flight_number": "ZZ247", "message": MSG}, caps,
                        inputResponses=answer, requestState=flipped),
                   "tampered state", "FAULT: one character of requestState changed; the seal no longer verifies")
        await post(_headers("tools/call", "notify_passengers"),
                   call(10, "notify_passengers", {"flight_number": "ZZ101", "message": MSG}, caps,
                        inputResponses=answer, requestState=state),
                   "state on another request",
                   "FAULT: a genuine requestState, replayed on a call for a different flight")
        retry = call(11, "notify_passengers", {"flight_number": "ZZ247", "message": MSG}, caps,
                     inputResponses=answer, requestState=state)
        await post(_headers("tools/call", "notify_passengers"), retry, "correct retry",
                   "correct: new id, the answer, the state echoed exactly; the message is sent")
        await post(_headers("tools/call", "notify_passengers"), {**retry, "id": 12}, "replayed retry",
                   "the same retry sent again: the seal still verifies (state is not single-use), so the tool's own "
                   "idempotency key is what stops a second send")
        await c.aclose()


async def record_legacy(rec: Recorder) -> None:
    srv = ops_server.build()
    app = srv.streamable_http_app()
    old = "2025-11-25"
    async with app.router.lifespan_context(app):
        c = rec.http(app)
        base = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
        r = await c.post("/mcp", headers=base, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": old, "capabilities": {}, "clientInfo": {"name": "old-host", "version": "1.9"}}})
        rec.label("initialize", "the 2025-11-25 handshake: the server mints a session and answers on an SSE stream")
        sid = r.headers["mcp-session-id"]
        h = {**base, "mcp-session-id": sid, "mcp-protocol-version": old}
        await c.post("/mcp", headers=h, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        rec.label("initialized", "the handshake's second half, a notification: 202 Accepted, no body")
        await c.post("/mcp", headers=h, json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "get_flight_status", "arguments": {"flight_number": "ZZ214", "date": "2026-10-09"}}})
        rec.label("legacy call", "no _meta, no Mcp-Method or Mcp-Name: the session ID carries the context")
        await c.post("/mcp", headers=h, json={"jsonrpc": "2.0", "id": 3, "method": "ping"})
        rec.label("legacy ping", "ping still works for a legacy client on this dual-era server")
        await c.aclose()


async def record_third_party(rec: Recorder) -> None:
    srv = third_party.build()
    app = srv.streamable_http_app()
    async with app.router.lifespan_context(app):
        transport = streamable_http_client(f"{BASE}/mcp", http_client=rec.http(app))
        async with Client(transport, client_info=Implementation(name="skyops-review", version="0.6.0")) as c:
            await c.list_tools()
            rec.label("third-party listing", "what flight-tracker-plus tells a client about itself", third_party.NAME)


def write(name: str, records: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return p


def main() -> None:
    hand = "hand-built with httpx (server answers are real)"
    for name, fn, client in (("session.jsonl", record_session, "mcp SDK client"), ("errors.jsonl", record_errors, hand),
                             ("legacy.jsonl", record_legacy, hand), ("third_party.jsonl", record_third_party, "mcp SDK client")):
        rec = Recorder(client)
        anyio.run(fn, rec)
        print(f"{write(name, rec.records).relative_to(ROOT)}: {len(rec.records)} exchanges")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    main()
