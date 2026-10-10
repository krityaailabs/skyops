"""Section 7 fitness: an MCP server from the Section 5 contracts, a resource and a prompt, authorisation, a
client that feeds the Section 4 loop, contract versioning, and a gateway. Fail at s07-start, pass at s07.
Everything runs in memory with the official SDK: no network, no keys."""
import json

import anyio
import pytest
from mcp.client import Client

from labs.s07 import agent_over_mcp, first_server, gateway_demo
from skyops.mcp import auth as A
from skyops.mcp import versioning
from skyops.mcp.adapter import annotations_for, to_call_result
from skyops.mcp.client import map_call_result
from skyops.mcp.gateway import decide
from skyops.mcp.server import booking_resource, build_flight_server, disruption_brief, http_app
from skyops.tools.catalog import make_catalog
from skyops.tools.contracts import SideEffect
from mcp_types import CallToolResult, TextContent

D = "2026-10-09"
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}


def _call(client_fn):
    return anyio.run(client_fn)


def test_l71_first_server_one_tool():
    srv = first_server.build()

    async def go():
        async with Client(srv) as c:
            tools = (await c.list_tools()).tools
            ok = await c.call_tool("get_flight_status", {"flight_number": "zz214", "date": D})
            bad = await c.call_tool("get_flight_status", {"flight_number": "ZZ999", "date": D})
            return tools, ok, bad
    tools, ok, bad = _call(go)
    assert [t.name for t in tools] == ["get_flight_status"]
    assert tools[0].input_schema["properties"]["flight_number"]["description"].startswith("Flight number")
    assert ok.structured_content["gate"] == "B7"
    assert bad.is_error and "No flight ZZ999" in bad.content[0].text      # ToolError: the model reads why


def test_l72_contracts_become_mcp_tools():
    contracts, _ = make_catalog()
    by = {c.name: c for c in contracts}
    assert annotations_for(by["get_flight_status"]).read_only_hint is True
    notify = annotations_for(by["notify_passengers"])
    assert notify.destructive_hint is True and notify.idempotent_hint is True and notify.open_world_hint is True
    err = to_call_result({"error": "not_found", "category": "non_retryable", "message": "No flight", "retryable": False})
    assert err.is_error and err.structured_content["category"] == "non_retryable" and err.content[0].text == "not_found: No flight"
    ok = to_call_result({"gate": "B7", "status": "DELAYED"})
    assert not ok.is_error and json.loads(ok.content[0].text) == {"gate": "B7", "status": "DELAYED"}

    async def go():
        async with Client(build_flight_server()) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            r1 = await c.call_tool("notify_passengers", {"flight_number": "ZZ247", "message": "ZZ247 tonight is cancelled.",
                                                         "reason": "cancellation"})
            r2 = await c.call_tool("notify_passengers", {"flight_number": "zz 247", "message": "ZZ247 tonight is cancelled.",
                                                         "reason": "cancellation"})
            return tools, r1, r2
    tools, r1, r2 = _call(go)
    assert set(tools) == {"list_departures", "get_flight_status", "notify_passengers", "issue_voucher"}
    assert tools["list_departures"].input_schema["properties"]["destination"]["description"].startswith("Optional IATA")
    assert r1.structured_content["replayed"] is False and r2.structured_content["replayed"] is True


def test_l73_resource_and_prompt():
    assert json.loads(booking_resource("Q7K2LM"))["flight_number"] == "ZZ247"
    text = disruption_brief("ZZ173")
    low = text.lower()
    assert "ZZ173" in text and "get_flight_status" in text and "get_crew_status" in text and "three lines" in low
    assert "not message passengers" in low

    async def go():
        async with Client(build_flight_server()) as c:
            res = await c.read_resource("skyops://bookings/Q7K2LM")
            prompt = await c.get_prompt("disruption_brief", {"flight_number": "ZZ214"})
            return res, prompt
    res, prompt = _call(go)
    assert res.cache_scope == "private" and "Q7K2LM" in res.contents[0].text
    assert "ZZ214" in prompt.messages[0].content.text


def _post(app, headers, body):
    import httpx2

    async def go():
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="https://mcp.skyops.example") as c:
                return await c.post("/mcp", headers=headers, json=body)
    return anyio.run(go)


def _h(method, name=None, token=None):
    h = {"accept": "application/json, text/event-stream", "content-type": "application/json",
         "mcp-protocol-version": "2026-07-28", "mcp-method": method}
    if name:
        h["mcp-name"] = name
    if token:
        h["authorization"] = f"Bearer {token}"
    return h


def test_l74_wrong_audience_is_rejected():
    app = http_app(build_flight_server(auth=True), auth=True)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": META}}
    r = _post(app, _h("tools/list", token=A.mint("dm", ["flights:read"], aud="https://gateway.skyops.example/mcp")), body)
    assert r.status_code == 401 and "resource_metadata=" in r.headers["www-authenticate"]


def test_l74_insufficient_scope_403():
    app = http_app(build_flight_server(auth=True), auth=True)
    body = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "notify_passengers", "arguments": {"flight_number": "ZZ247", "message": "ZZ247 is cancelled.",
                                                   "reason": "cancellation"}, "_meta": META}}
    r = _post(app, _h("tools/call", "notify_passengers", A.mint("dm", ["flights:read"])), body)
    assert r.status_code == 403
    assert 'error="insufficient_scope"' in r.headers["www-authenticate"] and 'scope="flights:read passengers:notify"' in r.headers["www-authenticate"]


def test_l74_client_checks_the_issuer():
    assert A.check_authorization_response({"code": "c1", "iss": A.ISSUER}, A.ISSUER, True) == "c1"
    with pytest.raises(PermissionError):
        A.check_authorization_response({"code": "c1"}, A.ISSUER, True)
    with pytest.raises(PermissionError):
        A.check_authorization_response({"code": "c1", "iss": A.ISSUER + "/"}, A.ISSUER, False)
    assert A.check_authorization_response({"code": "c1"}, A.ISSUER, False) == "c1"


def test_l75_client_maps_results_and_feeds_the_loop():
    taxonomy = {"error": "not_found", "category": "non_retryable", "message": "x", "retryable": False}
    assert map_call_result(CallToolResult(content=[TextContent(type="text", text="x")], structured_content=taxonomy,
                                          is_error=True)) == taxonomy
    prose = map_call_result(CallToolResult(content=[TextContent(type="text", text="boom"), TextContent(type="text", text="!")],
                                           is_error=True))
    assert prose == {"error": "tool_error", "category": "non_retryable", "message": "boom !", "retryable": False}
    assert map_call_result(CallToolResult(content=[TextContent(type="text", text="a"), TextContent(type="text", text="b")])) == {"text": "a\nb"}
    r, host = agent_over_mcp.run()
    assert r.stop == "answered" and [c["server"] for c in host.calls] == ["flight-ops", "flight-ops", "crew-ops"]


def test_l77_lock_catches_silent_changes():
    listing, version = versioning.current_listing()
    lock = json.loads(versioning.LOCK.read_text())
    assert versioning.check_lock(listing, version, lock) == []
    changed = json.loads(json.dumps(listing))
    changed["get_flight_status"]["description"] += " Now with more."
    assert versioning.check_lock(changed, version, lock) == ["tools changed without a version bump"]
    assert versioning.check_lock(changed, "0.8.0", lock) == []
    breaking = json.loads(json.dumps(listing))
    breaking["get_flight_status"]["inputSchema"]["required"].append("airline")
    breaking["get_flight_status"]["inputSchema"]["properties"]["airline"] = {"type": "string"}
    assert versioning.check_lock(breaking, "0.8.0", lock) == ["breaking change to get_flight_status"]
    assert versioning.check_lock(breaking, "1.0.0", lock) == []
    gone = {k: v for k, v in listing.items() if k != "issue_voucher"}
    assert versioning.check_lock(gone, "0.8.0", lock) == ["removed tool issue_voucher"]
    assert versioning.check_lock(listing, "0.7.1", lock) == ["version changed but the lock is stale: run make mcp-lock"]


def test_l79_gateway_decides_from_headers():
    s = ["flights:read", "crew:read"]
    assert decide("tools/call", "get_crew_status", "2026-07-28", s).backend == "crew-ops"
    assert decide("tools/call", "skyops://bookings/Q7K2LM", "2026-07-28", s).action == "refuse"   # bookings:read missing
    assert decide("resources/read", "skyops://bookings/Q7K2LM", "2026-07-28", s + ["bookings:read"]).backend == "flight-ops"
    no = decide("tools/call", "notify_passengers", "2026-07-28", s)
    assert (no.status, no.reason, no.scope) == (403, "insufficient_scope", ["flights:read", "passengers:notify"])
    assert decide("tools/call", "export_contacts", "2026-07-28", s).code == -32602
    assert decide("tools/list", None, "2026-07-28", s).action == "fan_out"
    assert decide("server/discover", None, "2026-07-28", s).action == "discover"
    assert decide("tools/list", None, "2025-11-25", s).code == -32022
    gw, out = gateway_demo.run(["flights:read", "crew:read"])
    assert [a["outcome"] for a in gw.audit][-3:] == ["200", "200", "403 insufficient_scope"]
    assert out[2][1]["error"] == "not_authorized"


def test_l710_unknown_host_rejected():
    app = http_app(build_flight_server())
    import httpx2

    async def go():
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="https://evil.example") as c:
                return await c.post("/mcp", headers=_h("tools/list"), json={"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                                                                            "params": {"_meta": META}})
    assert anyio.run(go).status_code == 421


def test_side_effect_classes_cover_reversible_writes():
    from skyops.tools.contracts import ToolContract
    from pydantic import BaseModel

    class A_(BaseModel):
        x: int
    c = ToolContract("move_booking", "Move a booking.", A_, lambda a: {}, SideEffect.REVERSIBLE_WRITE)
    a = annotations_for(c)
    assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint, a.open_world_hint) == (False, False, False, False)
