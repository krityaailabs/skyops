"""Section 6 checks (read-only section: these pass from tag s05). Run: make labs-s06"""
from __future__ import annotations

import json
from pathlib import Path

import anyio

from labs.s06 import gateway, primitives, record, review_server, wire

ROOT = Path(__file__).resolve().parents[2]
MCP = ROOT / "fixtures" / "mcp"


def _ex(name: str):
    return wire.load(MCP / name)


def test_recordings_say_who_built_the_requests():
    for f in MCP.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            meta = json.loads(line)["meta"]
            assert meta["synthetic"] is False and meta["sdk"].startswith("mcp 2.")
            want = "mcp SDK client" if f.name in ("session.jsonl", "third_party.jsonl") else "hand-built"
            assert meta["client"].startswith(want)


def test_every_modern_request_carries_meta_and_headers():
    for ex in _ex("session.jsonl"):
        h = {k.lower(): v for k, v in ex.req_headers.items()}
        assert h["mcp-protocol-version"] == ex.meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
        assert h["mcp-method"] == ex.method
        assert "io.modelcontextprotocol/clientCapabilities" in ex.meta
        if ex.method in wire.NAMED:
            assert h["mcp-name"] == ex.target


def test_cacheable_results_carry_hints_and_bookings_are_private():
    for ex in _ex("session.jsonl"):
        if ex.result and ex.result.get("resultType") == "complete" and ex.method in wire.CACHEABLE:
            assert {"ttlMs", "cacheScope"} <= set(ex.result)
            assert ex.result["cacheScope"] == ("private" if ex.method == "resources/read" else "public")


def test_multi_round_trip_retry_is_a_new_request_with_the_state_echoed():
    exs = {e.n: e for e in _ex("session.jsonl")}
    first = next(e for e in exs.values() if e.result and e.result.get("resultType") == "input_required")
    retry = exs[first.n + 1]
    assert retry.body["id"] != first.body["id"]
    assert retry.params["requestState"] == first.result["requestState"]
    assert set(retry.params["inputResponses"]) == set(first.result["inputRequests"])
    assert retry.result["resultType"] == "complete"


def test_session_has_one_rejection_and_the_inspector_sees_why():
    found = wire.check(_ex("session.jsonl"), wire.known_schemas(_ex("session.jsonl")))
    assert [(f.n, f.code) for f in found if f.side in ("client", "history")] == [(2, "missing_param_header")]


def test_inspector_finds_every_fault_a_request_shows():
    exs = _ex("errors.jsonl")
    found = wire.check(exs, wire.known_schemas(exs, MCP))
    rejected = {e.n for e in exs if e.error}
    alone = {f.n for f in found if f.side == "client"}
    history = {f.n for f in found if f.side == "history"}
    assert alone <= rejected                                  # nothing correct is flagged as a fault
    assert rejected - alone == {4, 9, 10}                     # these need more than the request
    assert {9, 10} <= history                                 # ...the earlier exchange shows the state faults
    assert wire.summary(exs, found)["server_only"] == 1       # ...and only the server knows the tool will ask


def test_a_replayed_retry_is_accepted_but_not_sent_twice():
    exs = {e.n: e for e in _ex("errors.jsonl")}
    first, again = exs[11], exs[12]
    assert first.params["requestState"] == again.params["requestState"] and first.body["id"] != again.body["id"]
    assert json.loads(again.result["content"][0]["text"])["replayed"] is True
    found = wire.check(list(exs.values()), wire.known_schemas(list(exs.values()), MCP))
    assert any(f.n == 12 and f.code == "state_replayed" for f in found)


def test_legacy_traffic_is_recognised():
    codes = {f.code for f in wire.check(_ex("legacy.jsonl"))}
    assert {"legacy_session", "removed_method", "no_result_type"} <= codes


def test_base64_sentinel_decodes():
    assert wire.decode_header("=?base64?SGVsbG8sIOS4lueVjA==?=") == "Hello, 世界"
    assert wire.decode_header("us-west1") == "us-west1"


def test_gateway_routes_on_headers_only():
    assert gateway.route({"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call",
                          "Mcp-Name": "get_crew_roster"}).startswith("crew-ops")
    assert "-32602" in gateway.route({"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call",
                                      "Mcp-Name": "export_contacts"})
    assert gateway.route({"MCP-Protocol-Version": "2025-11-25", "Mcp-Method": "tools/call",
                          "Mcp-Name": "get_flight_status"}).startswith("legacy path")
    a = gateway.route({"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call",
                       "Mcp-Name": "get_flight_status", "Mcp-Param-Flight": "ZZ214"})
    assert a == gateway.route({"mcp-protocol-version": "2026-07-28", "mcp-method": "tools/call",
                               "mcp-name": "get_flight_status", "mcp-param-flight": "ZZ214"})


def test_review_blocks_the_third_party_server_and_pins_the_listing():
    exs = _ex("third_party.jsonl")
    tools = next(e for e in exs if e.method == "tools/list").result["tools"]
    info = next(e for e in exs if e.method == "server/discover").result["_meta"]["io.modelcontextprotocol/serverInfo"]
    manifest = json.loads((ROOT / "labs/s06/third_party_manifest.json").read_text())
    policy = json.loads((ROOT / "labs/s06/mcp_policy.json").read_text())
    found = review_server.review(tools, info, manifest, policy)
    assert any(f.area == "descriptions" and f.detail.startswith("track_flight") and f.severity == "block" for f in found)
    assert review_server.allow_list(tools, found) == ["get_airport_weather"]
    changed = [dict(t, description=t["description"] + " ") if t["name"] == "get_airport_weather" else t for t in tools]
    assert review_server.listing_digest(changed) != review_server.listing_digest(tools)
    flipped = [dict(t, annotations={"readOnlyHint": False}) if t["name"] == "sync_bookings" else t for t in tools]
    assert review_server.listing_digest(flipped) != review_server.listing_digest(tools)     # annotations count too
    assert len(review_server.listing_digest(tools)) == len("sha256:") + 64


def test_connector_arithmetic():
    assert primitives.connectors(4, 6) == (24, 10)


def _wire_shape(r: dict) -> tuple:
    """Everything on the wire that a client, gateway or server depends on, minus the sealed state's bytes."""
    body, resp = r["request"]["body"], r["response"]["body"] or {}
    res, err = resp.get("result") or {}, resp.get("error") or {}
    params = body.get("params") or {}
    return (body["method"], sorted(r["request"]["headers"].items()), sorted((params.get("_meta") or {}).items(), key=str),
            sorted(k for k in params if k != "_meta"), r["response"]["status"], res.get("resultType"),
            res.get("ttlMs"), res.get("cacheScope"), err.get("code"), sorted(res), sorted(res.get("_meta") or {}))


def test_the_sdk_still_speaks_what_was_recorded():
    """Re-record the session in memory and compare the wire: headers, _meta, result types, cache hints, errors."""
    rec = record.Recorder()
    anyio.run(record.record_session, rec)
    committed = [json.loads(l) for l in (MCP / "session.jsonl").read_text().splitlines()]
    assert [_wire_shape(r) for r in rec.records] == [_wire_shape(r) for r in committed]


def test_gateway_decodes_names_and_answers_like_a_modern_server():
    enc = gateway.route({"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "resources/read",
                         "Mcp-Name": "=?base64?c2t5b3BzOi8vY3Jldy9aWjIxNA==?="})
    assert enc.startswith("crew-ops")
    assert "-32022" in gateway.route({"MCP-Protocol-Version": "2099-01-01", "Mcp-Method": "tools/list"})
    assert "-32601" in gateway.route({"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "ping"})
    assert gateway.route({"Mcp-Method": "initialize"}).startswith("legacy path")
