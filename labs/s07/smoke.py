"""Lecture 7.10: the host's smoke test against a deployed SkyOps MCP server.

    python -m labs.s07.smoke https://127.0.0.1:8443/mcp runs/certs/ca.pem
Five checks, each PASS or FAIL. It exits non-zero on any FAIL, so a deploy pipeline can stop on it.
"""
from __future__ import annotations

import sys

import anyio
import httpx2
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client

from skyops.mcp import auth as A

META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}
H = {"accept": "application/json, text/event-stream", "content-type": "application/json",
     "mcp-protocol-version": "2026-07-28", "mcp-method": "tools/list"}


async def run(url: str, ca: str) -> int:
    fails = 0

    def check(ok: bool, what: str) -> None:
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {what}")

    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": META}}
    async with httpx2.AsyncClient(verify=ca, timeout=10) as raw:
        r = await raw.post(url, json=body, headers=H)
        check(r.status_code == 401 and "resource_metadata=" in r.headers.get("www-authenticate", ""),
              "no token: 401 with a resource_metadata pointer")
        meta = await raw.get(url.rsplit("/mcp", 1)[0] + "/.well-known/oauth-protected-resource/mcp")
        check(meta.status_code == 200 and meta.json().get("resource") == A.RESOURCE,
              f"protected resource metadata names {A.RESOURCE}")
        other = A.mint("smoke", ["flights:read"], aud="https://gateway.skyops.example/mcp")
        r = await raw.post(url, json=body, headers={**H, "authorization": f"Bearer {other}"})
        check(r.status_code == 401, "token for another audience: 401 (no passthrough)")
    good = A.mint("smoke", ["flights:read"])
    http = httpx2.AsyncClient(verify=ca, timeout=10, headers={"authorization": f"Bearer {good}"})
    async with Client(streamable_http_client(url, http_client=http)) as c:
        tools = [t.name for t in (await c.list_tools()).tools]
        check("get_flight_status" in tools, f"tools/list over TLS: {len(tools)} tools")
        res = await c.call_tool("get_flight_status", {"flight_number": "ZZ214", "date": "2026-10-09"})
        check(not res.is_error and res.structured_content.get("gate") == "B7", "get_flight_status ZZ214: gate B7")
    print("smoke test:", "PASS" if not fails else f"{fails} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    sys.exit(anyio.run(run, sys.argv[1], sys.argv[2]))
