"""Lecture 7.4: what an authenticated SkyOps MCP server answers, request by request.

Six raw HTTP requests against flight-ops (in memory): no token, a token for another audience, a valid
read token, a write without its scope, the same write with it, and the protected-resource metadata.
    python -m labs.s07.auth_demo
"""
from __future__ import annotations

import anyio
import httpx2

from skyops.mcp import auth as A
from skyops.mcp.server import build_flight_server, http_app

META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}
NOTIFY = {"name": "notify_passengers", "arguments": {"flight_number": "ZZ247", "reason": "cancellation",
                                                      "message": "ZZ247 tonight is cancelled. Rebooking options follow."}}


def headers(method: str, name: str | None, token: str | None) -> dict[str, str]:
    h = {"accept": "application/json, text/event-stream", "content-type": "application/json",
         "mcp-protocol-version": "2026-07-28", "mcp-method": method}
    if name:
        h["mcp-name"] = name
    if token:
        h["authorization"] = f"Bearer {token}"
    return h


async def main() -> None:
    app = http_app(build_flight_server(auth=True), auth=True)
    cases = [
        ("no token", "tools/list", None, None),
        ("token for the gateway, not this server", "tools/list", None,
         A.mint("duty-manager-1", ["flights:read"], aud="https://gateway.skyops.example/mcp")),
        ("flights:read", "tools/list", None, A.mint("duty-manager-1", ["flights:read"])),
        ("flights:read, then a write", "tools/call", NOTIFY, A.mint("duty-manager-1", ["flights:read"])),
        ("flights:read passengers:notify", "tools/call", NOTIFY,
         A.mint("duty-manager-1", ["flights:read", "passengers:notify"])),
    ]
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="https://mcp.skyops.example") as c:
            for i, (label, method, params, token) in enumerate(cases, 1):
                body = {"jsonrpc": "2.0", "id": i, "method": method, "params": {**(params or {}), "_meta": META}}
                r = await c.post("/mcp", json=body, headers=headers(method, (params or {}).get("name"), token))
                print(f"{i}. {label}: {r.status_code}")
                if "www-authenticate" in r.headers:
                    print(f"   WWW-Authenticate: {r.headers['www-authenticate']}")
                elif r.status_code == 200:
                    res = r.json()["result"]
                    print("   " + (f"tools: {len(res['tools'])}" if "tools" in res else
                                  f"result: {res['structuredContent']['status']}, {res['structuredContent']['recipients']} recipients"))
            m = await c.get("/.well-known/oauth-protected-resource/mcp")
            print(f"6. protected resource metadata: {m.status_code}")
            print(f"   {m.text}")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    anyio.run(main)
