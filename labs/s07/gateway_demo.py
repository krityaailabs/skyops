"""Lecture 7.9: one host, one gateway, two MCP servers behind it.

The host holds ONE token, for the gateway. The gateway checks it, decides from the headers, and calls each
backend with a token minted for that backend (never the host's). Every request leaves one audit line.
    python -m labs.s07.gateway_demo
"""
from __future__ import annotations

from skyops.mcp import auth as A
from skyops.mcp.client import HttpTarget, connect
from skyops.mcp.gateway import GATEWAY, Gateway
from skyops.mcp.server import build_crew_server, build_flight_server, http_app

D = "2026-10-09"
CALLS = [("get_flight_status", {"flight_number": "ZZ173", "date": D}),
         ("get_crew_status", {"flight_number": "ZZ173", "date": D}),
         ("notify_passengers", {"flight_number": "ZZ173", "message": "ZZ173 to Bengaluru is delayed to 18:10.",
                                "reason": "delay"})]


def build() -> Gateway:
    return Gateway({"flight-ops": http_app(build_flight_server(auth=True), auth=True),
                    "crew-ops": http_app(build_crew_server(auth=True), auth=True, resource=A.CREW_RESOURCE)})


def run(scopes: list[str]) -> tuple[Gateway, list[tuple[str, dict]]]:
    gw = build()
    token = A.mint("duty-manager-1", scopes, aud=GATEWAY)
    out = []
    with connect({"gateway": HttpTarget(GATEWAY, token, app=gw)}) as host:
        box = host.toolbox()
        for name, args in CALLS:
            t = box.tools[name]
            out.append((name, t.fn(t.args_model.model_validate(args))))
    return gw, out


def main() -> None:
    gw, out = run(["flights:read", "crew:read"])
    for name, result in out:
        print(f"{name}: {result}")
    print("audit:")
    for a in gw.audit:
        print(f"  {a['sub']} {a['method']} {a['name'] or '-'} -> {a['backend'] or '-'}: {a['outcome']}")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    main()
