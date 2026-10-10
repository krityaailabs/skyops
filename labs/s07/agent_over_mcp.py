"""Lecture 7.5: the Section 4 agent, with its tools coming from two MCP servers instead of local functions.

flight-ops runs in-process (no HTTP); crew-ops is reached over Streamable HTTP with a bearer token scoped to
crew:read, served in memory. The loop, budgets and trace are unchanged: only where the tools come from moved.
    python -m labs.s07.agent_over_mcp
The model's turns are a scripted replay (fixtures/s07/agent_mcp.jsonl); the MCP traffic is real.
"""
from __future__ import annotations

from skyops.agent.loop import RunResult, run_agent
from skyops.mcp import auth as A
from skyops.mcp.client import Host, HttpTarget, connect
from skyops.mcp.server import build_crew_server, build_flight_server, http_app
from skyops.model.client import get_client

QUESTION = "Will the evening flight to Singapore leave tonight, and does it have a full crew?"
FIXTURE = "fixtures/s07/agent_mcp.jsonl"


def targets() -> dict:
    crew = http_app(build_crew_server(auth=True), auth=True, resource=A.CREW_RESOURCE)
    token = A.mint("duty-manager-1", ["crew:read"], aud=A.CREW_RESOURCE)
    return {"flight-ops": build_flight_server(), "crew-ops": HttpTarget(A.CREW_RESOURCE, token, app=crew)}


def run(client=None) -> tuple[RunResult, Host]:
    with connect(targets()) as host:
        box = host.toolbox()
        r = run_agent(QUESTION, client or get_client(fixture=FIXTURE, match="strict"), box)
        return r, host


def main() -> None:
    r, host = run()
    print(r.trace.render())
    servers = sorted({c["server"] for c in host.calls})
    print(f"tool calls over MCP: {len(host.calls)} ({', '.join(servers)})   stop: {r.stop}")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    main()
