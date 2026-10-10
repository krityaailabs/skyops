"""Lecture 7.1: your first MCP server, with the official Python SDK v2.

One tool, get_flight_status, over the Section 4 simulator. Three ways to run it:
    python -m labs.s07.first_server            in-process: an SDK client lists the tools and calls one
    python -m labs.s07.first_server --stdio    as a stdio server, for a host or the MCP Inspector:
        npx @modelcontextprotocol/inspector@2.10.1 --cli --config labs/s07/inspector.json --server skyops-first \
            --method tools/list     (a config file: the Inspector would swallow python's own -m)
Older tutorials import FastMCP from mcp.server.fastmcp: that is SDK v1. In v2 the class is MCPServer.
"""
from __future__ import annotations

import argparse
import json
from typing import Annotated

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from skyops.sim import flights as sim
from skyops.model.errors import LabNotDone


def build() -> MCPServer:
    """Lecture 7.1: an MCPServer named "skyops-first", version "0.1.0", with ONE tool:

        get_flight_status(flight_number, date) -> dict[str, str | None]
            flight_number: str, described "Flight number: ZZ then digits, e.g. ZZ214."
            date: str, described "Operating date at the hub, YYYY-MM-DD."
            Docstring (the model reads it as the description): "Get status, estimated departure, gate and
            delay reason for one flight on one date. Read-only."
            Look the flight up with sim.find(flight_number.replace(" ", "").upper(), date). Not found: raise
            ToolError(f"No flight {flight_number} on {date}.") - an anticipated failure, so the model reads
            the message; any other exception is a crash and the model sees only "Error executing tool".
            Found: return sim.as_dict(flight).

    Use Annotated[str, Field(description=...)] for the parameters, so the schema carries the descriptions,
    and @srv.tool(structured_output=True), so the dict comes back as structuredContent as well as text.
    """
    raise LabNotDone("Lecture 7.1: implement build(). Hints: labs/s07/README.md, lecture 7.1.")


async def inspect_in_process() -> None:
    """What an inspector does: list the tools, show each schema, call one."""
    from mcp.client import Client
    async with Client(build()) as client:
        tools = (await client.list_tools()).tools
        for t in tools:
            print(f"tool {t.name}: {t.description}")
            for p, s in t.input_schema["properties"].items():
                print(f"  {p}: {s['type']} - {s.get('description', '')}")
        r = await client.call_tool("get_flight_status", {"flight_number": "ZZ214", "date": "2026-10-09"})
        print("call get_flight_status(ZZ214, 2026-10-09):", json.dumps(r.structured_content))
        r = await client.call_tool("get_flight_status", {"flight_number": "ZZ999", "date": "2026-10-09"})
        print("call get_flight_status(ZZ999, 2026-10-09): isError =", r.is_error, "|", r.content[0].text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdio", action="store_true", help="serve over stdio (for a host or the MCP Inspector)")
    a = ap.parse_args(argv)
    if a.stdio:
        build().run("stdio")
    else:
        anyio.run(inspect_in_process)


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    main()
