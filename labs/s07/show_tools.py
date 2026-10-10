"""Lectures 7.2 and 7.3: what the SkyOps flight-ops server offers, as a client sees it.

    python -m labs.s07.show_tools                  tools: annotations and required arguments
    python -m labs.s07.show_tools --extras         plus the booking resource and the disruption prompt
"""
from __future__ import annotations

import argparse
import json

import anyio
from mcp.client import Client

from skyops.mcp.server import build_flight_server


def hints(a) -> str:
    if a is None:
        return "(no annotations)"
    flags = [n for n, v in (("read-only", a.read_only_hint), ("destructive", a.destructive_hint),
                            ("idempotent", a.idempotent_hint), ("open-world", a.open_world_hint)) if v]
    return ", ".join(flags) or "(none set)"


async def show(extras: bool) -> None:
    async with Client(build_flight_server()) as c:
        for t in (await c.list_tools()).tools:
            req = ", ".join(t.input_schema.get("required", []))
            print(f"{t.name:<18} {hints(t.annotations):<40} required: {req}")
        if extras:
            for tpl in (await c.list_resource_templates()).resource_templates:
                print(f"resource template  {tpl.uri_template}")
            r = await c.read_resource("skyops://bookings/Q7K2LM")
            print(f"read skyops://bookings/Q7K2LM  cacheScope={r.cache_scope} ttlMs={r.ttl_ms}")
            print(f"  {r.contents[0].text}")
            p = await c.get_prompt("disruption_brief", {"flight_number": "ZZ173"})
            print(f"prompt disruption_brief(ZZ173) -> {len(p.messages)} message:")
            print(f"  {p.messages[0].content.text}")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--extras", action="store_true")
    anyio.run(show, ap.parse_args().extras)
