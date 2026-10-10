"""Section 6 (read only): a small SkyOps MCP server, used to RECORD real wire traffic for the inspector.

Built with the official Python SDK v2 (`mcp` 2.3.0, protocol revision 2026-07-28). You don't need to
change anything here in Section 6: lecture 6.3 reads what this server and the SDK's own client put on the
wire. Section 7 builds the full server, with the Section 5 contracts, authorisation and tests.

What it exposes, one of each MCP primitive:
    tool      get_flight_status    read; flight_number mirrored into an Mcp-Param-Flight header
    tool      notify_passengers    irreversible write; asks for confirmation with a Multi Round-Trip Request
    resource  skyops://bookings/{booking_ref}   a booking, private to the caller (cacheScope "private")
    prompt    disruption_brief     a template a duty manager picks from a menu

Caching hints: discover and tools/list are the same for every caller, so "public"; a booking is per caller,
so "private". The requestState sealed by the SDK is AES-GCM with a fixed course key; never ship a key in code.
"""
from __future__ import annotations

from typing import Annotated

from mcp.server import MCPServer
from mcp.server.caching import CacheHint
from mcp.server.mcpserver import Elicit, Resolve
from mcp.server.request_state import RequestStateSecurity
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from skyops.sim import flights as sim
from skyops.tools.catalog import BOOKED

NAME, VERSION = "skyops-ops", "0.6.0"
TOOLS_TTL_MS = 300_000          # tools/list: five minutes, any caller
DISCOVER_TTL_MS = 3_600_000     # server/discover: an hour, any caller
BOOKING_TTL_MS = 60_000         # resources/read: a minute, same caller only
STATE_TTL_S = 300               # a sealed requestState is refused after five minutes
COURSE_KEY = b"skyops-course-demo-key-not-secret"[:32]   # demo only: real keys come from a secret store

SENT: list[dict] = []           # what notify_passengers "sent" (no real gateway in Section 6)


class FlightStatus(BaseModel):
    flight_number: str
    destination: str
    scheduled_departure: str
    estimated_departure: str
    status: str
    gate: str | None
    delay_reason: str | None


class Confirm(BaseModel):
    confirm: bool = Field(description="Send this message to every booked passenger now?")


def confirm_send(flight_number: str, message: str) -> Elicit[Confirm]:
    """Ask the person in the host to confirm before an irreversible send (MRTR: input_required)."""
    return Elicit(f"Send to all {BOOKED.get(flight_number, 0)} passengers on {flight_number}: {message!r}?", Confirm)


def build() -> MCPServer:
    SENT.clear()                    # each build is a fresh server with nothing sent yet
    srv = MCPServer(
        NAME, version=VERSION,
        instructions="SkyOps flight operations at the hub. Look flights up before you tell anyone anything.",
        cache_hints={"server/discover": CacheHint(DISCOVER_TTL_MS, "public"),
                     "tools/list": CacheHint(TOOLS_TTL_MS, "public"),
                     "prompts/list": CacheHint(TOOLS_TTL_MS, "public"),
                     "resources/templates/list": CacheHint(TOOLS_TTL_MS, "public"),
                     "resources/read": CacheHint(BOOKING_TTL_MS, "private")},
        request_state_security=RequestStateSecurity(keys=[COURSE_KEY], ttl=STATE_TTL_S, bind_principal=None),
    )

    @srv.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def get_flight_status(
        flight_number: Annotated[str, Field(description="Flight number: ZZ then digits, e.g. ZZ214.",
                                            json_schema_extra={"x-mcp-header": "Flight"})],
        date: Annotated[str, Field(description="Operating date at the hub, YYYY-MM-DD.")],
    ) -> FlightStatus:
        """Get status, estimated departure, gate and delay reason for one flight on one date. Read-only."""
        f = sim.find(flight_number, date)
        if f is None:
            raise ValueError(f"No flight {flight_number} on {date}.")
        return FlightStatus(**{k: v for k, v in sim.as_dict(f).items() if k in FlightStatus.model_fields})

    @srv.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                          openWorldHint=True))
    def notify_passengers(
        flight_number: Annotated[str, Field(description="Flight number: ZZ then digits, e.g. ZZ247.")],
        message: Annotated[str, Field(min_length=10, max_length=320, description="Plain text, at most 320 characters.")],
        ok: Annotated[Confirm, Resolve(confirm_send)],
    ) -> dict:
        """Send one SMS and email to every passenger booked on a flight. Irreversible: asks a person to confirm."""
        if not ok.confirm:
            return {"status": "not_sent", "reason": "declined by the duty manager"}
        # A sealed requestState can be replayed until it expires; it is not single-use. So the send itself
        # is idempotent, keyed on flight and message as in Section 5: a replayed retry returns the first result.
        key = (flight_number, " ".join(message.split()).lower())
        if any((m["flight_number"], " ".join(m["message"].split()).lower()) == key for m in SENT):
            return {"status": "sent", "recipients": BOOKED.get(flight_number, 0), "replayed": True}
        SENT.append({"flight_number": flight_number, "message": message})
        return {"status": "sent", "recipients": BOOKED.get(flight_number, 0)}

    @srv.resource("skyops://bookings/{booking_ref}", mime_type="application/json",
                  description="One booking: flight, cabin and contact preference. Private to the caller.")
    def booking(booking_ref: str) -> str:
        import json
        return json.dumps({"booking_ref": booking_ref, "flight_number": "ZZ247", "cabin": "economy",
                           "contact": "sms"})

    @srv.prompt(description="Draft a disruption brief for one flight, for the duty manager to review.")
    def disruption_brief(flight_number: str) -> str:
        return (f"Look up {flight_number} with get_flight_status. Write a three-line brief for the duty manager: "
                "status, cause, and the next decision needed. Don't message passengers.")

    return srv
