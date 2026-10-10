"""Lecture 7.6: an MCP facade over the legacy SOAP-style crew service (skyops/sim/crew_soap.py).

The facade is an anti-corruption layer. The model sees one small contract in SkyOps words:
    get_crew_status(flight_number, date) -> {flight_number, date, captain, first_officer, cabin_crew,
                                             report_time, duty_limit, complete}
Everything legacy stays behind it: XML envelopes, the split carrier/zero-padded number, DDMONYY dates,
"1810L" times, rank codes, staff numbers and names (personal data the task never needs), and SOAP faults,
which become Section 5 taxonomy errors (Server.Busy -> retryable; Client.NoSuchFlight -> not_found;
anything unmapped -> logged with its fault code, and needs_human "legacy_fault").
"""
from __future__ import annotations

import datetime as dt
import logging
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

from ..sim import crew_soap
from ..tools.contracts import SideEffect, ToolContract, ToolError

log = logging.getLogger("skyops.mcp.facade")
RANKS = {"CP": "captain", "FO": "first_officer", "PU": "cabin_crew", "CC": "cabin_crew"}
MINIMUM = {"captain": 1, "first_officer": 1, "cabin_crew": 4}      # synthetic minimum crew for these aircraft


class CrewArgs(BaseModel):
    flight_number: str = Field(description="Flight number: carrier code ZZ then digits, e.g. ZZ214.")
    date: dt.date = Field(description="Operating date at the hub, ISO format YYYY-MM-DD.")

    @field_validator("flight_number")
    @classmethod
    def normalise(cls, v: str) -> str:
        v = v.replace(" ", "").upper()
        if not (v.startswith("ZZ") and v[2:].isdigit()):
            raise ValueError("flight number must be ZZ followed by digits, e.g. ZZ214")
        return v


def to_legacy(args: CrewArgs) -> str:
    """SkyOps words -> the legacy request. The only place that knows the legacy formats."""
    number = args.flight_number[2:].zfill(4)
    day = args.date.strftime("%d%b%y").upper()
    return (f'<soap:Envelope xmlns:soap="{crew_soap.NS}"><soap:Body><GetRoster><Carrier>ZZ</Carrier>'
            f"<FltNbr>{number}</FltNbr><DepDate>{day}</DepDate></GetRoster></soap:Body></soap:Envelope>")


def _clock(v: str) -> str:
    """'1810L' -> '18:10' (local hub time)."""
    return f"{v[:2]}:{v[2:4]}"


def from_legacy(xml: str, args: CrewArgs) -> dict[str, Any]:
    """The legacy response -> SkyOps words, or a taxonomy error. Names and staff numbers never leave."""
    body = ET.fromstring(xml).find(f"{{{crew_soap.NS}}}Body")
    fault = body.find(f"{{{crew_soap.NS}}}Fault")
    if fault is not None:
        code = fault.findtext("faultcode", "")
        if code.endswith("Server.Busy"):
            raise ToolError("retryable", "unavailable", "The crew system is busy; try again shortly.")
        if code.endswith("Client.NoSuchFlight"):
            raise ToolError("non_retryable", "not_found", f"No crew roster for {args.flight_number} on {args.date}.")
        # A fault we have no mapping for: log the code (never the fault text, which can echo roster data)
        # so a person sees it, and tell the model a person must look rather than inviting a retry.
        log.error("crew facade: unmapped legacy fault %s for %s", code[:80], args.flight_number)
        raise ToolError("needs_human", "legacy_fault", "The crew system rejected the request; operations has been told.")
    resp = body.find("RosterResponse")
    counts = Counter(RANKS.get(m.findtext("Rank"), "other") for m in resp.findall("CrewMember"))
    out = {"flight_number": args.flight_number, "date": args.date.isoformat(),
           **{k: counts.get(k, 0) for k in ("captain", "first_officer", "cabin_crew")},
           "report_time": _clock(resp.findtext("ReportTime")), "duty_limit": _clock(resp.findtext("DutyLimit"))}
    out["complete"] = all(out[k] >= n for k, n in MINIMUM.items())
    return out


def get_crew_status(args: CrewArgs, call: Callable[[str], str] = crew_soap.handle) -> dict[str, Any]:
    return from_legacy(call(to_legacy(args)), args)


def crew_contract() -> ToolContract:
    return ToolContract(
        "get_crew_status",
        "Get crew numbers by role, report time and duty-time limit for one flight on one date. Read-only. "
        "Compare duty_limit with the flight's estimated departure: a crew can't operate past it. "
        "No names: use the crew planning tools for people.",
        CrewArgs, get_crew_status, SideEffect.READ, errors=["invalid_arguments", "not_found", "unavailable"])
