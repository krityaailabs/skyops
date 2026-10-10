"""Lecture 7.6: the legacy crew service, before and after the facade.

    python -m labs.s07.facade_demo
"""
from __future__ import annotations

import datetime as dt

from skyops.mcp.facade import CrewArgs, get_crew_status, to_legacy
from skyops.sim import crew_soap
from skyops.tools.contracts import ToolError

D = dt.date(2026, 10, 9)


def main() -> None:
    args = CrewArgs(flight_number="zz173", date=D)
    req = to_legacy(args)
    print("legacy request: ", req[req.index("<GetRoster>"):req.index("</GetRoster>") + 12])
    resp = crew_soap.handle(req)
    print("legacy response:", resp[resp.index("<RosterResponse>"):][:150] + "…")
    print("facade, ZZ173:  ", get_crew_status(args))
    print("facade, ZZ214:  ", get_crew_status(CrewArgs(flight_number="ZZ214", date=D)))
    crew_soap.CALLS["busy_next"] = 1
    try:
        get_crew_status(args)
    except ToolError as e:
        print("legacy fault Server.Busy ->", e.as_result())
    try:
        get_crew_status(CrewArgs(flight_number="ZZ999", date=D))
    except ToolError as e:
        print("legacy fault Client.NoSuchFlight ->", e.as_result())


if __name__ == "__main__":
    main()
