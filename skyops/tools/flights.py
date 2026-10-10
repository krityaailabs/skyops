"""Read-only flight tools for the quick-win agent.

Each tool has a Pydantic argument model. The model proposes arguments; this code validates them
before anything runs, and returns errors as data the model can read. Section 5 turns this into
full tool contracts (side-effect classes, error taxonomy, idempotency).
"""
from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from ..model.types import ToolSpec
from ..sim import flights as sim


class FindDeparturesArgs(BaseModel):
    destination: str = Field(description="IATA airport code of the destination, three letters, e.g. SIN",
                             pattern=r"^[A-Za-z]{3}$")
    date: str = Field(description="Operating date, ISO format YYYY-MM-DD", pattern=r"^\d{4}-\d{2}-\d{2}$")


class FlightStatusArgs(BaseModel):
    flight_number: str = Field(description="Flight number, carrier code then digits, e.g. ZZ214",
                               pattern=r"^[A-Za-z]{2} ?\d{1,4}$")
    date: str = Field(description="Operating date, ISO format YYYY-MM-DD", pattern=r"^\d{4}-\d{2}-\d{2}$")


def find_departures(args: FindDeparturesArgs) -> dict[str, Any]:
    rows = sim.departures_to(args.destination, args.date)
    return {"departures": [{"flight_number": f.flight_number, "scheduled_departure": f.scheduled_departure,
                            "status": f.status} for f in rows]}


def get_flight_status(args: FlightStatusArgs) -> dict[str, Any]:
    f = sim.find(args.flight_number, args.date)
    if f is None:
        return {"error": "not_found", "message": f"No flight {args.flight_number} on {args.date}."}
    return {k: v for k, v in sim.as_dict(f).items() if k not in ("origin", "operating_date")}


TOOLS: dict[str, tuple[type[BaseModel], Callable[[Any], dict[str, Any]], str]] = {
    "find_departures": (FindDeparturesArgs, find_departures,
                        "List today's departures from the hub to one destination airport. Read-only."),
    "get_flight_status": (FlightStatusArgs, get_flight_status,
                          "Get status, estimated departure time, gate and delay reason for one flight. Read-only."),
}


def specs() -> list[ToolSpec]:
    return [ToolSpec(name=n, description=d, input_schema=m.model_json_schema()) for n, (m, _, d) in TOOLS.items()]


def run_tool(name: str, raw_args: dict[str, Any]) -> tuple[bool, str]:
    """Validate, then execute. Returns (ok, JSON text for the model). Never raises to the loop."""
    if name not in TOOLS:
        return False, json.dumps({"error": "unknown_tool", "message": f"No tool named {name}."})
    model, fn, _ = TOOLS[name]
    try:
        args = model.model_validate(raw_args)
    except ValidationError as e:
        return False, json.dumps({"error": "invalid_arguments", "message": e.errors(include_url=False)[0]["msg"]})
    result = fn(args)
    return "error" not in result, json.dumps(result)
