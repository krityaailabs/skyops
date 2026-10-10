"""Lecture 2.1 (demo): Pydantic as the contract layer.

A contract is the shape AND the rules of data crossing a boundary. With language models in the
loop, more data crosses boundaries you don't control, so contracts matter more, not less.
Run:  python -m labs.s02.contracts
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Status = Literal["SCHEDULED", "BOARDING", "DELAYED", "CANCELLED", "DEPARTED"]


class FlightStatus(BaseModel):
    """What the Flight Operations API returns for one flight on one day."""
    model_config = ConfigDict(extra="ignore", frozen=True)   # ignore new upstream fields; never mutate

    flight_number: str = Field(pattern=r"^ZZ\d{1,4}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    scheduled_departure: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    estimated_departure: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    status: Status
    gate: str | None = None
    delay_reason: str | None = None

    @field_validator("flight_number", mode="before")
    @classmethod
    def normalise_flight(cls, v: str) -> str:
        """Accept "zz 214" from a person or a model; store "ZZ214"."""
        return v.replace(" ", "").upper() if isinstance(v, str) else v


def main() -> None:
    ok = FlightStatus.model_validate({"flight_number": "zz 214", "destination": "SIN", "scheduled_departure": "19:40",
                                      "estimated_departure": "20:15", "status": "DELAYED", "gate": "B7",
                                      "delay_reason": "Late inbound aircraft", "new_upstream_field": 1})
    print("parsed:", ok.model_dump(mode="json"))
    try:
        FlightStatus.model_validate({"flight_number": "ZZ214", "destination": "Singapore",
                                     "scheduled_departure": "7:40pm", "estimated_departure": "20:15", "status": "LATE"})
    except ValidationError as e:
        print(f"rejected with {e.error_count()} errors:")
        for err in e.errors(include_url=False):
            print(f"  {'.'.join(map(str, err['loc']))}: {err['type']}")
    print("schema keys:", sorted(json.loads(json.dumps(FlightStatus.model_json_schema()))["properties"]))


if __name__ == "__main__":
    main()
