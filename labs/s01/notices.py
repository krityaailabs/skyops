"""Shared data for the Section 1 labs: the DisruptionNotice contract and five synthetic ops notices."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Developer note: every field is required but may be null. Strict structured-output modes want every
# property listed as required and nothing extra, so this one schema stays valid on every provider.
# The class docstring and field descriptions are sent to the model inside the schema: write them for it.
class DisruptionNotice(BaseModel):
    """One airline operations message about a single departing flight, as structured data."""
    model_config = ConfigDict(extra="forbid")

    flight_number: str = Field(pattern=r"^ZZ\d{1,4}$", description="Carrier code ZZ then digits, no space, e.g. ZZ173")
    status: Literal["on_time", "delayed", "cancelled", "diverted"] = Field(
        description="on_time if the departure time is unchanged (a gate change alone is still on_time)")
    new_departure_local: str | None = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
                                            description="New local departure time HH:MM, or null if unchanged or unknown")
    gate: str | None = Field(description="Departure gate, or null if none")
    reason: str | None = Field(description="Short reason in plain words, or null if none given")
    passengers_affected: int | None = Field(ge=0, description="Booked passengers, or null if not stated")

    @model_validator(mode="after")
    def _business_rules(self) -> "DisruptionNotice":
        # Rules that span fields: a JSON schema can't express them, so they live in code.
        if self.status in ("on_time", "cancelled") and self.new_departure_local is not None:
            raise ValueError(f"a {self.status} flight can't have a new departure time")
        return self


NOTICES = [
    "ZZ173 to BLR now ETD 18:10 local, was 16:40. Crew out of hours, standby crew called. Gate stays B9. 212 pax booked.",
    "CNX ZZ247 HUB-NRT 09OCT, aircraft tech inspection. Rebook via ops desk. 286 pax.",
    "ZZ142 JFK boarding on time, gate C2.",
    "ZZ305 inbound from MAD diverted to VLC, weather at HUB, onward times TBC.",
    "Gate change ZZ261 CDG now A4 (was A1). Dep 23:05 unchanged.",
]

EXPECTED = [
    DisruptionNotice(flight_number="ZZ173", status="delayed", new_departure_local="18:10", gate="B9",
                     reason="Crew out of hours; standby crew called", passengers_affected=212),
    DisruptionNotice(flight_number="ZZ247", status="cancelled", new_departure_local=None, gate=None,
                     reason="Aircraft technical inspection", passengers_affected=286),
    DisruptionNotice(flight_number="ZZ142", status="on_time", new_departure_local=None, gate="C2",
                     reason=None, passengers_affected=None),
    DisruptionNotice(flight_number="ZZ305", status="diverted", new_departure_local=None, gate=None,
                     reason="Weather at HUB; diverted to VLC", passengers_affected=None),
    DisruptionNotice(flight_number="ZZ261", status="on_time", new_departure_local=None, gate="A4",
                     reason="Gate change from A1", passengers_affected=None),
]

SYSTEM = ("You extract structured data from airline operations messages. Use only what the message says. "
          "If a value is not stated, use null. Times are local, 24-hour HH:MM.")
