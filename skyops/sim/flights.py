"""Deterministic flight-operations simulator.

Synthetic data only. "ZZ" is a placeholder carrier code and "HUB" a fictional hub airport;
destinations use real airport codes purely as places. Record fields are named in the spirit of
AIDX-style flight messages (flight identification, operational times, resources); Section 16
puts an anti-corruption layer between messages like these and the agent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

DATE = "2026-10-09"


@dataclass(frozen=True)
class FlightRecord:
    flight_number: str          # e.g. "ZZ214"
    operating_date: str         # ISO date, local to the hub
    origin: str
    destination: str
    scheduled_departure: str    # local time, 24-hour HH:MM
    estimated_departure: str
    status: str                 # SCHEDULED | BOARDING | DELAYED | CANCELLED | DEPARTED
    gate: str | None
    delay_reason: str | None


_FLIGHTS = [
    FlightRecord("ZZ101", DATE, "HUB", "LHR", "07:10", "07:10", "DEPARTED", "A3", None),
    FlightRecord("ZZ118", DATE, "HUB", "FRA", "09:45", "10:30", "DEPARTED", "A7", "Late inbound aircraft"),
    FlightRecord("ZZ142", DATE, "HUB", "JFK", "13:20", "13:20", "BOARDING", "C2", None),
    FlightRecord("ZZ156", DATE, "HUB", "DXB", "15:05", "15:05", "SCHEDULED", "B4", None),
    FlightRecord("ZZ173", DATE, "HUB", "BLR", "16:40", "18:10", "DELAYED", "B9", "Crew duty-time limit"),
    FlightRecord("ZZ188", DATE, "HUB", "SIN", "08:15", "08:15", "DEPARTED", "C5", None),
    FlightRecord("ZZ214", DATE, "HUB", "SIN", "19:40", "20:15", "DELAYED", "B7", "Late inbound aircraft"),
    FlightRecord("ZZ230", DATE, "HUB", "SYD", "21:55", "21:55", "SCHEDULED", "C8", None),
    FlightRecord("ZZ247", DATE, "HUB", "NRT", "22:30", "22:30", "CANCELLED", None, "Aircraft technical inspection"),
    FlightRecord("ZZ261", DATE, "HUB", "CDG", "23:05", "23:05", "SCHEDULED", "A1", None),
]


def find(flight_number: str, operating_date: str) -> FlightRecord | None:
    key = flight_number.replace(" ", "").upper()
    return next((f for f in _FLIGHTS if f.flight_number == key and f.operating_date == operating_date), None)


def departures_to(destination: str, operating_date: str) -> list[FlightRecord]:
    return [f for f in _FLIGHTS if f.destination == destination.upper() and f.operating_date == operating_date]


def departures_on(operating_date: str) -> list[FlightRecord]:
    """Every departure on a date, in timetable order."""
    return sorted((f for f in _FLIGHTS if f.operating_date == operating_date), key=lambda f: f.scheduled_departure)


def as_dict(f: FlightRecord) -> dict:
    return asdict(f)
