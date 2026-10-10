"""Synthetic bookings (no real people). Section 7 exposes one booking at a time as an MCP resource."""
from __future__ import annotations

BOOKINGS = {
    "Q7K2LM": {"booking_ref": "Q7K2LM", "flight_number": "ZZ247", "date": "2026-10-09", "cabin": "economy",
               "passengers": 2, "contact": "sms", "status": "disrupted: cancellation"},
    "H4TR9B": {"booking_ref": "H4TR9B", "flight_number": "ZZ214", "date": "2026-10-09", "cabin": "business",
               "passengers": 1, "contact": "email", "status": "disrupted: delay"},
    "M2PX7C": {"booking_ref": "M2PX7C", "flight_number": "ZZ261", "date": "2026-10-09", "cabin": "economy",
               "passengers": 3, "contact": "sms", "status": "confirmed"},
}


def get(booking_ref: str) -> dict | None:
    return BOOKINGS.get(booking_ref.strip().upper())
