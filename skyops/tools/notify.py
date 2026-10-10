"""A WRITE tool for Section 3: notify the passengers on a flight. Deliberately unsafe.

It has no idempotency key, no approval, no policy check and no deduplication. Each call appends a
message to an outbox that stands in for an SMS and email gateway. Section 5 adds the contract,
Section 15 the idempotency, Section 18 the approval tier and Section 19 the injection defences.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..model.types import ToolSpec


class NotifyArgs(BaseModel):
    flight_number: str = Field(description="Flight number, e.g. ZZ214", pattern=r"^[A-Za-z]{2} ?\d{1,4}$")
    message: str = Field(description="The text to send to every booked passenger", max_length=500)


SPEC = ToolSpec(name="notify_passengers",
                description="Send a message by SMS and email to every passenger booked on a flight.",
                input_schema=NotifyArgs.model_json_schema())


@dataclass
class Outbox:
    """Stands in for the messaging gateway. `fail_first_with_timeout` simulates the classic: the gateway
    sends the message, but the response is lost, so the caller sees a timeout."""
    sent: list[dict[str, str]] = field(default_factory=list)
    fail_first_with_timeout: bool = False

    def notify(self, raw: dict[str, Any]) -> tuple[bool, str]:
        try:
            a = NotifyArgs.model_validate(raw)
        except ValidationError as e:
            return False, json.dumps({"error": "invalid_arguments", "message": e.errors(include_url=False)[0]["msg"]})
        self.sent.append({"flight_number": a.flight_number.replace(" ", "").upper(), "message": a.message})
        if self.fail_first_with_timeout and len(self.sent) == 1:
            return False, json.dumps({"error": "timeout", "message": "gateway did not respond within 10 s"})
        return True, json.dumps({"status": "sent", "recipients": 212})
