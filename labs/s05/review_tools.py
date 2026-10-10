"""Lecture 5.8 (review): ten tools that look fine and are not.

Each is a real-looking contract with at least one problem. `lint_contract` (which you write in this lecture)
finds some of them automatically. The rest need a reviewer: lint reads the contract, not what the tool
returns or how much power it has. Run:  python -m labs.s05.review_tools
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from skyops.tools.contracts import SideEffect, ToolContract, lint_contract


def _noop(_: Any) -> dict[str, Any]:
    return {}


class SqlArgs(BaseModel):
    sql: str = Field(description="SQL to run against the operations database.")


class PassengerArgs(BaseModel):
    booking_ref: str = Field(description="Six-character booking reference.")


class FlightsArgs(BaseModel):
    date: str = Field(description="Operating date, YYYY-MM-DD.")


class BookingUpdateArgs(BaseModel):
    booking_ref: str = Field(description="Six-character booking reference.")
    new_flight: str = Field(description="Flight number to move the booking to.")


class RefundArgs(BaseModel):
    booking_ref: str = Field(description="Six-character booking reference.")
    amount: float = Field(description="Amount to refund.")
    card_number: str = Field(description="Card to refund to.")


class SmsArgs(BaseModel):
    phone: str = Field(description="Phone number in international format.")
    text: str = Field(description="Message text.")


class SearchArgs(BaseModel):
    q: str


class RosterArgs(BaseModel):
    flight_number: str = Field(description="Flight number, e.g. ZZ214.")


class CancelArgs(BaseModel):
    flight_number: str = Field(description="Flight number, e.g. ZZ214.")
    date: str = Field(description="Operating date, YYYY-MM-DD.")


class ReportArgs(BaseModel):
    expression: str = Field(description="Filter expression evaluated over today's flights.")


R, RW, IW = SideEffect.READ, SideEffect.REVERSIBLE_WRITE, SideEffect.IRREVERSIBLE_WRITE

TOOLS = [
    ToolContract("query", "Run a query against the operations database and return the rows.", SqlArgs, _noop, R,
                 errors=["invalid_arguments"]),
    ToolContract("get_passenger", "Return the full passenger record for a booking, as stored in the reservation system.",
                 PassengerArgs, _noop, R, errors=["not_found"]),
    ToolContract("flights", "Return every flight record for a date, with all fields, for any purpose.", FlightsArgs,
                 _noop, R, errors=["invalid_arguments"]),
    ToolContract("update_booking", "Move a booking to another flight. This is reversible by moving it back.",
                 BookingUpdateArgs, _noop, RW, errors=["not_found", "no_seat"]),
    ToolContract("pay_refund", "Refund a passenger to their card. Irreversible once the payment provider accepts it.",
                 RefundArgs, _noop, IW, errors=["declined"]),
    ToolContract("send_sms", "Sends an SMS.", SmsArgs, _noop, IW, errors=["unavailable"]),
    ToolContract("search", "Search operational documents and messages for anything relevant to the question.",
                 SearchArgs, _noop, R, errors=["invalid_arguments"]),
    ToolContract("get_crew_roster", "Return the crew roster for a flight, including the roster system's connection "
                 "details so follow-up calls can reuse them.", RosterArgs, _noop, R, errors=["not_found"]),
    ToolContract("cancel_flight", "Cancel a flight in the operations system. Irreversible: rebooking starts at once. "
                 "Idempotent per flight and date.", CancelArgs, _noop, IW, errors=["not_found", "already_cancelled"],
                 key_fields=["flight_number", "date"]),
    ToolContract("run_report", "Evaluate a filter expression over today's flights and return the matching rows.",
                 ReportArgs, _noop, R),
]

# What a reviewer finds. "lint" = lint_contract should report it; "human" = only a person reading the design will.
FINDINGS: dict[str, list[tuple[str, str]]] = {
    "query": [("lint", "free_form_query"), ("lint", "vague_name"), ("human", "any SQL, any table: no least privilege")],
    "get_passenger": [("human", "returns the whole record, including passport and contact data the task never needs")],
    "flights": [("lint", "vague_name"), ("human", "all fields, all flights: unbounded result size")],
    "update_booking": [("lint", "write_without_key"), ("human", "reversible in theory; a seat may be gone by the time you move back")],
    "pay_refund": [("lint", "secret_argument"), ("lint", "write_without_key"),
                   ("human", "no limit and no approval: unsafe payments")],
    "send_sms": [("lint", "short_description"), ("lint", "no_side_effect_in_description"), ("lint", "write_without_key"),
                 ("human", "a raw phone number: the agent can text anyone")],
    "search": [("lint", "vague_name"), ("lint", "undocumented_field"),
               ("human", "searches messages: untrusted text enters the conversation with no source attached")],
    "get_crew_roster": [("human", "secrets in output: connection details reach the model, the logs and maybe the user")],
    "cancel_flight": [("human", "excessive agency: an agent should propose a cancellation, a person decides")],
    "run_report": [("lint", "free_form_query"), ("lint", "no_documented_errors"),
                   ("human", "an expression evaluator is a code-execution surface")],
}


def main() -> None:
    from skyops.model.errors import LabNotDone
    caught = missed = 0
    for t in TOOLS:
        try:
            found = set(lint_contract(t))
        except LabNotDone as e:
            print(f"LAB NOT DONE: {e}")
            return
        expected = {code for kind, code in FINDINGS[t.name] if kind == "lint"}
        human = [code for kind, code in FINDINGS[t.name] if kind == "human"]
        caught += len(found & expected)
        missed += len(human)
        print(f"{t.name:16} lint: {', '.join(sorted(found)) or '(nothing)'}")
        for h in human:
            print(f"{'':16} reviewer: {h}")
    print(f"lint found {caught} problems; {missed} need a reviewer")


if __name__ == "__main__":
    main()
