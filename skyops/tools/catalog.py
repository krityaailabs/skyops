"""The SkyOps tool catalog with full contracts (Section 5 lab, lecture 5.9).

Four tools, one of each kind the course needs:
    list_departures      read, paged and shaped
    get_flight_status    read, typed date, a documented not_found
    notify_passengers    irreversible write: idempotency key, dry run
    issue_voucher        irreversible write: idempotency key, a limit above which a person must decide

Every tool fails only through the ToolError taxonomy. `lint_contract` must find nothing here.
"""
from __future__ import annotations

import base64
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..sim import flights as sim
from .contracts import IdempotencyStore, SideEffect, ToolContract, ToolError, as_tool, shape_result

VOUCHER_LIMIT_EUR = 150          # above this, a person approves (Section 18 turns this into an approval tier)
PAGE_SIZE = 5
BOOKED = {"ZZ101": 168, "ZZ118": 174, "ZZ142": 296, "ZZ156": 188, "ZZ173": 212, "ZZ188": 180, "ZZ214": 212,
          "ZZ230": 301, "ZZ247": 254, "ZZ261": 176}


def _flight(v: str) -> str:
    v = v.replace(" ", "").upper()
    if not (len(v) >= 3 and v[:2] == "ZZ" and v[2:].isdigit()):
        raise ValueError("flight number must be ZZ followed by digits, e.g. ZZ214")
    return v


class ListDeparturesArgs(BaseModel):
    date: dt.date = Field(description="Operating date at the hub, ISO format YYYY-MM-DD.")
    destination: str | None = Field(default=None, pattern=r"^[A-Za-z]{3}$",
                                    description="Optional IATA destination code, three letters, e.g. SIN. Omit for all.")
    cursor: str | None = Field(default=None, max_length=40,
                               description="Opaque page token: pass next_cursor from the previous page; omit for the first page.")


class StatusArgs(BaseModel):
    flight_number: str = Field(description="Flight number: carrier code ZZ then digits, e.g. ZZ214.")
    date: dt.date = Field(description="Operating date at the hub, ISO format YYYY-MM-DD.")

    @field_validator("flight_number")
    @classmethod
    def normalise_flight(cls, v: str) -> str:
        return _flight(v)


class NotifyArgs(BaseModel):
    flight_number: str = Field(description="Flight number: carrier code ZZ then digits, e.g. ZZ214.")
    message: str = Field(min_length=10, max_length=320,
                         description="Text sent by SMS and email to every booked passenger. Plain text, at most 320 characters.")
    reason: Literal["delay", "gate_change", "cancellation", "diversion"] = Field(
        description="Why passengers are being told: delay, gate_change, cancellation or diversion.")

    @field_validator("flight_number")
    @classmethod
    def normalise_flight(cls, v: str) -> str:
        return _flight(v)


class VoucherArgs(BaseModel):
    booking_ref: str = Field(pattern=r"^[A-Z0-9]{6}$", description="Six-character booking reference, e.g. Q7K2LM.")
    flight_number: str = Field(description="The disrupted flight: carrier code ZZ then digits, e.g. ZZ247.")
    kind: Literal["meal", "hotel"] = Field(description="Voucher type: meal or hotel.")
    amount_eur: int = Field(gt=0, le=1000, description="Voucher value in whole euros.")

    @field_validator("flight_number")
    @classmethod
    def normalise_flight(cls, v: str) -> str:
        return _flight(v)


@dataclass
class Gateways:
    """Stand-ins for the messaging gateway and the voucher system. Inspect them in tests.
    `lose_next_response`: the gateway SENDS the next message, then raises TimeoutError, as in Section 3."""
    sent: list[dict[str, Any]] = field(default_factory=list)
    vouchers: list[dict[str, Any]] = field(default_factory=list)
    lose_next_response: bool = False


def encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(f"offset:{offset}".encode()).decode()


def decode_cursor(token: str | None) -> int:
    """Turn a page token back into an offset. A token we didn't issue is an invalid argument, not a crash."""
    if token is None:
        return 0
    try:
        kind, _, value = base64.urlsafe_b64decode(token.encode()).decode().partition(":")
        if kind != "offset" or not value.isdigit():
            raise ValueError
        return int(value)
    except (ValueError, UnicodeDecodeError):
        raise ToolError("non_retryable", "invalid_cursor", "Pass next_cursor exactly as returned, or omit it.")


def list_departures(args: ListDeparturesArgs) -> dict[str, Any]:
    """Lecture 5.9: departures for the REQUESTED date (optionally one destination), shaped and paged.

    Rows: sim.departures_on(args.date.isoformat()), filtered by destination if given (case-insensitive),
    in timetable order. Page with shape_result(rows, ["flight_number", "destination", "scheduled_departure",
    "status"], PAGE_SIZE, decode_cursor(args.cursor)), then replace next_cursor with encode_cursor(next_cursor)
    (or None). The model only ever sees opaque page tokens.
    """
    day = args.date.isoformat()
    rows = [sim.as_dict(f) for f in sim.departures_on(day)
            if args.destination is None or f.destination == args.destination.upper()]
    page = shape_result(rows, ["flight_number", "destination", "scheduled_departure", "status"], PAGE_SIZE,
                        decode_cursor(args.cursor))
    page["next_cursor"] = encode_cursor(page["next_cursor"]) if page["next_cursor"] is not None else None
    return page


def get_flight_status(args: StatusArgs) -> dict[str, Any]:
    f = sim.find(args.flight_number, args.date.isoformat())
    if f is None:
        raise ToolError("non_retryable", "not_found", f"No flight {args.flight_number} on {args.date.isoformat()}.")
    return {k: v for k, v in sim.as_dict(f).items() if k not in ("origin", "operating_date")}


def make_catalog(gw: Gateways | None = None) -> tuple[list[ToolContract], Gateways]:
    gw = gw or Gateways()

    def notify_rules(args: NotifyArgs) -> None:
        if args.flight_number not in BOOKED:
            raise ToolError("non_retryable", "not_found", f"No bookings found for {args.flight_number}.")

    def notify(args: NotifyArgs) -> dict[str, Any]:
        gw.sent.append({"flight_number": args.flight_number, "message": args.message, "reason": args.reason})
        if gw.lose_next_response:
            gw.lose_next_response = False
            raise TimeoutError("gateway did not respond")
        return {"status": "sent", "recipients": BOOKED[args.flight_number]}

    def voucher(args: VoucherArgs) -> dict[str, Any]:
        return issue_voucher(args, gw)

    def voucher_rules(args: VoucherArgs) -> None:
        check_voucher(args)

    contracts = [
        ToolContract("list_departures",
                     "List departures from the hub for one date, optionally to one destination, five per page; pass next_cursor to get the next page. "
                     "Read-only. Returns flight number, destination, scheduled time and status; use "
                     "get_flight_status for gates, estimates and reasons.",
                     ListDeparturesArgs, list_departures, SideEffect.READ, errors=["invalid_arguments", "invalid_cursor"]),
        ToolContract("get_flight_status",
                     "Get status, estimated departure, gate and delay reason for one flight on one date. Read-only. "
                     "Returns not_found if the flight doesn't operate that day.",
                     StatusArgs, get_flight_status, SideEffect.READ, errors=["invalid_arguments", "not_found"]),
        ToolContract("notify_passengers",
                     "Send one SMS and email to every passenger booked on a flight. Irreversible: messages can't be "
                     "recalled. Within one disruption, repeating the same message for the same flight returns the "
                     "original result instead of sending again. Supports dry run.",
                     NotifyArgs, notify, SideEffect.IRREVERSIBLE_WRITE,
                     errors=["invalid_arguments", "not_found", "outcome_unknown", "idempotency_conflict"],
                     key_fields=["flight_number", "message"], supports_dry_run=True, precheck=notify_rules),
        ToolContract("issue_voucher",
                     f"Issue a meal or hotel voucher to one booking on a disrupted flight. Irreversible: issued "
                     f"vouchers can be spent at once. Above {VOUCHER_LIMIT_EUR} EUR, returns approval_required "
                     "and a person decides. One voucher per booking, flight and kind: a repeat returns the original "
                     "voucher. Supports dry run.",
                     VoucherArgs, voucher, SideEffect.IRREVERSIBLE_WRITE,
                     errors=["invalid_arguments", "not_found", "approval_required", "flight_not_disrupted",
                             "idempotency_conflict"],
                     key_fields=["booking_ref", "flight_number", "kind"], supports_dry_run=True, precheck=voucher_rules),
    ]
    return contracts, gw


def check_voucher(args: VoucherArgs) -> None:
    """Lecture 5.9: the rules for a voucher, checked before anything is issued (and in dry runs).

    - unknown flight on the simulator's date: ToolError("non_retryable", "not_found", ...)
    - flight status not DELAYED or CANCELLED: ToolError("non_retryable", "flight_not_disrupted", ...)
    - amount_eur > VOUCHER_LIMIT_EUR: ToolError("needs_human", "approval_required", ...)
    """
    f = sim.find(args.flight_number, sim.DATE)
    if f is None:
        raise ToolError("non_retryable", "not_found", f"No flight {args.flight_number} today.")
    if f.status not in ("DELAYED", "CANCELLED"):
        raise ToolError("non_retryable", "flight_not_disrupted", f"{args.flight_number} is {f.status.lower()}.")
    if args.amount_eur > VOUCHER_LIMIT_EUR:
        raise ToolError("needs_human", "approval_required",
                        f"{args.amount_eur} EUR is above the {VOUCHER_LIMIT_EUR} EUR limit; a duty manager must approve.")


def issue_voucher(args: VoucherArgs, gw: Gateways) -> dict[str, Any]:
    """Issue the voucher. Only reached after check_voucher passed (execute_write runs the precheck)."""
    gw.vouchers.append({"booking_ref": args.booking_ref, "flight_number": args.flight_number, "kind": args.kind,
                        "amount_eur": args.amount_eur})
    return {"status": "issued", "voucher_id": f"V-{args.booking_ref}-{args.kind}", "amount_eur": args.amount_eur}


def toolbox(scope: str, dry_run: bool = False, gw: Gateways | None = None):
    """A Section 4 ToolBox of the contracted tools, sharing one idempotency store. The scope is required:
    the run, ticket or disruption these actions belong to."""
    from ..agent.loop import ToolBox
    contracts, gw = make_catalog(gw)
    store = IdempotencyStore()
    box = ToolBox()
    for c in contracts:
        box.add(as_tool(c, store, scope, dry_run))
    return box, gw, store
