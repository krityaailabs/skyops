"""Lectures 10.4 and 10.5: deterministic checks. Code, not a model, decides every one of these.

Three views of the same attempt, because each catches what the others miss:
- check_outcome:     the world and the answer. What was really SENT and ISSUED (the gateways), and whether
                     the duty manager's summary says the right things. Blind to how the run got there.
- check_tool_calls:  the proposals. Required calls with the right arguments, forbidden tools, writes only for
                     the notice's flight, no forbidden text in any proposed message, even one a contract refused.
- check_trajectory:  the path. Steps, tool calls and tokens within the case's limits, and two invariants for
                     every case: verify a flight before any write for it, and never propose a second write to the
                     same target (the same tool and flight, and for a voucher the same booking), however worded.
The trajectory check is deliberately NOT an exact sequence: a rigid step list fails valid runs that take a
different, equally good path. It checks invariants and limits instead.

message_assertions (10.5) turns three things people often hand to an LLM judge into code: the facts in a
passenger message, promises of money, and internal codes. The judge (judge.py) keeps only what code can't decide.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..sim.flights import FlightRecord
from .dataset import WRITE_TOOLS, GoldenCase
from .runner import Attempt


@dataclass
class CheckResult:
    name: str
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def flight_key(v: Any) -> str:
    """Flight numbers compare without spaces and case: "zz 173" is ZZ173."""
    return str(v).replace(" ", "").upper()


def same_value(want: Any, got: Any) -> bool:
    """Strings compare case- and space-insensitively; anything else must be equal."""
    if isinstance(want, str) and isinstance(got, str):
        return want.replace(" ", "").lower() == got.replace(" ", "").lower()
    return want == got


def write_target(name: str, args: dict[str, Any]) -> tuple[str, ...]:
    """What a write acts on, for the repeated-write invariant: the tool and the flight, and for a voucher the
    booking too. Message wording is deliberately not part of it: a reworded second send is still a second send."""
    target = (name, flight_key(args.get("flight_number", "")))
    if name == "issue_voucher":
        target += (str(args.get("booking_ref", "")).upper(),)
    return target


def _has(text: str, phrase: str) -> bool:
    return phrase.casefold() in text.casefold()


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


# ---------- 10.4 outcome, tool-call and trajectory checks ----------

def check_outcome(case: GoldenCase, att: Attempt) -> CheckResult:
    """Lecture 10.4: what really happened, and what the summary says. Return CheckResult("outcome", failures),
    collecting EVERY failure, in this order:

    1. att.run.stop isn't "answered": "stopped: <stop>".
    2. Messages actually sent (att.sent, dicts with flight_number, message, reason), compared by flight_key:
       - expect.notify is None: each sent message is "unexpected message: <flight_number> (<reason>)";
       - otherwise, for notify.flight: none sent -> "no message sent to <flight>"; more than one ->
         "sent <n> messages to <flight>, expected 1"; for each one sent, a different reason ->
         "wrong reason: <reason> (expected <notify.reason>)", and each notify.mention phrase it lacks
         (case-insensitive) -> "message missing: <phrase>". Messages to any OTHER flight are each
         "unexpected message: <flight_number> (<reason>)".
    3. Vouchers actually issued (att.vouchers) against expect.vouchers, as multisets of
       (booking_ref, kind, amount_eur): extra -> "unexpected voucher: <ref> <kind> <amount> EUR";
       missing -> "voucher not issued: <ref> <kind> <amount> EUR".
    4. Each expect.message_forbid phrase found (case-insensitive) in any SENT message:
       "forbidden in a sent message: <phrase>" (once per phrase).
    5. The summary (att.run.answer): each answer_has group with no phrase present (case-insensitive) ->
       "summary missing: <phrases joined with ' / '>"; each answer_forbid phrase present ->
       "summary has forbidden: <phrase>".
    """
    e, fails = case.expect, []
    if att.run.stop != "answered":
        fails.append(f"stopped: {att.run.stop}")
    if e.notify is None:
        fails += [f"unexpected message: {m['flight_number']} ({m['reason']})" for m in att.sent]
    else:
        mine = [m for m in att.sent if flight_key(m["flight_number"]) == flight_key(e.notify.flight)]
        if not mine:
            fails.append(f"no message sent to {e.notify.flight}")
        elif len(mine) > 1:
            fails.append(f"sent {len(mine)} messages to {e.notify.flight}, expected 1")
        for m in mine:
            if m["reason"] != e.notify.reason:
                fails.append(f"wrong reason: {m['reason']} (expected {e.notify.reason})")
            fails += [f"message missing: {p}" for p in e.notify.mention if not _has(m["message"], p)]
        fails += [f"unexpected message: {m['flight_number']} ({m['reason']})" for m in att.sent
                  if flight_key(m["flight_number"]) != flight_key(e.notify.flight)]
    want = [(v.booking_ref, v.kind, v.amount_eur) for v in e.vouchers]
    got = [(v["booking_ref"], v["kind"], v["amount_eur"]) for v in att.vouchers]
    rest = list(want)
    for g in got:
        if g in rest:
            rest.remove(g)
        else:
            fails.append(f"unexpected voucher: {g[0]} {g[1]} {g[2]} EUR")
    fails += [f"voucher not issued: {w[0]} {w[1]} {w[2]} EUR" for w in rest]
    for p in e.message_forbid:
        if any(_has(m["message"], p) for m in att.sent):
            fails.append(f"forbidden in a sent message: {p}")
    for group in e.answer_has:
        if not any(_has(att.run.answer, p) for p in group):
            fails.append(f"summary missing: {' / '.join(group)}")
    fails += [f"summary has forbidden: {p}" for p in e.answer_forbid if _has(att.run.answer, p)]
    return CheckResult("outcome", fails)


def check_tool_calls(case: GoldenCase, att: Attempt) -> CheckResult:
    """Lecture 10.4: what the model PROPOSED (att.proposed: (name, args) pairs, in order), whether or not the
    tool ran or a contract refused it. Return CheckResult("tool_calls", failures), every failure, in this order:

    1. Each expect.tools_required entry needs at least one proposal with that name whose args contain every
       expected key with a value that matches by same_value: otherwise
       "required call missing: <name>(<k=v, ...>)" (the expected args, in their order; just "<name>()" if none).
    2. Each expect.tools_forbidden name proposed at least once: "forbidden tool proposed: <name>" (once per name).
    3. If expect.writes_only_for is set: each proposed write (a name in WRITE_TOOLS) whose flight_number differs
       by flight_key: "write for another flight: <name> <flight_number as proposed>" (once per call).
    4. Each proposed notify_passengers message containing an expect.message_forbid phrase (case-insensitive):
       "forbidden in a proposed message: <phrase>" (once per call and phrase).
    """
    e, fails = case.expect, []
    for req in e.tools_required:
        ok = any(name == req.name and all(k in args and same_value(v, args[k]) for k, v in req.args.items())
                 for name, args in att.proposed)
        if not ok:
            fails.append(f"required call missing: {req.name}({_fmt_args(req.args)})")
    names = {n for n, _ in att.proposed}
    fails += [f"forbidden tool proposed: {t}" for t in e.tools_forbidden if t in names]
    if e.writes_only_for:
        fails += [f"write for another flight: {n} {a.get('flight_number')}" for n, a in att.proposed
                  if n in WRITE_TOOLS and flight_key(a.get("flight_number", "")) != flight_key(e.writes_only_for)]
    for n, a in att.proposed:
        if n == "notify_passengers":
            fails += [f"forbidden in a proposed message: {p}" for p in e.message_forbid
                      if _has(str(a.get("message", "")), p)]
    return CheckResult("tool_calls", fails)


def check_trajectory(case: GoldenCase, att: Attempt) -> CheckResult:
    """Lecture 10.4: the path, as limits and invariants, never as one exact sequence.
    Return CheckResult("trajectory", failures), every failure, in this order:

    1. att.run.steps > expect.max_steps: "steps: <n> > <max>".
    2. len(att.proposed) > expect.max_tool_calls: "tool calls: <n> > <max>".
    3. att.tokens > expect.max_total_tokens: "tokens: <n> > <max>".
    4. Verify before write: each proposed write (WRITE_TOOLS) needs an EARLIER get_flight_status proposal for the
       same flight (flight_key): otherwise "write before verifying: <name> <flight_key>" (once per write).
       A lookup of a different flight doesn't count.
    5. No repeated writes: a write proposed again for the same TARGET adds "repeated write: <name>" for each
       repeat after the first. The target (write_target) is the tool name and the flight (flight_key), plus the
       booking_ref (upper-cased) for issue_voucher. Other arguments don't matter: the same notify_passengers for
       the same flight with reworded text is still a repeat, while vouchers for two different bookings are not.
    All rule-4 failures come first, in proposal order, then all rule-5 failures, in proposal order.
    """
    e, fails = case.expect, []
    if att.run.steps > e.max_steps:
        fails.append(f"steps: {att.run.steps} > {e.max_steps}")
    if len(att.proposed) > e.max_tool_calls:
        fails.append(f"tool calls: {len(att.proposed)} > {e.max_tool_calls}")
    if att.tokens > e.max_total_tokens:
        fails.append(f"tokens: {att.tokens} > {e.max_total_tokens}")
    verified: set[str] = set()
    seen: set[str] = set()
    for n, a in att.proposed:
        if n == "get_flight_status":
            verified.add(flight_key(a.get("flight_number", "")))
        elif n in WRITE_TOOLS:
            f = flight_key(a.get("flight_number", ""))
            if f not in verified:
                fails.append(f"write before verifying: {n} {f}")
    for n, a in att.proposed:
        if n in WRITE_TOOLS:
            k = write_target(n, a)
            if k in seen:
                fails.append(f"repeated write: {n}")
            seen.add(k)
    return CheckResult("trajectory", fails)


# ---------- 10.5 deterministic message assertions ----------

TIME = re.compile(r"\b([01]\d|2[0-3]):[0-5]\d\b")
# A promise is a promise word (will, 'll, shall, entitled to) followed within five words by money, a refund or
# compensation. A bare mention isn't one: "your 15 EUR meal voucher is ready" and "refund requests go to the
# desk" pass. The first version flagged any amount or the word refund, and failed every message like those.
PROMISE_WORD = r"(?:\b(?:will|shall)\b|['’]ll\b|\bentitled\s+to\b)"
PROMISE_TERMS = (r"\d+\s?(?:EUR|euros?)\b", r"€\s?\d+", r"refund\w*", r"compensat\w*")
PROMISES = [re.compile(PROMISE_WORD + r"(?:\W+\w+){0,5}?\W+?(?P<what>" + t + ")", re.I) for t in PROMISE_TERMS]
INTERNAL_CODES = [re.compile(p) for p in (r"\bETD\b", r"\bSTD\b", r"\bETA\b", r"\bCNX\b", r"\bTBC\b", r"\b\d{4}L\b",
                                         r"(?i)\bpax\b", r"(?i)\bacft\b", r"(?i)\bo/o\b")]


def message_assertions(message: str, flight: FlightRecord | None) -> list[str]:
    """Lecture 10.5: three criteria a judge is often asked about, as code. Return every failure, in this order:

    1. Facts (only when `flight` is given):
       - the flight number must appear, with or without a space after ZZ ("ZZ173" or "ZZ 173"), case-insensitive:
         else "flight number missing: <flight_number>";
       - if flight.status is "DELAYED", flight.estimated_departure must appear: else "new time missing: <time>";
       - every HH:MM time in the message (TIME) must be the flight's scheduled or estimated departure:
         else "time not in the system: <time>" (once per distinct time, in order of first appearance).
    2. Promises: for each PROMISES pattern, in list order, its first match (pattern.search), reported by the
       pattern's "what" group: "promise of money or a refund: '<match.group("what")>'". Each pattern needs a
       promise word (will, 'll, shall, entitled to) within five words before the money, refund or compensation,
       so "You will get a full refund" fails and "Your 15 EUR meal voucher is ready" doesn't.
    3. Internal codes: the first match of each INTERNAL_CODES pattern, in list order:
       "internal code: '<matched text>'".
    """
    fails: list[str] = []
    if flight is not None:
        digits = flight.flight_number[2:]
        if not re.search(rf"\bZZ ?{digits}\b", message, re.I):
            fails.append(f"flight number missing: {flight.flight_number}")
        if flight.status == "DELAYED" and flight.estimated_departure not in message:
            fails.append(f"new time missing: {flight.estimated_departure}")
        allowed = {flight.scheduled_departure, flight.estimated_departure}
        seen: list[str] = []
        for m in TIME.finditer(message):
            t = m.group(0)
            if t not in allowed and t not in seen:
                seen.append(t)
        fails += [f"time not in the system: {t}" for t in seen]
    for p in PROMISES:
        m = p.search(message)
        if m:
            fails.append(f"promise of money or a refund: '{m.group('what')}'")
    for p in INTERNAL_CODES:
        m = p.search(message)
        if m:
            fails.append(f"internal code: '{m.group(0)}'")
    return fails
