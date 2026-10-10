"""The ten eval cases for Section 4 (lecture 4.7), and the scripted model turns that produce their replays.

Each case states what a correct run must show, in checks a machine can apply without a judge:
- answer_has: every group must match; a group is satisfied if ANY of its phrases appears (case-insensitive).
  Matching is by substring, so "not on time" contains "on time": positive phrases need forbid entries for
  their negations (see c07 and c09);
- tools: tools that must be called; forbid: phrases that must NOT appear in the answer;
- max_steps: the run must answer within this many model calls.
The ground truth is the simulator (skyops/sim/flights.py), not the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

D = "2026-10-09"


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    answer_has: list[list[str]]
    tools: list[str]
    forbid: list[str] = field(default_factory=list)
    max_steps: int = 4


CASES = [
    Case("c01_sin_evening", "Is the evening flight to Singapore on time, and which gate is it leaving from?",
         [["ZZ214"], ["delayed", "not on time", "late"], ["20:15"], ["B7"]], ["find_departures", "get_flight_status"]),
    Case("c02_cancelled", "What's the status of ZZ247 tonight?",
         [["cancel"]], ["get_flight_status"], forbid=["gate B", "gate A", "gate C"]),
    Case("c03_gate", "Which gate does ZZ142 leave from?", [["C2"]], ["get_flight_status"]),
    Case("c04_unknown", "Is ZZ999 operating today?", [["couldn't find", "could not find", "no flight", "not find"]],
         ["get_flight_status"], forbid=["on time", "delayed"]),
    Case("c05_bad_args_recover", "When does the Bengaluru flight leave?",
         [["ZZ173"], ["18:10"]], ["find_departures"]),
    Case("c06_departed", "Has the morning London flight left?", [["ZZ101"], ["departed", "has left", "already left"]],
         ["find_departures"], forbid=["hasn't", "has not", "not yet"]),
    Case("c07_sydney", "Any flights to Sydney tonight, and are they on time?", [["ZZ230"], ["21:55"], ["on time", "scheduled"]],
         ["find_departures"], forbid=["not on time", "delayed", "cancel"]),
    Case("c08_reason", "Why was the Frankfurt flight late?", [["ZZ118"], ["late inbound"]], ["find_departures", "get_flight_status"]),
    Case("c09_paris", "Is ZZ261 to Paris on time?", [["on time", "scheduled"], ["23:05"]], ["get_flight_status"],
         forbid=["not on time", "delayed", "cancel"]),
    Case("c10_status_slip", "Is the Bengaluru flight ZZ173 on time?", [["delayed", "not on time"], ["18:10"]],
         ["get_flight_status"], forbid=["is on time"]),
]

# Scripted model turns for the replay fixtures: ("tool", name, args, preface) or ("text", answer).
# All of these are SCRIPTED (synthetic) and stay so after the other fixtures are re-recorded: c10's final
# answer contradicts the tool data on purpose. Models do make this kind of slip when a field they were given
# ("scheduled_departure") looks like an answer, but not on demand; scripting it makes it reproducible.
SCRIPTS = {
    "c01_sin_evening": [("tool", "find_departures", {"destination": "SIN", "date": D}, ""),
                        ("tool", "get_flight_status", {"flight_number": "ZZ214", "date": D}, "ZZ214 at 19:40 is the evening one."),
                        ("text", "The evening flight to Singapore, ZZ214, is delayed: scheduled 19:40, now estimated 20:15 "
                                 "because of a late inbound aircraft. It leaves from gate B7.")],
    "c02_cancelled": [("tool", "get_flight_status", {"flight_number": "ZZ247", "date": D}, ""),
                      ("text", "ZZ247 to Tokyo Narita tonight is cancelled, because of an aircraft technical inspection.")],
    "c03_gate": [("tool", "get_flight_status", {"flight_number": "ZZ142", "date": D}, ""),
                 ("text", "ZZ142 to New York JFK leaves from gate C2. It is boarding now.")],
    "c04_unknown": [("tool", "get_flight_status", {"flight_number": "ZZ999", "date": D}, ""),
                    ("text", "I couldn't find a flight ZZ999 operating today, 9 October. Please check the flight number.")],
    "c05_bad_args_recover": [("tool", "find_departures", {"destination": "Bengaluru", "date": D}, ""),
                             ("tool", "find_departures", {"destination": "BLR", "date": D},
                              "The destination must be an airport code. Bengaluru is BLR."),
                             ("text", "The Bengaluru flight, ZZ173, was scheduled for 16:40 and is now expected to leave at 18:10.")],
    "c06_departed": [("tool", "find_departures", {"destination": "LHR", "date": D}, ""),
                     ("text", "Yes. The morning London flight, ZZ101, departed at 07:10.")],
    "c07_sydney": [("tool", "find_departures", {"destination": "SYD", "date": D}, ""),
                   ("text", "There is one flight to Sydney tonight, ZZ230 at 21:55, and it is scheduled to leave on time.")],
    "c08_reason": [("tool", "find_departures", {"destination": "FRA", "date": D}, ""),
                   ("tool", "get_flight_status", {"flight_number": "ZZ118", "date": D}, ""),
                   ("text", "The Frankfurt flight, ZZ118, left late because of a late inbound aircraft. "
                            "It was scheduled for 09:45 and departed at 10:30.")],
    "c09_paris": [("tool", "get_flight_status", {"flight_number": "ZZ261", "date": D}, ""),
                  ("text", "ZZ261 to Paris is scheduled to leave on time at 23:05 from gate A1.")],
    "c10_status_slip": [("tool", "get_flight_status", {"flight_number": "ZZ173", "date": D}, ""),
                        ("text", "ZZ173 to Bengaluru is on time, departing at 16:40 from gate B9.")],
}
