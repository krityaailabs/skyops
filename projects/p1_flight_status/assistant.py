"""Project 1 (beginner): the flight-status assistant. Reference solution; see BRIEF.md.

The design rule this project teaches: the model finds the flight; your code states the facts.
The agent loop lets the model choose tools and arguments. The final status, times and gate are then
read from the simulator by your code and written from a template, so a model slip like the one in
eval case c10 ("ZZ173 is on time") can never reach the user.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from skyops.agent.loop import Budgets, RunResult, ToolBox, run_agent
from skyops.model.client import ModelClient
from skyops.sim import flights as sim
from skyops.model.errors import LabNotDone

NAMES = {"LHR": "London Heathrow", "FRA": "Frankfurt", "JFK": "New York JFK", "DXB": "Dubai", "BLR": "Bengaluru",
         "SIN": "Singapore", "SYD": "Sydney", "NRT": "Tokyo Narita", "CDG": "Paris"}


@dataclass(frozen=True)
class FlightAnswer:
    text: str                       # what the user sees
    flight_number: str | None       # the flight the facts came from, or None
    status: str | None              # simulator status, never the model's wording
    grounded: bool                  # True when every fact in text came from tool data
    stop: str                       # how the agent run ended


def describe(f: sim.FlightRecord) -> str:
    """One sentence from a flight record. Every flight fact comes from the record; the city name comes
    from the NAMES table in this file. Free-text fields (delay_reason) are length-limited."""
    where = NAMES.get(f.destination, f.destination)
    reason = (f.delay_reason or "")[:80].lower()
    because = f" Reason: {reason}." if reason else ""
    gate = f" from gate {f.gate}" if f.gate else " (gate not yet assigned)"
    if f.status == "CANCELLED":
        return f"{f.flight_number} to {where} is cancelled.{because}"
    if f.status == "DEPARTED":
        if f.estimated_departure != f.scheduled_departure:
            return (f"{f.flight_number} to {where} departed late, at {f.estimated_departure} instead of "
                    f"{f.scheduled_departure}.{because}")
        return f"{f.flight_number} to {where} departed on time, at {f.estimated_departure}."
    if f.status == "DELAYED":
        return (f"{f.flight_number} to {where} is delayed: scheduled {f.scheduled_departure}, now "
                f"estimated {f.estimated_departure}{gate}.{because}")
    if f.status == "BOARDING":
        return f"{f.flight_number} to {where} is boarding now{gate}, departing {f.scheduled_departure}."
    return f"{f.flight_number} to {where} is scheduled to leave on time at {f.scheduled_departure}{gate}."


def chosen_flight(run: RunResult) -> tuple[str | None, str | None]:
    """Which flight did the model's tool calls settle on? Returns (flight_number, date) or (None, None).

    The most recent lookup decides. Walk the calls backwards and stop at the first lookup:
    - get_flight_status -> that flight;
    - find_departures with exactly one departure that day -> that flight;
    - find_departures with none or several -> no choice (None, None).
    """
    raise LabNotDone("Lecture 4.8: implement chosen_flight(). Hints: labs/s04/README.md, lecture 4.8.")


# Sized from the Section 4 eval (lecture 4.6): about twice the largest healthy run.
BUDGETS = Budgets(max_steps=6, max_tokens=6_000, max_seconds=60.0)


def ambiguous(run: RunResult) -> list[sim.FlightRecord]:
    """The flights the model found but never chose between: the last find_departures with several results."""
    for name, args in reversed(run.tool_calls):
        if name == "find_departures" and isinstance(args.get("destination"), str):
            rows = sim.departures_to(args["destination"], args.get("date", sim.DATE))
            return rows if len(rows) > 1 else []
    return []


def answer(question: str, client: ModelClient, budgets: Budgets = BUDGETS) -> FlightAnswer:
    """Run the agent with the read-only flight tools, then build the answer from the simulator.

    - If the run didn't end with "answered", return its stop message, grounded=False.
    - If no flight was chosen but several were found, list them from the records and ask which one,
      grounded=False ("I found 2 flights to Singapore: ZZ188 at 08:15 and ZZ214 at 19:40. Which one do you mean?").
    - If no flight was chosen and none were found, return the model's own text, grounded=False.
    - If the chosen flight doesn't exist, say "I couldn't find flight <number> on <date>.", grounded=True.
    - Otherwise return describe(record), with flight_number and status from the record, grounded=True.
    """
    raise LabNotDone("Lecture 4.8: implement answer(). Hints: labs/s04/README.md, lecture 4.8.")


def main() -> None:
    from labs.s04.cases import CASES
    from skyops.model.client import get_client
    for c in CASES:
        a = answer(c.question, get_client(fixture=f"fixtures/s04/{c.id}.jsonl", match="strict"))
        print(json.dumps({"case": c.id, "answer": a.text, "grounded": a.grounded}))


if __name__ == "__main__":
    main()
