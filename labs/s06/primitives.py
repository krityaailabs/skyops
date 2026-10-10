"""Lectures 6.1 and 6.2: the integration arithmetic, and ten SkyOps capabilities sorted into MCP primitives.

    python -m labs.s06.primitives
"""
from __future__ import annotations

from typing import Literal

Primitive = Literal["tool", "resource", "prompt"]


def connectors(hosts: int, systems: int) -> tuple[int, int]:
    """Bespoke integrations (every host to every system) versus one protocol (each side implements it once)."""
    return hosts * systems, hosts + systems


# Who decides that it's used: the model (tool), the application (resource), or the user (prompt).
CAPABILITIES: list[tuple[str, Primitive, str]] = [
    ("Look up one flight's status", "tool", "the model decides when it needs it, mid-task"),
    ("Send a message to a flight's passengers", "tool", "an action with a side effect; the model proposes it"),
    ("Issue a meal or hotel voucher", "tool", "an action with a side effect and a limit"),
    ("Search departures for a date", "tool", "a parameterised query the model chooses to run"),
    ("Read one booking by reference", "resource", "data with an address; the application attaches it as context"),
    ("The disruption-handling policy document", "resource", "reference text the application loads, not an action"),
    ("Today's departure board, as a file", "resource", "data with a URI that changes over the day"),
    ("Draft a disruption brief for a flight", "prompt", "a template the duty manager picks from a menu"),
    ("Write a passenger apology in the house style", "prompt", "a user-chosen template with arguments"),
    ("Hand over the shift: summarise open issues", "prompt", "a workflow the user starts on purpose"),
]


def main() -> None:
    for h, s in ((4, 6), (10, 30)):
        bespoke, shared = connectors(h, s)
        print(f"{h} hosts x {s} systems: {bespoke} bespoke integrations, or {shared} protocol implementations")
    print()
    for cap, prim, why in CAPABILITIES:
        print(f"{prim:<9} {cap:<46} {why}")
    counts = {p: sum(1 for _, q, _ in CAPABILITIES if q == p) for p in ("tool", "resource", "prompt")}
    print(f"{counts['tool']} tools, {counts['resource']} resources, {counts['prompt']} prompts")


if __name__ == "__main__":
    main()
