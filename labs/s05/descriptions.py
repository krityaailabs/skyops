"""Lecture 5.2 (demo): tool names, descriptions and schemas are prompts. Measure the effect on tool choice.

Two tool sets do the same thing. VAGUE has short names, one-line descriptions and undocumented fields.
PRECISE is the Section 5 catalog. Eight questions; for each, we record only the model's FIRST proposed
call and check the tool and its key argument. Run:  python -m labs.s05.descriptions

The recordings are scripted until the instructor re-records them live (`make synthetic-report`); tool
choice is a behaviour a live model does show, so re-recording will replace these numbers with real ones.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from skyops.agent.loop import SYSTEM, Tool, ToolBox
from skyops.model.client import ModelClient, get_client
from skyops.model.types import ChatRequest, Message
from skyops.tools.catalog import make_catalog
from skyops.tools.contracts import as_tool, estimate_tokens


class FlightsArgs(BaseModel):
    dest: str
    d: str


class StatusArgs(BaseModel):
    fn: str
    d: str


def vague_box() -> ToolBox:
    noop = lambda a: {}  # noqa: E731 - only the first proposed call is measured
    return ToolBox().add(Tool("flights", FlightsArgs, noop, "Gets flights.")).add(Tool("status", StatusArgs, noop, "Gets status."))


def precise_box() -> ToolBox:
    contracts, _ = make_catalog()
    box = ToolBox()
    for c in contracts[:2]:                  # the two read tools
        box.add(as_tool(c, None, scope="choice-demo"))
    return box


# (question, expected tool in the precise set, the same tool in the vague set, expected key argument value)
QUESTIONS = [
    ("Which flights leave for Singapore today?", "list_departures", "flights", "SIN"),
    ("Is ZZ214 delayed?", "get_flight_status", "status", "ZZ214"),
    ("What gate does the Sydney flight use?", "list_departures", "flights", "SYD"),
    ("What's the status of the 22:30 Tokyo departure?", "list_departures", "flights", "NRT"),
    ("Has ZZ101 left yet?", "get_flight_status", "status", "ZZ101"),
    ("Anything going to Paris tonight?", "list_departures", "flights", "CDG"),
    ("When does the Bengaluru flight leave?", "list_departures", "flights", "BLR"),
    ("Why is ZZ173 late?", "get_flight_status", "status", "ZZ173"),
]
KEY_ARGS = {"list_departures": "destination", "get_flight_status": "flight_number", "flights": "dest", "status": "fn"}


@dataclass
class Choice:
    question: str
    tool: str | None
    arg: str | None
    correct: bool


def first_choice(question: str, client: ModelClient, box: ToolBox, expected_tool: str, expected_arg: str) -> Choice:
    resp = client.complete(ChatRequest(model=client.model, system=SYSTEM, messages=[Message.user(question)],
                                       tools=box.specs(), max_tokens=512))
    if not resp.tool_calls:
        return Choice(question, None, None, False)
    call = resp.tool_calls[0]
    arg = str(call.input.get(KEY_ARGS.get(call.name, ""), ""))
    return Choice(question, call.name, arg, call.name == expected_tool and arg.upper() == expected_arg)


def measure(which: str) -> list[Choice]:
    box = vague_box() if which == "vague" else precise_box()
    out = []
    for i, (q, precise_tool, vague_tool, arg) in enumerate(QUESTIONS, 1):
        client = get_client(fixture=f"fixtures/s05/choice_{which}_{i:02d}.jsonl", match="strict")
        out.append(first_choice(q, client, box, vague_tool if which == "vague" else precise_tool, arg))
    return out


def description_tokens(box: ToolBox) -> int:
    return estimate_tokens([s.model_dump() for s in box.specs()])


def main() -> None:
    for which in ("vague", "precise"):
        res = measure(which)
        print(f"{which}: {sum(c.correct for c in res)}/{len(res)} first calls correct; "
              f"tool descriptions about {description_tokens(vague_box() if which == 'vague' else precise_box())} tokens")
        for c in res:
            if not c.correct:
                print(f"   wrong: {c.question!r} -> {c.tool}({c.arg!r})")


if __name__ == "__main__":
    main()
