"""Lab 4.6: three ways an agent runs away, each reproduced from a recording, detected and capped.

    repeat     the model asks for the same lookup again and again        -> repeated_call guard
    sweep      the model checks every flight one by one; context grows   -> token budget
    slow       a status tool takes 25 simulated seconds per call           -> time budget

For each, the "uncapped" run has only a loose step budget of 12 (twice the quick win's) and no runaway guard;
the "capped" run uses budgets sized from the eval. Run:  python -m labs.s04.runaway
"""
from __future__ import annotations

from dataclasses import dataclass

from skyops.agent.loop import Budgets, RunResult, Tool, ToolBox, run_agent
from skyops.cost import per_thousand
from skyops.model.client import get_client

QUESTIONS = {
    "repeat": "Is the evening flight to Singapore on time?",
    "sweep": "Give me the status of every departure today.",
    "slow": "Which evening departures are running late?",
}
UNCAPPED = Budgets(max_steps=12, max_tokens=10**9, max_seconds=10**9, max_repeats=10**9)
# Sized from the eval (lecture 4.7): the largest healthy run is under 3,000 tokens and 3 model calls,
# so about twice that leaves room for a retry without letting a runaway get far.
CAPPED = Budgets(max_steps=8, max_tokens=6_000, max_seconds=60.0, max_repeats=3)
SLOW_S = 25.0


class SimTime:
    """A clock that only moves when a slow tool says so: 25 'seconds' cost nothing in a test."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def toolbox(name: str, clock: SimTime) -> ToolBox:
    box = ToolBox.flight_tools()
    if name == "slow":
        t = box.tools["get_flight_status"]

        def slow(args, _fn=t.fn):
            clock.now += SLOW_S
            return _fn(args)
        box.add(Tool(t.name, t.args_model, slow, t.description))
    return box


@dataclass
class Pair:
    name: str
    uncapped: RunResult
    capped: RunResult
    uncapped_s: float = 0.0       # simulated seconds when each run ended
    capped_s: float = 0.0


def run_pair(name: str) -> Pair:
    out, secs = [], []
    for b in (UNCAPPED, CAPPED):
        clock = SimTime()
        client = get_client(fixture=f"fixtures/s04/runaway_{name}.jsonl", match="sequence")
        out.append(run_agent(QUESTIONS[name], client, toolbox(name, clock), budgets=b, clock=clock))
        secs.append(clock.now)
    return Pair(name, *out, *secs)


def main() -> None:
    print(f"{'mode':7} {'run':9} {'stop':14} {'calls':>5} {'tokens':>7} {'USD/1k runs':>11}")
    for name in QUESTIONS:
        p = run_pair(name)
        for label, r in (("uncapped", p.uncapped), ("capped", p.capped)):
            s = r.trace.summary()
            print(f"{name:7} {label:9} {r.stop:14} {s['model_calls']:>5} {s['total_tokens']:>7} "
                  f"{per_thousand(r.trace.cost):>11.2f}")


if __name__ == "__main__":
    main()
