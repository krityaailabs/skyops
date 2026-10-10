"""Lectures 4.2, 4.3 and 4.5: run one question through the Section 4 loop and print its trace.

    python -m labs.s04.demo                 the evening Singapore flight (c01)
    python -m labs.s04.demo c05_bad_args_recover   the model passes a city name; validation sends it back
"""
from __future__ import annotations

import sys

from skyops.agent.loop import ToolBox, run_agent
from skyops.model.client import get_client

from .cases import CASES


def main(argv: list[str] | None = None) -> None:
    case_id = (argv or sys.argv[1:] or ["c01_sin_evening"])[0]
    case = next(c for c in CASES if c.id == case_id)
    r = run_agent(case.question, get_client(fixture=f"fixtures/s04/{case.id}.jsonl", match="strict"),
                  ToolBox.flight_tools())
    print(r.trace.render())
    print(f"stop: {r.stop}  steps: {r.steps}")


if __name__ == "__main__":
    main()
