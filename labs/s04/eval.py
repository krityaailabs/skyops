"""Lab 4.7: your first ten-case eval. Pass/fail checks on the answer, tool-call assertions, a printed score.

Run:  python -m labs.s04.eval
Each case replays its own recorded conversation (fixtures/s04/<case>.jsonl) through the real loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from skyops.agent.loop import RunResult, ToolBox, run_agent
from skyops.cost import per_thousand
from skyops.model.client import get_client

from .cases import CASES, Case
from skyops.model.errors import LabNotDone


@dataclass
class CaseResult:
    id: str
    passed: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    results: list[CaseResult]
    runs: list[RunResult]

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def score(self) -> float:
        return self.passed / len(self.results)


def check_case(case: Case, run: RunResult) -> CaseResult:
    """Apply the case's checks to one run. Collect every failure, not just the first: a failing case
    should tell you everything that's wrong with it.

    - the run must have stopped with "answered", within case.max_steps model calls (use run.steps);
    - every group in answer_has needs at least one of its phrases in the answer (case-insensitive);
    - every tool in case.tools must appear in run.tools_used;
    - no phrase in case.forbid may appear in the answer (case-insensitive).
    Failure messages: "stopped: <stop>", "steps: <n> > <max>", "missing: <group joined by ' / '>",
    "tool not called: <name>", "forbidden: <phrase>".
    """
    raise LabNotDone("Lecture 4.7: implement check_case(). Hints: labs/s04/README.md, lecture 4.7.")


def run_eval(cases: list[Case] = CASES, fixture_dir: str = "fixtures/s04") -> EvalReport:
    """Run every case through the real loop against its replay, check it, and collect the report."""
    raise LabNotDone("Lecture 4.7: implement run_eval(). Hints: labs/s04/README.md, lecture 4.7.")


def main() -> None:
    rep = run_eval()
    for r, run in zip(rep.results, rep.runs):
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {r.id:22} steps={run.steps}  {'; '.join(r.failures)}")
    cost = sum(run.trace.cost for run in rep.runs)
    print(f"score: {rep.passed}/{len(rep.results)} ({rep.score:.0%})   eval cost: "
          f"{per_thousand(cost):.2f} USD per 1,000 eval runs (example prices)")


if __name__ == "__main__":
    main()
