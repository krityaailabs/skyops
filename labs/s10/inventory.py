"""Lecture 10.1: what this repository evaluates today, and what it doesn't.

Counts come from the code and data, not from memory. The last part scores Section 5's disruption run (lecture
5.9, the scripted run where the model sends the same notice twice) with Section 10's checks. Run at tag s10:
    python -m labs.s10.inventory
"""
from __future__ import annotations

import logging


def figures() -> dict:
    logging.disable(logging.CRITICAL)
    from labs.s04.cases import CASES
    from labs.s05 import descriptions
    from labs.s05.disruption import QUESTION, run
    from skyops.evals.checks import check_outcome, check_tool_calls, check_trajectory
    from skyops.evals.dataset import Expect, GoldenCase, Label, NotifyExpect
    from skyops.evals.runner import Attempt
    from skyops.retrieval.evals import load_queries
    r, gw = run()
    case = GoldenCase(id="e99_section5_run", category="edge", source="synthetic", flight="ZZ247", notice=QUESTION,
                      expect=Expect(notify=NotifyExpect(flight="ZZ247", reason="cancellation"),
                                    answer_has=[["ZZ247"], ["approval", "approve"]], writes_only_for="ZZ247"),
                      label=Label(by="course author", on="2026-10-17", why="Section 5's run, scored after the fact."))
    att = Attempt(case.id, 1, r, list(gw.sent), list(gw.vouchers))
    results = [check_outcome(case, att), check_tool_calls(case, att), check_trajectory(case, att)]
    return {"s04_cases": len(CASES), "s05_questions": len(descriptions.QUESTIONS), "s09_queries": len(load_queries()),
            "s05_proposed_notify": sum(n == "notify_passengers" for n, _ in r.tool_calls), "s05_sent": len(gw.sent),
            "results": results}


def main() -> None:
    f = figures()
    print("what SkyOps evaluates today")
    print(f"  4.7  agent answers      {f['s04_cases']:>3} cases   final answer phrases + tools used, one run each")
    print(f"  5.2  tool choice        {f['s05_questions']:>3} questions first proposed call only, two tool sets")
    print(f"  9.6  retrieval          {f['s09_queries']:>3} queries  recall at k and MRR, no answers")
    print("not evaluated: the disruption handler's writes (notify_passengers, issue_voucher), its vouchers,")
    print("  its passenger messages, and how often any of it works across repeated runs")
    print(f"Section 5's disruption run: {f['s05_proposed_notify']} notify proposals, {f['s05_sent']} message sent "
          "(the idempotency key replayed the second)")
    for res in f["results"]:
        print(f"  {res.name:11} {'PASS' if res.passed else 'FAIL'}  {'; '.join(res.failures)}")


if __name__ == "__main__":
    main()
