"""Lecture 10.4: outcome, tool-call and trajectory checks on the same attempts, and what each one catches.

    python -m labs.s10.checks_demo                  five attempts side by side, then the whole suite
    python -m labs.s10.checks_demo --outcome-only   break it: score every attempt on the outcome check alone
The model turns are scripted replays; the loop, the tools, the gateways and the checks run for real.
"""
from __future__ import annotations

import argparse
import logging

SHOWN = [("n01_delay_blr", 1), ("n02_cancel_nrt", 3), ("n06_delay_hotel", 3), ("a01_injection_voucher", 2),
         ("e01_gate_mismatch", 2)]


def suite() -> dict:
    """Every recorded attempt, scored by each check; counts used by the narration."""
    logging.disable(logging.CRITICAL)
    from skyops.evals.checks import check_outcome, check_tool_calls, check_trajectory
    from skyops.evals.dataset import load_golden
    from skyops.evals.runner import run_case
    rows = []
    for c in load_golden():
        for a in run_case(c):
            rows.append((c.id, a.attempt, check_outcome(c, a), check_tool_calls(c, a), check_trajectory(c, a)))
    only_outcome = [r for r in rows if r[2].passed and not (r[3].passed and r[4].passed)]
    return {"rows": rows, "attempts": len(rows), "outcome_pass": sum(r[2].passed for r in rows),
            "all_pass": sum(r[2].passed and r[3].passed and r[4].passed for r in rows),
            "hidden_by_outcome": [(r[0], r[1]) for r in only_outcome]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome-only", action="store_true")
    a = ap.parse_args(argv)
    s = suite()
    by = {(r[0], r[1]): r for r in s["rows"]}
    if not a.outcome_only:
        print(f"{'case / attempt':28} {'outcome':8} {'tools':8} {'path':8} first failure")
        for key in SHOWN:
            _, _, o, t, p = by[key]
            first = next((f"{c.name}: {c.failures[0]}" for c in (o, t, p) if c.failures), "-")
            print(f"{key[0] + ' a' + str(key[1]):28} {'pass' if o.passed else 'FAIL':8} {'pass' if t.passed else 'FAIL':8} "
                  f"{'pass' if p.passed else 'FAIL':8} {first}")
        print(f"suite: {s['attempts']} attempts; outcome passes {s['outcome_pass']}; all three pass {s['all_pass']}")
    else:
        print(f"outcome check only: {s['outcome_pass']} of {s['attempts']} attempts pass "
              f"(all three checks: {s['all_pass']})")
        print("green on outcome, red on tool calls or path:")
        for cid, n in s["hidden_by_outcome"]:
            _, _, _, t, p = by[(cid, n)]
            print(f"  {cid} a{n}: {'; '.join(t.failures + p.failures)}")


if __name__ == "__main__":
    main()
