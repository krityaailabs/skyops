"""Lecture 10.2: eval-driven development. Five cases written BEFORE the mismatch rule existed (tag spec:mismatch),
run against the handler without the rule (v1) and with it (v2).

    python -m labs.s10.edd
Both handlers' model turns are scripted replays. v1 replays fixtures/s10/edd_v1/<case>/a1.jsonl; v2 replays ONE
recorded attempt (attempt 3) of the golden runs. One run is a demonstration, not a measurement: lecture 10.7
runs every case five times.
"""
from __future__ import annotations

import logging
from pathlib import Path

V2_ATTEMPT = 3


def results() -> dict:
    logging.disable(logging.CRITICAL)
    from skyops.evals.checks import check_outcome, check_tool_calls, check_trajectory
    from skyops.evals.dataset import load_golden
    from skyops.evals.runner import run_attempt
    spec = [c for c in load_golden() if "spec:mismatch" in c.tags]
    out = {}
    for version, attempt, folder in (("v1", 1, Path("fixtures/s10/edd_v1")), ("v2", V2_ATTEMPT, Path("fixtures/s10/attempts"))):
        rows = []
        for c in spec:
            a = run_attempt(c, attempt, version=version, folder=folder)
            fails = [f for chk in (check_outcome(c, a), check_tool_calls(c, a), check_trajectory(c, a)) for f in chk.failures]
            rows.append((c.id, not fails, fails))
        out[version] = rows
    out["all_pass_attempts"] = all_pass_attempts(spec)
    return out


def all_pass_attempts(spec=None) -> list[int]:
    """The recorded v2 attempts on which every spec case passes: attempt 3 is one of these, not a lucky pick."""
    from skyops.evals.checks import check_outcome, check_tool_calls, check_trajectory
    from skyops.evals.dataset import load_golden
    from skyops.evals.runner import N_ATTEMPTS, run_attempt
    spec = spec or [c for c in load_golden() if "spec:mismatch" in c.tags]
    ok = []
    for n in range(1, N_ATTEMPTS + 1):
        if all(chk(c, a).passed for c in spec for a in [run_attempt(c, n)]
               for chk in (check_outcome, check_tool_calls, check_trajectory)):
            ok.append(n)
    return ok


def main() -> None:
    res = results()
    for version, rows in ((v, res[v]) for v in ("v1", "v2")):
        label = "v1: notify on what the notice says" if version == "v1" else "v2: verify, and notify only what the system confirms"
        print(f"{label}: {sum(ok for _, ok, _ in rows)}/{len(rows)} spec cases pass")
        for cid, ok, fails in rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {cid:20} {fails[0] if fails else ''}")


if __name__ == "__main__":
    main()
