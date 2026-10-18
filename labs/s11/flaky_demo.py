"""Lecture 11.2: a flaky gate, and the same gate stabilised.

    python -m labs.s11.flaky_demo                 # run on the solved tag; it uses 11.2's interval and test
    python -m labs.s11.flaky_demo --today 2026-11-01   # break it: the day after the quarantine expires
"Five CI runs of one build" are simulated honestly: a live run with one attempt per case is one sample of each
case, so run i uses recorded attempt i of every case (the same scripted replays as Section 10). Everything that
judges them is the real gate code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging

from skyops.evals.gate import gate_today, load_config, load_quarantine, quarantine_problems
from skyops.evals.harness import run_suite
from skyops.evals.stats import wilson_interval

NAIVE_MIN = 0.85          # the naive gate: one attempt per case, pass rate at least 85%


def figures(today: str = "2026-10-17") -> dict:
    logging.disable(logging.CRITICAL)
    cfg = load_config()
    rep = run_suite(suite="golden", version="v2", k=cfg["k"])
    from skyops.evals.dataset import load_golden
    cases = load_golden()
    from skyops.evals.runner import run_case
    from skyops.evals.harness import score_attempt
    green = {c.id: [score_attempt(c, a).passed for a in run_case(c)] for c in cases}
    n = len(next(iter(green.values())))
    runs = [sum(g[i] for g in green.values()) for i in range(n)]
    total_c = sum(r["passes"] for r in rep["cases"])
    total_n = sum(r["n"] for r in rep["cases"])
    single = [(g, *wilson_interval(g, len(cases))) for g in runs]      # each run's own 95% interval
    rerun = sum(any(g[:3]) for g in green.values())
    shas = {hashlib.sha256(json.dumps(run_suite(suite="golden", version="v2", k=cfg["k"]), sort_keys=True)
                           .encode()).hexdigest()[:12] for _ in range(2)}
    cats = {}
    for cat in ("normal", "edge", "adversarial"):
        rs = [r for r in rep["cases"] if r["category"] == cat]
        c, m = sum(r["passes"] for r in rs), sum(r["n"] for r in rs)
        cats[cat] = (c, m, *wilson_interval(c, m))
    flaky = [r for r in rep["cases"] if 0 < r["passes"] < r["n"]]
    known = {"golden": {r["id"]: r["category"] for r in rep["cases"]}}
    active, problems = quarantine_problems(load_quarantine(), today, known, cfg["max_quarantine_days"])
    return {"cases": len(cases), "n": n, "runs": runs, "single": single, "rerun_green": rerun, "shas": shas,
            "cats": cats, "flaky": flaky, "k": cfg["k"], "pass_hat_k": rep["overall"]["pass_hat_k"],
            "active": sorted(active), "problems": problems, "today": today,
            "total": (total_c, total_n)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=None)
    a = ap.parse_args(argv)
    f = figures(a.today or gate_today())
    m = f["cases"]
    print(f"one build, {f['n']} CI runs as live runs would see them (one attempt per case); gate: pass rate at least "
          f"{NAIVE_MIN:.0%}")
    print("  " + "   ".join(f"run {i + 1}: {g}/{m} {g / m:.0%} {'GREEN' if g / m >= NAIVE_MIN else 'RED'}"
                           for i, g in enumerate(f["runs"])))
    print(f"rerun each red case until green (up to 3 tries): {f['rerun_green']}/{m} green, every flaky case hidden")
    c, n = f["total"]
    print(f"stabilised: replay, all {f['n']} attempts per case ({c}/{n}), compared with the baseline")
    print(f"  two runs, same recordings: report sha256 {' and '.join(sorted(f['shas']))}"
          f" ({'identical' if len(f['shas']) == 1 else 'DIFFERENT'})")
    print("  95% intervals: " + " · ".join(f"{cat} {x}/{y} {lo:.0%}-{hi:.0%}" for cat, (x, y, lo, hi) in f["cats"].items()))
    worst = min(f["single"])
    print(f"  the worst single run, {worst[0]}/{m}, has a 95% interval of {worst[1]:.0%}-{worst[2]:.0%}: it contains "
          f"the full rate, {c / n:.0%}, so one run can't tell this from noise")
    print(f"flaky cases (0 < passes < {f['n']}): {len(f['flaky'])} of {m}; counted through pass^{f['k']} = "
          f"{f['pass_hat_k']:.2f}, never rerun away")
    for q in f["active"]:
        print(f"quarantine in force on {f['today']}: {q}")
    for p in f["problems"]:
        print(f"FAIL {p}")
    print("quarantine: " + ("RED, an expired or invalid entry fails the gate" if f["problems"] else
                            "every entry has an owner, a ticket and an expiry"))


if __name__ == "__main__":
    main()
