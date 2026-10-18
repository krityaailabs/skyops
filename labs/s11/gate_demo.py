"""Lecture 11.1: one overall threshold versus a per-category gate against the baseline.

    python -m labs.s11.gate_demo                  # the deployed handler (v2) and the token-diet candidate (v3)
    python -m labs.s11.gate_demo --overall-only   # break it: gate on one overall number, as many teams do
The golden suite only, as the gate stood before Incident 18 (lecture 11.5 adds the held-out suite and the
contamination check). Model turns are scripted replays; the gate's arithmetic is real.
"""
from __future__ import annotations

import argparse
import json
import logging

from skyops.evals.gate import BASELINE, evaluate_gate, load_config, load_quarantine
from skyops.evals.harness import run_suite

OVERALL_MIN = 0.70        # the single number a naive gate checks


def figures(version: str) -> dict:
    logging.disable(logging.CRITICAL)
    cfg = {**load_config(), "suites": ["golden"]}
    base = json.loads(BASELINE.read_text())
    rep = run_suite(suite="golden", version=version, k=cfg["k"])
    res = evaluate_gate({"golden": rep}, base, cfg, load_quarantine(), "2026-10-17")
    from skyops.agent.disruption import SYSTEM
    return {"version": version, "overall": rep["overall"]["pass_rate"], "prompt_chars": len(SYSTEM[version]),
            "result": res}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overall-only", action="store_true")
    a = ap.parse_args(argv)
    for v in ("v2", "v3"):
        f = figures(v)
        if a.overall_only:
            ok = f["overall"] >= OVERALL_MIN
            print(f"handler {v}: overall pass rate {f['overall']:.0%} (gate: at least {OVERALL_MIN:.0%}) -> "
                  f"{'GREEN' if ok else 'RED'}")
            continue
        r = f["result"]
        print(f"handler {v}: instructions {f['prompt_chars']} characters, overall pass rate {f['overall']:.0%} -> "
              f"{'GREEN' if r.passed else 'RED'}")
        for x in r.failures:
            print(f"  FAIL {x}")
        for x in r.warnings:
            print(f"  WARN {x}")


if __name__ == "__main__":
    main()
