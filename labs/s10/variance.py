"""Lecture 10.7: variance across runs. Every golden case has five recorded attempts; one green run proves little.

    python -m labs.s10.variance
Deterministic checks only (no judge), so the numbers are about the handler, not the grader. The attempts are
scripted replays: which attempts fail is written down in labs/s10/build_fixtures_s10.py, not sampled.
"""
from __future__ import annotations

import logging

K = 3


def figures() -> dict:
    logging.disable(logging.CRITICAL)
    from skyops.evals.dataset import load_golden
    from skyops.evals.harness import score_attempt
    from skyops.evals.runner import N_ATTEMPTS, run_case
    from skyops.evals.stats import pass_at_k, pass_hat_k
    from skyops.evals.runner import attempt_files, turns_label
    cases = load_golden()
    passes = {c.id: [score_attempt(c, a).passed for a in run_case(c)] for c in cases}
    n = N_ATTEMPTS
    per_run = [sum(p[i] for p in passes.values()) for i in range(n)]
    cs = {cid: sum(p) for cid, p in passes.items()}
    mean = lambda xs: sum(xs) / len(xs)   # noqa: E731
    p1 = mean([c / n for c in cs.values()])
    example = next(cid for cid, c in cs.items() if c == 3)
    return {"cases": len(cases), "n": n, "turns": turns_label(attempt_files([c.id for c in cases], n)), "k": K, "per_run": per_run, "passes": passes, "counts": cs,
            "pass_rate": p1, "pass_at_k": mean([pass_at_k(n, c, K) for c in cs.values()]),
            "pass_hat_k": mean([pass_hat_k(n, c, K) for c in cs.values()]),
            "every_green": sum(c == n for c in cs.values()),
            "example": example, "naive_at_k": 1 - (1 - 3 / n) ** K, "naive_hat_k": (3 / n) ** K,
            "unbiased_at_k": pass_at_k(n, 3, K), "unbiased_hat_k": pass_hat_k(n, 3, K),
            "challenge_at_2": pass_at_k(n, 3, 2), "challenge_hat_2": pass_hat_k(n, 3, 2)}


def main() -> None:
    f = figures()
    print(f"{f['cases']} cases x {f['n']} attempts, deterministic checks; model turns: {f['turns']}")
    print("green cases, run by run: " + "  ".join(f"run {i + 1}: {g}/{f['cases']}" for i, g in enumerate(f["per_run"])))
    hist = {}
    for c in f["counts"].values():
        hist[c] = hist.get(c, 0) + 1
    print("cases by passes out of 5: " + ", ".join(f"{c}/5: {hist[c]}" for c in sorted(hist, reverse=True)))
    print(f"pass rate {f['pass_rate']:.0%}   pass@{f['k']} {f['pass_at_k']:.0%}   pass^{f['k']} {f['pass_hat_k']:.0%}")
    print(f"every attempt green: {f['every_green']} of {f['cases']} cases")
    print(f"one case, 3 of 5 passing ({f['example']}): pass@3 unbiased {f['unbiased_at_k']:.2f}, naive "
          f"{f['naive_at_k']:.3f};  pass^3 unbiased {f['unbiased_hat_k']:.2f}, naive {f['naive_hat_k']:.3f}")


if __name__ == "__main__":
    main()
