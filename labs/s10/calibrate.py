"""Lecture 10.6 (demo): calibrate the LLM judge against human labels, and look for its biases.

    python -m labs.s10.calibrate                  agreement, kappa, the confusion matrix, length and position tests
    python -m labs.s10.calibrate --always-pass    break it: a "judge" that passes everything, no model at all
About this recording: the judge's verdicts are a SCRIPTED replay (fixtures/s10/judge.jsonl), written to show
the kinds of bias published research reports. The kappa, the bias measurements and the decision are real code.
"""
from __future__ import annotations

import argparse


class AlwaysPass:
    """Not a model: the cheapest possible judge. Its agreement looks respectable; its kappa doesn't."""
    calls = 0
    per_1000_calls_usd = 0.0

    def grade(self, message, flight):
        from skyops.evals.judge import Verdict
        return Verdict("pass", "always")


def main(argv: list[str] | None = None) -> None:
    from skyops.evals.judge import LONG_CHARS, THRESHOLD, Judge, calibrate, position_test
    from skyops.evals.stats import interpret_kappa
    ap = argparse.ArgumentParser()
    ap.add_argument("--always-pass", action="store_true")
    a = ap.parse_args(argv)
    if a.always_pass:
        cal = calibrate(AlwaysPass())
        print(f"always-pass judge: agreement {cal.agreement_judge:.0%}, kappa {cal.kappa_judge:.2f} "
              f"({interpret_kappa(cal.kappa_judge)})")
        return
    judge = Judge()
    cal = calibrate(judge)
    print(f"{cal.items} held-out labelled messages ({cal.human_pass} pass by people); judge: {judge.client.model} "
          f"(replay)")
    lo, hi = cal.kappa_judge_ci
    print(f"judge alone:            agreement {cal.agreement_judge:.0%}, kappa {cal.kappa_judge:.2f} "
          f"({interpret_kappa(cal.kappa_judge)}; 95% interval {lo:.2f} to {hi:.2f})")
    c = cal.confusion("judge")
    print("  " + ", ".join(f"{k} {v}" for k, v in c.items()))
    print(f"assertions, then judge: agreement {cal.agreement_pipeline:.0%}, kappa {cal.kappa_pipeline:.2f} "
          f"({interpret_kappa(cal.kappa_pipeline)}); {cal.caught_by_assertions} of the {cal.items} decided by code")
    lo, hi = cal.kappa_judged_ci
    print(f"judge on the {cal.judged_items} it grades: agreement {cal.agreement_judged:.0%}, kappa "
          f"{cal.kappa_judged:.2f} ({interpret_kappa(cal.kappa_judged)}; 95% interval {lo:.2f} to {hi:.2f})")
    fp = [d for d in cal.disagreements if d["judge"] == "pass"]
    print(f"disagreements: {len(cal.disagreements)}; judge passed {len(fp)} that people failed, "
          f"{sum(d['chars'] >= LONG_CHARS for d in fp)} of them long")
    for name, s in cal.by_length.items():
        print(f"  {name:5} messages ({s['items']}): people pass {s['human_pass_share']:.0%}, judge passes "
              f"{s['judge_pass_share']:.0%}")
    for pd in cal.padding:
        print(f"padding test: {pd['plain']} plain -> judge {pd['judge_plain']}; {pd['padded']} (the same, padded) -> "
              f"judge {pd['judge_padded']}; people: {pd['human']} both")
    pt = position_test(judge)
    print(f"position test: {pt.pairs} pairs, each shown both ways = {pt.calls} calls")
    print(f"  consistent {pt.consistent}, inconsistent {pt.inconsistent} (all {pt.inconsistent_all_first} chose "
          f"the message shown first); first slot won {pt.first_slot_share:.0%}")
    print(f"  swap-and-agree keeps {pt.consistent} pairs, ties {pt.inconsistent}; agrees with people on "
          f"{pt.agree_with_human} of {pt.consistent}")
    print(f"decision: the judge's own kappa {cal.kappa_judged:.2f} {'>=' if cal.usable else '<'} {THRESHOLD}: "
          f"{'count the judge, after the assertions, pointwise only' if cal.usable else 'do not count the judge yet'}")

if __name__ == "__main__":
    main()
