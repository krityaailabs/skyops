"""Lecture 11.3: an online eval over a week of (synthetic) production traces.

    python -m labs.s11.online_demo                # the week of 21 September, before Incident 18
    python -m labs.s11.online_demo --queue-day 2026-09-27   # print one day's human review queue
Sample by hashing the trace id, score the sample with the reference-free checks, compare two windows for score
drift (a two-proportion test per check, PSI over outcome classes), compare the golden set's notice formats with
traffic for input drift, and build the review queue. The log is synthetic (data/prod/traces.jsonl, built by
labs/s11/build_prod_log.py); the sampling, checks, tests and PSI are real code.
"""
from __future__ import annotations

import argparse
import tomllib
from collections import Counter
from datetime import date

from skyops.evals.dataset import load_golden
from skyops.evals.online import (CHECKS, check_failures, load_traces, new_bins, notice_format, outcome_counts, psi,
                                 psi_band, review_queue, sample, score_trace)
from skyops.evals.stats import drop_p, wilson_interval

from skyops.settings import ROOT

CONFIG = tomllib.loads((ROOT / "config/online.toml").read_text())["online"]
RATE = CONFIG["sample_rate"]
FORMATS = ("prose", "shorthand", "feed")


def window(traces, first: str, last: str):
    return [t for t in traces if first <= t["day"] <= last]


def scored_sample(traces, rate: float = RATE):
    return [score_trace(t) for t in sample(traces, rate)]


def formats(notices) -> dict[str, int]:
    c = Counter(notice_format(n) for n in notices)
    return {f: c.get(f, 0) for f in FORMATS}


def score_drift(ref, cur, eps: float = 1e-4) -> dict:
    """Per check: failures in each window and a one-sided test that the pass rate fell (Fisher's exact test when
    the cells are small, as they are here); PSI over outcome classes."""
    fr, fc = check_failures(ref), check_failures(cur)
    out = {}
    for name in CHECKS:
        p, method = drop_p(len(ref) - fr[name], len(ref), len(cur) - fc[name], len(cur))
        out[name] = {"ref_fail": fr[name], "cur_fail": fc[name], "p": p, "method": method}
    out["psi_outcomes"] = psi(outcome_counts(ref), outcome_counts(cur), eps=eps)
    return out


def figures(first="2026-09-21", split="2026-09-23", last="2026-09-27", rate=RATE, eps: float = 1e-4) -> dict:
    from datetime import date, timedelta
    traces = load_traces()
    week = window(traces, first, last)
    ref = scored_sample(window(traces, first, (date.fromisoformat(split) - timedelta(days=1)).isoformat()), rate)
    cur = scored_sample(window(traces, split, last), rate)
    golden = formats(c.notice for c in load_golden())
    return {"traces": traces, "week": week, "sampled": ref + cur, "ref": ref, "cur": cur,
            "drift": score_drift(ref, cur, eps), "golden_formats": golden, "eps": eps,
            "ref_formats": formats(s.trace["notice"] for s in ref),
            "cur_formats": formats(s.trace["notice"] for s in cur),
            "first": first, "split": split, "last": last, "rate": rate}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-day", default=None)
    ap.add_argument("--eps", type=float, default=1e-4, help="the value PSI puts in an empty bin")
    a = ap.parse_args(argv)
    f = figures(eps=a.eps)
    t = f["traces"]
    print(f"production log: {len(t):,} traces, {t[0]['day']} to {t[-1]['day']} (synthetic); week of {f['first']}: "
          f"{len(f['week'])}; sampled {f['rate']:.0%} by trace-id hash: {len(f['sampled'])}")
    fails = check_failures(f["sampled"])
    n = len(f["sampled"])
    print("failing checks in the sample: " + " · ".join(
        f"{k} {v} ({v / n:.1%}, 95% up to {wilson_interval(v, n)[1]:.1%})" for k, v in fails.items()))
    d = f["drift"]
    print(f"score drift, {f['split']} onward vs before ({len(f['cur'])} vs {len(f['ref'])} sampled):")
    for name in CHECKS:
        x = d[name]
        print(f"  {name:11} failing {x['ref_fail']} -> {x['cur_fail']}   p = {x['p']:.2f}"
              + (" (exact)" if x["method"] == "exact" else ""))
    print(f"  outcome classes: PSI {d['psi_outcomes']:.3f} ({psi_band(d['psi_outcomes'])}; empty bins set to "
          f"{f['eps']:g})")
    g, r, c = f["golden_formats"], f["ref_formats"], f["cur_formats"]
    show = lambda x: ", ".join(f"{k} {v}" for k, v in x.items())  # noqa: E731
    print(f"input drift, notice formats: golden set ({show(g)})")
    for label, x in ((f"traffic before {f['split']}", r), (f"traffic from {f['split']}", c)):
        fresh = new_bins(g, x)
        if fresh:
            total = sum(x.values())
            print(f"  {label} ({show(x)}): " + "; ".join(f"{b}: {x[b]} of {total} ({x[b] / total:.0%}), absent from "
                                                       "the golden set" for b in fresh))
        else:
            print(f"  {label} ({show(x)}): PSI {psi(g, x):.2f} ({psi_band(psi(g, x))})")
    q = review_queue(f["sampled"], per_day=CONFIG["queue_per_day"], rate_pass=CONFIG["random_pass_rate"])
    sizes = [len(v) for v in q.values()]
    print(f"review queue: {len(q)} days, {min(sizes)} to {max(sizes)} traces a day "
          f"({sum(i['why'] != 'random pass' for v in q.values() for i in v)} failing, the rest random passes)")
    if a.queue_day:
        items = q.get(a.queue_day, [])
        who = CONFIG["reviewers"]["weekend" if date.fromisoformat(a.queue_day).weekday() >= 5 else "weekday"]
        passes = [i for i in items if i["why"] == "random pass"]
        print(f"queue for {a.queue_day} ({len(items)} traces), reviewer: {who}")
        for item in items:
            if item["why"] != "random pass":
                print(f"  {item['trace_id']}  {item['why']}: {item['failures'][0]}")
        if passes:
            print(f"  ... and {len(passes)} random passes, starting {passes[0]['trace_id']}")


if __name__ == "__main__":
    main()
