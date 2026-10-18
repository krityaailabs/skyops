"""Lecture 11.5, Incident 18: eval contamination hides a regression.

    python -m labs.s11.incident --symptoms    # what production showed, and what the gate said
    python -m labs.s11.incident --evidence    # where v4 fails, the golden/held-out gap, the contamination scan
    python -m labs.s11.incident --signal      # the detection signal, replayed day by day
The gate itself is replayed with the real CLI:
    python -m skyops.evals.gate --handler v4 --suite golden --no-contamination --today 2026-09-28   # as it was
    python -m skyops.evals.gate --handler v4 --today 2026-09-28                                       # as it is now
The fault scenario itself (the feed format, the release, the rollback) is skyops/sim/incident18.py.
About the evidence: the production log is synthetic (labs/s11/build_prod_log.py) and the handler's eval
attempts are scripted replays; the checks, the scan, the tests and the gate are real code.
"""
from __future__ import annotations

import argparse
import logging
import tomllib
from collections import Counter

from skyops.evals.contamination import eval_items, find_contamination, model_inputs
from skyops.evals.dataset import load_golden
from skyops.evals.online import check_failures, load_traces, new_bins, notice_format, sample, score_trace
from skyops.evals.stats import drop_p
from skyops.settings import ROOT

CFG = tomllib.loads((ROOT / "config/online.toml").read_text())["online"]
REPORTED = "t-1002-0060"          # the trace behind the gate agent's report
RELEASE, ROLLBACK = "2026-09-28 10:00", "2026-10-02 16:30"
KNOWN_GOOD = ("2026-09-21", "2026-09-27")
LOGS = "data/prod/traces.jsonl"   # one JSON trace per handler run: notice, lookups with results, proposals, sends


def unconfirmed(scored) -> list:
    return [s for s in scored if not s.checks[0].passed]


def recipients(scored) -> int:
    return sum(c["result"]["recipients"] for s in scored for c in s.trace["calls"]
               if c["name"] == "notify_passengers" and c["result"].get("status") == "sent")


def figures() -> dict:
    logging.disable(logging.CRITICAL)
    traces = load_traces()
    scored = [score_trace(t) for t in traces]
    v4 = [s for s in scored if s.trace["prompt"] == "v4"]
    before = [s for s in scored if KNOWN_GOOD[0] <= s.trace["day"] <= KNOWN_GOOD[1]]
    rep = next(s for s in scored if s.trace["trace_id"] == REPORTED)
    bad = unconfirmed(v4)
    by_fmt = {f: (sum(notice_format(s.trace["notice"]) == f for s in bad),
                  sum(notice_format(s.trace["notice"]) == f for s in v4)) for f in ("feed", "prose", "shorthand")}
    v2 = [s for s in scored if s.trace["prompt"] == "v2"]
    v2_bad = unconfirmed(v2)
    v2_feed = (sum(notice_format(s.trace["notice"]) == "feed" for s in v2_bad),
               sum(notice_format(s.trace["notice"]) == "feed" for s in v2))
    v2_other = (len(v2_bad) - v2_feed[0], len(v2) - v2_feed[1])
    golden_fmt = Counter(notice_format(c.notice) for c in load_golden())
    v4_fmt = Counter(notice_format(s.trace["notice"]) for s in v4)
    from skyops.evals.harness import run_suite
    rates = {(s, v): run_suite(suite=s, version=v)["overall"]["pass_rate"]
             for s in ("golden", "heldout") for v in ("v2", "v4")}
    leaks = find_contamination(eval_items(), model_inputs("v4"))
    days = sorted({t["day"] for t in traces})

    def daily(pool, ref):
        fr = check_failures(ref)["confirmed"]
        out = []
        for d in days:
            if d <= KNOWN_GOOD[1]:
                continue
            cur = [s for s in pool if s.trace["day"] == d]
            fc = check_failures(cur)["confirmed"]
            p, _ = drop_p(len(ref) - fr, len(ref), len(cur) - fc, len(cur))
            out.append((d, fc, len(cur), p, p < CFG["alert_p"] and fc >= CFG["alert_min_failures"]))
        return out
    picked = {s.trace["trace_id"] for s in [score_trace(t) for t in sample(traces, CFG["sample_rate"])]}
    smp = [s for s in scored if s.trace["trace_id"] in picked]
    smp_before = [s for s in before if s.trace["trace_id"] in picked]
    feed_day = next(d for d in days if any(notice_format(s.trace["notice"]) == "feed"
                                           for s in scored if s.trace["day"] == d))
    feed_week = Counter(notice_format(s.trace["notice"]) for s in scored if feed_day <= s.trace["day"] <= KNOWN_GOOD[1])
    return {"traces": traces, "v4_runs": len(v4), "v4_bad": len(bad), "v4_people": recipients(bad),
            "before_runs": len(before), "before_bad": len(unconfirmed(before)), "reported": rep,
            "by_fmt": by_fmt, "golden_fmt": golden_fmt, "v4_fmt": v4_fmt, "v2_feed": v2_feed, "v2_other": v2_other,
            "new_v4": new_bins(golden_fmt, v4_fmt), "feed_week": feed_week,
            "feed_day": feed_day, "rates": rates, "leaks": leaks,
            "daily_all": daily(scored, before), "daily_sample": daily(smp, smp_before),
            "errors": sum(1 for s in v4 for c in s.trace["calls"] if c["result"].get("status") == "error")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    for flag in ("symptoms", "evidence", "signal"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args(argv)
    every = not (a.symptoms or a.evidence or a.signal)
    f = figures()
    if a.symptoms or every:
        r = f["reported"].trace
        sent = next(c for c in r["calls"] if c["name"] == "notify_passengers")
        print(f"reported: {r['trace_id']} at {r['ts'][11:16]}: {sent['result']['recipients']} passengers told: "
              f"\"{sent['args']['message']}\"")
        print(f"  notice: {r['notice']}")
        print(f"  the system at the time: {f['reported'].checks[0].failures[0]}")
        print(f"handler v4, {RELEASE} to {ROLLBACK}: {f['v4_runs']} runs; messages the system didn't support: "
              f"{f['v4_bad']}, to {f['v4_people']:,} passengers")
        print(f"handler v2, the week before: {f['before_runs']} runs; {f['before_bad']} such messages")
        print(f"tool errors {f['errors']} · every run answered · the eval gate on {RELEASE[:10]}: green")
        print(f"logs: {LOGS}, one trace per run (notice, lookups with their results, proposals, sends)")
    if a.evidence or every:
        print("where v4 went wrong: " + " · ".join(f"{k} {b} of {n} ({b / n:.0%})" for k, (b, n) in f["by_fmt"].items()))
        (fb, fn), (ob, on) = f["v2_feed"], f["v2_other"]
        print(f"control, v2 on the same kinds of notice: feed {fb} of {fn} ({fb / fn:.0%}) · prose and shorthand "
              f"{ob} of {on} ({ob / on:.0%})")
        g, v = f["golden_fmt"], f["v4_fmt"]
        total = sum(v.values())
        print(f"notice formats: golden set prose {g['prose']}, shorthand {g['shorthand']}, feed {g['feed']} · v4 traffic "
              f"prose {v['prose']}, shorthand {v['shorthand']}, feed {v['feed']}: "
              + "; ".join(f"{b} is {v[b] / total:.0%} of traffic and absent from the golden set" for b in f["new_v4"]))
        rt = f["rates"]
        print(f"pass rate, golden vs held-out: v2 {rt[('golden', 'v2')]:.0%} vs {rt[('heldout', 'v2')]:.0%} · "
              f"v4 {rt[('golden', 'v4')]:.0%} vs {rt[('heldout', 'v4')]:.0%}")
        print(f"contamination scan of prompt v4: {len(f['leaks'])} golden notices inside it: "
              + ", ".join(f"{h.item} ({h.share:.0%})" for h in f["leaks"]))
    if a.signal or every:
        print(f"signal: daily pass rate of 'confirmed' vs the last known-good week ({KNOWN_GOOD[0]} to {KNOWN_GOOD[1]}); "
              f"alert at p < {CFG['alert_p']} with {CFG['alert_min_failures']}+ failures")
        for label, rows in (("every trace scored by code", f["daily_all"]),
                            (f"a {CFG['sample_rate']:.0%} sample", f["daily_sample"])):
            first = next((d for d, *_, alert in rows if alert), None)
            show = " ".join(f"{d[5:]}:{fc}/{n}{'!' if al else ''}" for d, fc, n, _, al in rows)
            print(f"  {label}: {show}  first alert {first}")
        fw = f["feed_week"]
        print(f"input drift: from {f['feed_day']}, {fw['feed'] / sum(fw.values()):.0%} of notices were feed lines, a "
              "format absent from the golden set")


if __name__ == "__main__":
    main()
