"""Lecture 10.5: deterministic checks before an LLM judge. Run message_assertions over the DEVELOPMENT part of the
human-labelled passenger messages (the part the rules were written against), and see what code decides on its
own, at no model cost. The held-out part is kept for the judge's calibration in lecture 10.6.

    python -m labs.s10.assertions_demo
    python -m labs.s10.assertions_demo --any-time         break it: the time check only asks "is there a time?"
    python -m labs.s10.assertions_demo --naive-promises   the first version of the money rules: any amount or "refund"
"""
from __future__ import annotations

import argparse
import re

# The rules before the labelled set had messages that mention a voucher or a refund without promising anything.
NAIVE_PROMISES = [re.compile(p, re.I) for p in (r"\d+\s?(?:EUR|euros?)\b", r"€\s?\d+", r"\brefund\w*", r"\bcompensat\w*")]
KINDS = {"facts": lambda x: re.match(r"(flight number|new time|time not)", x),
         "promises": lambda x: x.startswith("promise"),
         "internal codes": lambda x: x.startswith("internal code")}


def figures(any_time: bool = False, naive_promises: bool = False) -> dict:
    from skyops.evals import checks
    from skyops.evals.judge import load_labels
    items = load_labels(split="dev")
    rows = []
    for it in items:
        rec = it.record
        fails = checks.message_assertions(it.message, rec)
        if any_time:                                   # the weaker rule: any HH:MM will do
            fails = [f for f in fails if not f.startswith(("new time missing", "time not in the system"))]
            if rec is not None and rec.status == "DELAYED" and not checks.TIME.search(it.message):
                fails.append(f"new time missing: {rec.estimated_departure}")
        if naive_promises:                             # any amount or the word refund, promise or not
            fails = [f for f in fails if not f.startswith("promise")]
            fails += [f"promise of money or a refund: '{m.group(0)}'" for p in NAIVE_PROMISES
                      if (m := p.search(it.message))]
        rows.append((it, fails))
    caught = [(it, f) for it, f in rows if f]
    kinds = {k: 0 for k in KINDS}
    example = {}
    for it, f in caught:
        for kind, is_kind in KINDS.items():
            hits = [x for x in f if is_kind(x)]
            kinds[kind] += bool(hits)
            if hits and kind not in example:
                example[kind] = (it.id, hits[0])
    return {"items": len(items), "human_fail": sum(it.human == "fail" for it in items),
            "caught": len(caught), "caught_human_fail": sum(it.human == "fail" for it, _ in caught),
            "false_alarms": sum(it.human == "pass" for it, _ in caught), "kinds": kinds, "example": example,
            "left_for_judge": len(items) - len(caught),
            "human_fail_left": sum(it.human == "fail" for it, f in rows if not f), "rows": rows}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--any-time", action="store_true")
    ap.add_argument("--naive-promises", action="store_true")
    a = ap.parse_args(argv)
    f = figures(a.any_time, a.naive_promises)
    print(f"{f['items']} labelled messages (development part), {f['human_fail']} failed by people")
    print(f"assertions fail {f['caught']}: {f['caught_human_fail']} people also failed, {f['false_alarms']} false alarms")
    print("  by kind: " + ", ".join(f"{k} {v}" for k, v in f["kinds"].items()))
    if a.naive_promises:
        for it, fails in f["rows"]:
            if fails and it.human == "pass":
                print(f"  false alarm {it.id}: {fails[0]}")
    else:
        for kind in ("promises", "internal codes", "facts"):
            if kind in f["example"]:
                print(f"  {f['example'][kind][0]}: {f['example'][kind][1]}")
    print(f"left for the judge: {f['left_for_judge']} messages, of which {f['human_fail_left']} people failed")


if __name__ == "__main__":
    main()
