"""Lecture 10.3: load the golden set against its schema, validate it, and show its coverage.

    python -m labs.s10.golden_demo                the real set
    python -m labs.s10.golden_demo --mislabel     break it: n01's label says passengers hear the OLD time, 16:40
    python -m labs.s10.golden_demo --no-adversarial   break it: drop the adversarial cases
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    from skyops.evals.dataset import SCHEMA, coverage, load_golden, validate_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--mislabel", action="store_true")
    ap.add_argument("--no-adversarial", action="store_true")
    a = ap.parse_args(argv)
    cases = load_golden()
    print(f"loaded {len(cases)} cases, every line valid against {SCHEMA.name}")
    if a.mislabel:
        n01 = next(c for c in cases if c.id == "n01_delay_blr")
        n01.expect.notify.mention = ["16:40"]
    if a.no_adversarial:
        cases = [c for c in cases if c.category != "adversarial"]
    cov = coverage(cases)
    print("by category: " + ", ".join(f"{k} {v}" for k, v in cov["by_category"].items()))
    print(f"from Section 1's notices: {cov['from_section_1']}; synthetic: {cov['cases'] - cov['from_section_1']}")
    tags = cov["by_tag"]
    print("tags: " + ", ".join(f"{k} {v}" for k, v in tags.items() if not k.startswith("spec:")))
    problems = validate_dataset(cases)
    print(f"validate_dataset: {len(problems)} problem(s)")
    for p in problems:
        print(f"  {p}")


if __name__ == "__main__":
    main()
