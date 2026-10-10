"""Lecture 9.6: recall@5 and MRR on the labelled set, for two chunkers and four search modes.

    python -m labs.s09.eval_demo
Retrieval only: no model, no answers. The same labels score both chunkers, because they name sections.
"""
from __future__ import annotations

import sys

from skyops.retrieval.evals import evaluate, load_queries
from skyops.retrieval.indexing import build
from skyops.retrieval.search import Retriever

K = 5
MODES = [("keyword", False), ("vector", False), ("hybrid", False), ("hybrid", True)]


def reports() -> dict[str, list]:
    return {s: [evaluate(Retriever(build(strategy=s)), k=K, mode=m, rerank=rr) for m, rr in MODES]
            for s in ("naive", "structure")}


def loose() -> dict[str, object]:
    """Break it: score without the evidence phrase. Any chunk that touches a labelled section counts."""
    return {s: evaluate(Retriever(build(strategy=s)), k=K, mode="hybrid", rerank=True, evidence=False)
            for s in ("naive", "structure")}


def main() -> None:
    qs = load_queries()
    print(f"{len(qs)} labelled queries, k = {K}")
    print(f"{'chunking':10} {'mode':14} {'recall@5':>8} {'MRR':>6}  misses")
    for s, reps in reports().items():
        for e in reps:
            print(f"{s:10} {e.name:14} {e.recall:8.2f} {e.mrr:6.2f}  {', '.join(e.misses) or '-'}")
    if "--loose" in sys.argv:
        for s, e in loose().items():
            print(f"{s:10} {'(no evidence)':14} {e.recall:8.2f} {e.mrr:6.2f}  {', '.join(e.misses) or '-'}")


if __name__ == "__main__":
    main()
