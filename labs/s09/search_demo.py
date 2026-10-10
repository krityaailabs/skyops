"""Lecture 9.3: keyword, vector, hybrid and reranked search compared on the labelled set; metadata filters;
cost per query.

    python -m labs.s09.search_demo
"""
from __future__ import annotations

import datetime as dt
import sys

from skyops.retrieval.access import DIRECTORY
from skyops.retrieval.evals import evaluate
from skyops.retrieval.indexing import build
from skyops.retrieval.search import Filters, Retriever

MODES = [("keyword", False), ("vector", False), ("hybrid", False), ("hybrid", True)]
K = 3
TYPO = "cancelation report deadline"


def table(r: Retriever) -> list:
    return [evaluate(r, k=K, mode=m, rerank=rr) for m, rr in MODES]


def score_sum(r: Retriever):
    """Break it: fuse by adding raw scores. BM25 scores often reach several units or more, cosines stay below one, so the sum mostly follows BM25."""
    return evaluate(r, k=K, mode="hybrid", name="hybrid (sum)", fusion="sum")


def main() -> None:
    r = Retriever(build())
    print(f"{'mode':15} {'recall@3':>8} {'MRR':>6} {'ctx tokens':>10} {'USD/1,000 queries':>18}")
    rows = table(r) + ([score_sum(r)] if "--sum" in sys.argv else [])
    for e in rows:
        print(f"{e.name:15} {e.recall:8.2f} {e.mrr:6.2f} {e.context_tokens:10.0f} {e.usd_per_1000:18.2f}")
    dm = DIRECTORY["dm-deepa"]
    for m, rr in MODES:
        res = r.search(TYPO, dm, k=5, mode=m, rerank=rr)
        rank = next((i for i, h in enumerate(res.hits, 1) if "after-the-event" in h.chunk.sections), None)
        print(f"{TYPO!r} {res.mode:15} -> SOP-OPS-040 'After the event' at rank {rank or 'not in top 5'}")
    agent = DIRECTORY["agent-anna"]
    for day in (dt.date(2026, 10, 9), dt.date(2026, 11, 2)):
        res = r.search("meal voucher value per passenger", agent, k=1, mode="keyword", filters=Filters(as_of=day))
        h = res.hits[0].chunk
        amount = "18 EUR" if "18 EUR" in h.text else "15 EUR"
        print(f"as of {day}: {h.meta.key} ({amount})")
    res = r.search("passenger care", agent, k=5, filters=Filters(owners=frozenset({"Baggage Services"})))
    print(f"owner filter Baggage Services: {sorted({h.chunk.meta.doc_id for h in res.hits})}")


if __name__ == "__main__":
    main()
