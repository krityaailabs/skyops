"""Lecture 9.1: retrieval, long context, or a database query? The numbers behind the decision, for SkyOps.

    python -m labs.s09.when_to_retrieve
Context sizes are estimates (about four characters per token); costs use the example price parameters.
"""
from __future__ import annotations

from skyops.retrieval.access import DIRECTORY, can_read
from skyops.retrieval.docs import TODAY, load_corpus
from skyops.retrieval.evals import evaluate
from skyops.retrieval.indexing import build
from skyops.retrieval.search import Retriever, RetrievalPrice
from skyops.retrieval.text import count_tokens
from skyops.sim import flights as sim

QUESTIONS = [
    ("How many departures are delayed right now?", "query", "a count over live operational data: ask the system of record"),
    ("What does WCHC mean?", "retrieval", "one passage in a procedure; exact code"),
    ("Compare every delay-care rule we have and list contradictions", "long context", "needs all of a small set at once"),
    ("What hotel voucher may I issue without approval?", "retrieval", "one rule, access-controlled, must be current"),
    ("Summarise this one procedure for a new starter", "long context", "one document, read whole"),
]


STATION_DOCS = 2000          # a real station's procedure library is in the thousands (illustrative scale)


def figures() -> dict:
    p = DIRECTORY["agent-anna"]
    docs = [d for d in load_corpus() if can_read(p, d.meta) and d.meta.current_on(TODAY)]
    everything = sum(count_tokens(d.body) for d in docs)
    r = Retriever(build())
    rep = evaluate(r, k=5, mode="hybrid", rerank=True)
    plain = evaluate(r, k=5, mode="hybrid")
    price = RetrievalPrice.load()
    delayed = sum(1 for f in sim.departures_on(sim.DATE) if f.status == "DELAYED")
    return {"docs": len(docs), "all_tokens": everything, "retrieved_tokens": round(rep.context_tokens),
            "all_per_1000_usd": round(everything * price.context_per_mtok / 1e6 * 1000, 2),
            "retrieved_per_1000_usd": round(rep.usd_per_1000, 2), "plain_per_1000_usd": round(plain.usd_per_1000, 2),
            "delayed_now": delayed, "station_docs": STATION_DOCS,
            "station_tokens": round(everything / len(docs) * STATION_DOCS)}


def main() -> None:
    f = figures()
    print(f"procedures an agent may read today: {f['docs']}, about {f['all_tokens']} tokens in all")
    print(f"whole set in context: {f['all_tokens']} tokens per question, {f['all_per_1000_usd']:.2f} USD per 1,000 questions")
    print(f"retrieval, top 5:     {f['retrieved_tokens']} tokens per question, {f['retrieved_per_1000_usd']:.2f} USD per 1,000 "
          "questions (including rerank)")
    print(f"                      {f['plain_per_1000_usd']:.2f} USD per 1,000 questions without the reranker")
    print(f"at {f['station_docs']} procedures, the whole set would be about {f['station_tokens']} tokens per question")
    print(f"delayed departures right now (a query, not retrieval): {f['delayed_now']}")
    for q, tool, why in QUESTIONS:
        print(f"  {tool:12} {q}  ({why})")


if __name__ == "__main__":
    main()
