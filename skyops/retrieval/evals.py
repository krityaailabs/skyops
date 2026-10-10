"""Lecture 9.6: measure retrieval on its own, before any model writes an answer.

The labelled set (data/procedures/eval_queries.jsonl) gives, for each query, who asks (tenant and role) and
the sections that answer it, as "<doc_id>#<section slug>" plus an evidence phrase. Labels name sections, not
chunk ids, so the same labels still work after you change the chunking: that is the point of comparing
chunkers. A retrieved chunk SATISFIES a label when it comes from that section and contains the evidence
phrase (whitespace and case ignored): a chunk that has the heading but lost the step doesn't count.

    recall@k   share of a query's labels satisfied somewhere in the top k: "did we fetch what's needed?"
    MRR        mean over queries of 1 / rank of the first result that satisfies any label (0 if none):
               "how high is the first useful result?" Models read the top of the context most reliably,
               and k is a cost, so rank matters, not only presence.

Neither says anything about the answer. An answer can be wrong with perfect retrieval (the model misread
the table) and right with none (it guessed). Measure both, separately, or you can't tell which stage broke.
Answer evals are Section 10.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .access import Principal
from .chunking import Chunk
from .docs import ROOT
from .search import Retriever

QUERIES = ROOT / "data/procedures/eval_queries.jsonl"


@dataclass(frozen=True)
class LabelledQuery:
    id: str
    query: str
    tenant: str
    role: str
    relevant: tuple[tuple[str, str], ...]          # (label, evidence)

    @property
    def principal(self) -> Principal:
        return Principal(f"eval-{self.id}", self.tenant, (self.role,))

    @property
    def labels(self) -> set[str]:
        return {lab for lab, _ in self.relevant}


def load_queries(path: Path = QUERIES) -> list[LabelledQuery]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(LabelledQuery(d["id"], d["query"], d["tenant"], d["role"],
                                     tuple((r["label"], r["evidence"]) for r in d["relevant"])))
    return out


def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def satisfied(c: Chunk, q: LabelledQuery, evidence: bool = True) -> set[str]:
    """The labels of q this chunk satisfies: right section, and the evidence phrase is in its text.
    evidence=False drops the phrase check (9.6's break-it): any chunk touching the section counts."""
    text = _norm(c.text)
    return {lab for lab, ev in q.relevant if lab in c.labels and (not evidence or _norm(ev) in text)}


def recall_at_k(ranked: list[set[str]], relevant: set[str], k: int) -> float:
    """Lecture 9.6: recall at k.

    ranked[i] is the set of relevant labels the result at rank i + 1 satisfies (empty if none).
    Return |(union of ranked[:k]) & relevant| / |relevant|. An empty `relevant` is an error (ValueError):
    a query with nothing to find can't measure recall; test it with an access or a no-answer check instead.
    """
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant labels")
    found: set[str] = set()
    for s in ranked[:k]:
        found |= s
    return len(found & relevant) / len(relevant)


def reciprocal_rank(ranked: list[set[str]], relevant: set[str]) -> float:
    """Lecture 9.6: 1 / rank of the first result whose set shares at least one label with `relevant` (ranks
    start at 1); 0.0 if no result does. The mean of this over queries is MRR."""
    for i, s in enumerate(ranked, 1):
        if s & relevant:
            return 1.0 / i
    return 0.0


@dataclass
class EvalReport:
    name: str
    k: int
    recall: float
    mrr: float
    misses: list[str] = field(default_factory=list)            # query ids with recall@k = 0
    rows: list[dict] = field(default_factory=list)
    context_tokens: float = 0.0                                 # mean per query
    usd_per_1000: float = 0.0


def evaluate(retriever: Retriever, queries: list[LabelledQuery] | None = None, k: int = 5, mode: str = "hybrid",
             rerank: bool = False, name: str | None = None, evidence: bool = True, **search_kw) -> EvalReport:
    queries = queries if queries is not None else load_queries()
    rows, rec, rr, tok, usd = [], 0.0, 0.0, 0, 0.0
    for q in queries:
        res = retriever.search(q.query, q.principal, k=k, mode=mode, rerank=rerank, **search_kw)
        ranked = [satisfied(h.chunk, q, evidence) for h in res.hits]
        r, m = recall_at_k(ranked, q.labels, k), reciprocal_rank(ranked, q.labels)
        rows.append({"id": q.id, "recall": r, "rr": m, "top": res.ids[:3]})
        rec, rr, tok, usd = rec + r, rr + m, tok + res.cost.context_tokens, usd + res.cost.usd
    n = len(queries)
    return EvalReport(name or f"{mode}{'+rerank' if rerank else ''}", k, round(rec / n, 4), round(rr / n, 4),
                      [row["id"] for row in rows if row["recall"] == 0], rows, round(tok / n, 1),
                      round(usd / n * 1000, 4))
