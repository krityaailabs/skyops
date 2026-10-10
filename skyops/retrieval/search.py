"""Lecture 9.3: keyword search, vector search, fusion, metadata filters, reranking, and cost per query.

    keyword    BM25 over chunk text: exact codes, numbers and rare words (WCHC, SOP-GND-014, 150)
    vector     cosine over embeddings: word forms, spelling slips; with a real model, paraphrases
    hybrid     both lists fused with Reciprocal Rank Fusion (Cormack, Clarke and Buettcher, SIGIR 2009):
               score(d) = sum over lists of 1 / (k + rank), k = 60, ranks from 1. Ranks, not scores, so BM25's
               unbounded scores and cosine's [-1, 1] never have to be put on one scale.
    rerank     re-score the fused candidates by looking at query and chunk together. OverlapReranker is a
               deterministic stand-in for a cross-encoder or a hosted rerank model: same interface, same
               place in the pipeline, measured by the same evals.

Every search returns a QueryCost: tokens embedded for the query, rerank calls, and the context tokens the
results will cost when they are handed to a model, priced from config/retrieval.toml and config/prices.toml
(example parameters, not quotes).
"""
from __future__ import annotations

import datetime as dt
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..cost import load_price
from .access import Principal, can_read
from .chunking import Chunk
from .docs import TODAY, DocMeta
from .store import ChunkStore, Hit, TenantIndexes
from .text import count_tokens, tokens

RETRIEVAL_PRICES = Path(__file__).resolve().parent.parent.parent / "config/retrieval.toml"
RRF_K = 60


@dataclass(frozen=True)
class Filters:
    """Metadata filters. as_of is always set: a search is always 'as of' a day."""
    as_of: dt.date = TODAY
    owners: frozenset[str] | None = None        # e.g. {"Ground Operations"}
    doc_prefix: str | None = None               # e.g. "SOP-GND"


def matches(meta: DocMeta, f: Filters) -> bool:
    """Lecture 9.3: does a document's metadata pass the filters?

    - current on f.as_of: meta.effective <= f.as_of, and meta.withdrawn is None or f.as_of < meta.withdrawn
      (a withdrawn document is never current again; a future version isn't current yet);
    - if f.owners is set: meta.owner is in it;
    - if f.doc_prefix is set: meta.doc_id starts with it.
    """
    if not (meta.effective <= f.as_of and (meta.withdrawn is None or f.as_of < meta.withdrawn)):
        return False
    if f.owners is not None and meta.owner not in f.owners:
        return False
    return f.doc_prefix is None or meta.doc_id.startswith(f.doc_prefix)


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Lecture 9.3: Reciprocal Rank Fusion of several ranked lists of chunk ids.

    score(id) = sum over the lists that contain id of 1 / (k + rank), with rank starting at 1.
    Return (id, score) pairs, highest score first; ties broken by id, ascending, so the order is stable.
    An id missing from a list simply gets nothing from that list.
    """
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


_CODE = re.compile(r"\b[A-Z]{2,}[A-Z0-9-]*\d*\b")


@dataclass
class OverlapReranker:
    """Query and chunk scored together: how much of the query the chunk covers, whether its heading is about
    the query, whether query phrases appear in order, and whether codes in the query appear exactly."""
    name: str = "overlap-v1"

    def score(self, query: str, c: Chunk) -> float:
        q = tokens(query)
        if not q:
            return 0.0
        qs = set(q)
        first, _, rest = c.text.partition("\n")
        body = tokens(c.text)
        head = set(tokens(first.split(" > ", 1)[1])) if " > " in first else set()   # its section heading
        cover = len(qs & set(body)) / len(qs)
        heading = len(qs & head) / len(qs)
        qb = {(a, b) for a, b in zip(q, q[1:])}
        cb = set(zip(body, body[1:]))
        phrase = len(qb & cb) / len(qb) if qb else 0.0
        codes = set(_CODE.findall(query)) - {"ZZ", "YY", "HUB", "EUR"}
        exact = (sum(code in c.text for code in codes) / len(codes)) if codes else 0.0
        return round(0.5 * cover + 0.25 * heading + 0.15 * phrase + 0.1 * exact, 6)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        scored = [Hit(h.chunk, self.score(query, h.chunk), "rerank") for h in hits]
        return sorted(scored, key=lambda h: (-h.score, h.chunk.chunk_id))


@dataclass(frozen=True)
class RetrievalPrice:
    embed_per_mtok: float
    rerank_per_1000_searches: float
    rerank_docs_per_search: int
    context_per_mtok: float
    name: str = "example"

    @classmethod
    def load(cls, name: str = "example", path: Path = RETRIEVAL_PRICES) -> "RetrievalPrice":
        row = tomllib.loads(path.read_text(encoding="utf-8"))[name]
        model = load_price(row["model_price_profile"])
        return cls(row["embed_per_mtok"], row["rerank_per_1000_searches"], row["rerank_docs_per_search"],
                   model.input_per_mtok, name)


@dataclass
class QueryCost:
    query_tokens: int = 0            # embedded at search time (vector and hybrid only)
    rerank_calls: int = 0
    reranked_docs: int = 0
    context_tokens: int = 0          # what the returned chunks cost when a model reads them
    usd: float = 0.0


@dataclass
class SearchResult:
    hits: list[Hit]
    cost: QueryCost
    mode: str
    candidates: int = 0

    @property
    def ids(self) -> list[str]:
        return [h.chunk.chunk_id for h in self.hits]


@dataclass
class Retriever:
    """Access-controlled search over per-tenant indexes. The principal decides which store is searched and
    which chunks may be ranked at all; filters narrow further; nothing a caller passes can widen access.
    enforce_access=False and shared exist ONLY to demonstrate the leaks in 9.4."""
    indexes: TenantIndexes
    reranker: OverlapReranker = field(default_factory=OverlapReranker)
    price: RetrievalPrice = field(default_factory=RetrievalPrice.load)
    enforce_access: bool = True
    shared: ChunkStore | None = None            # one store for every tenant (the leak demo)

    def _store(self, p: Principal) -> ChunkStore:
        return self.shared if self.shared is not None else self.indexes.store(p.tenant)

    def search(self, query: str, principal: Principal, k: int = 5, mode: str = "hybrid", rerank: bool = False,
               filters: Filters | None = None, candidates: int = 20, prefilter: bool = True,
               fusion: str = "rrf") -> SearchResult:
        """fusion="sum" adds raw scores instead of fusing ranks: shown in 9.3's break-it, never used."""
        f = filters or Filters()

        def allow(c: Chunk) -> bool:
            return (not self.enforce_access or can_read(principal, c.meta)) and matches(c.meta, f)

        store = self._store(principal)
        n = max(candidates, k)
        cost = QueryCost()
        if mode == "keyword":
            hits = store.keyword(query, n, allow, prefilter)
        elif mode == "vector":
            hits = store.vector(query, n, allow, prefilter)
            cost.query_tokens = count_tokens(query)
        elif mode == "hybrid":
            kw = store.keyword(query, n, allow, prefilter)
            vec = store.vector(query, n, allow, prefilter)
            cost.query_tokens = count_tokens(query)
            if fusion == "sum":
                total: dict[str, float] = {}
                for h in kw + vec:
                    total[h.chunk.chunk_id] = total.get(h.chunk.chunk_id, 0.0) + h.score
                fused = sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))
            else:
                fused = rrf_fuse([[h.chunk.chunk_id for h in kw], [h.chunk.chunk_id for h in vec]])
            hits = [Hit(store.chunks[cid], s, "hybrid") for cid, s in fused[:n]]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        pool = len(hits)
        if rerank and hits:
            hits = self.reranker.rerank(query, hits)
            cost.rerank_calls = -(-len(hits) // self.price.rerank_docs_per_search)
            cost.reranked_docs = len(hits)
        hits = hits[:k]
        cost.context_tokens = sum(h.chunk.tokens for h in hits)
        cost.usd = (cost.query_tokens * self.price.embed_per_mtok / 1e6
                    + cost.rerank_calls * self.price.rerank_per_1000_searches / 1000
                    + cost.context_tokens * self.price.context_per_mtok / 1e6)
        return SearchResult(hits, cost, mode + ("+rerank" if rerank else ""), pool)
