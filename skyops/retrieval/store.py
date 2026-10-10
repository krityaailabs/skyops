"""Lectures 9.2 and 9.3: a chunk store for ONE tenant: chunk texts, their vectors, and a BM25 keyword index.

Exact search (every vector is scored), which is right for a few thousand chunks and makes results
reproducible. Approximate indexes (HNSW, IVF) trade a little recall for speed at millions of chunks; lecture
9.8 is about when that trade is worth making and where it lives.

Filtering: `allow` is a predicate on a chunk. With prefilter=True (the default) it runs BEFORE ranking, so the
k results are the best k the caller may see. prefilter=False ranks everything and filters the top k after,
which is what many vector stores do by default: a caller with narrow access can get fewer than k results,
or none, while allowed matches exist further down.

The store records the embedder's id and refuses vectors from any other: an index built with one model can't
be queried with another (re-embed everything when you change models).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .chunking import Chunk
from .embed import Embedder, cosine
from .text import tokens

Allow = Callable[[Chunk], bool]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str                     # keyword | vector | hybrid | rerank


@dataclass
class BM25:
    """Okapi BM25 with the Lucene idf, ln(1 + (N - n + 0.5) / (n + 0.5)). k1 = 1.2 and b = 0.75 are the usual
    defaults. Incremental: documents (here, chunks) can be added and removed."""
    k1: float = 1.2
    b: float = 0.75
    tf: dict[str, Counter] = field(default_factory=dict)
    length: dict[str, int] = field(default_factory=dict)
    df: Counter = field(default_factory=Counter)
    postings: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, cid: str, text: str) -> None:
        if cid in self.tf:
            self.remove(cid)
        t = Counter(tokens(text))
        self.tf[cid], self.length[cid] = t, sum(t.values())
        for term in t:
            self.df[term] += 1
            self.postings[term].add(cid)

    def remove(self, cid: str) -> None:
        for term in self.tf.pop(cid, Counter()):
            self.df[term] -= 1
            self.postings[term].discard(cid)
            if self.df[term] == 0:
                del self.df[term], self.postings[term]
        self.length.pop(cid, None)

    def scores(self, query: str) -> dict[str, float]:
        n = len(self.tf)
        if n == 0:
            return {}
        avg = sum(self.length.values()) / n
        out: dict[str, float] = defaultdict(float)
        for term in set(tokens(query)):
            if term not in self.df:
                continue
            idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            for cid in self.postings[term]:
                f = self.tf[cid][term]
                out[cid] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.length[cid] / avg))
        return dict(out)


@dataclass
class ChunkStore:
    tenant: str                       # "*" = one shared store for every tenant (only to demonstrate 9.4's leak)
    embedder: Embedder
    chunks: dict[str, Chunk] = field(default_factory=dict)
    vectors: dict[str, list[float]] = field(default_factory=dict)
    bm25: BM25 = field(default_factory=BM25)
    embedder_id: str = ""

    def __post_init__(self) -> None:
        self.embedder_id = self.embedder.id

    def add(self, chunks: list[Chunk]) -> int:
        bad = [c.chunk_id for c in chunks if self.tenant != "*" and c.meta.tenant != self.tenant]
        if bad:
            raise ValueError(f"store for tenant {self.tenant!r} refused chunks of another tenant: {bad[:3]}")
        if self.embedder.id != self.embedder_id:
            raise ValueError(f"index built with {self.embedder_id}; refusing vectors from {self.embedder.id}")
        vecs = self.embedder.embed([c.text for c in chunks]) if chunks else []
        for c, v in zip(chunks, vecs):
            self.chunks[c.chunk_id], self.vectors[c.chunk_id] = c, v
            self.bm25.add(c.chunk_id, c.text)
        return len(chunks)

    def remove_where(self, pred: Callable[[Chunk], bool]) -> int:
        gone = [cid for cid, c in self.chunks.items() if pred(c)]
        for cid in gone:
            del self.chunks[cid], self.vectors[cid]
            self.bm25.remove(cid)
        return len(gone)

    def doc_keys(self) -> set[str]:
        return {c.meta.key for c in self.chunks.values()}

    def _top(self, scored: dict[str, float], k: int, allow: Allow | None, prefilter: bool, source: str) -> list[Hit]:
        order = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        if allow is not None and prefilter:
            order = [(cid, s) for cid, s in order if allow(self.chunks[cid])]
        top = [Hit(self.chunks[cid], s, source) for cid, s in order[:k]]
        if allow is not None and not prefilter:
            top = [h for h in top if allow(h.chunk)]
        return top

    def keyword(self, query: str, k: int = 10, allow: Allow | None = None, prefilter: bool = True) -> list[Hit]:
        return self._top(self.bm25.scores(query), k, allow, prefilter, "keyword")

    def vector(self, query: str, k: int = 10, allow: Allow | None = None, prefilter: bool = True) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        return self._top({cid: cosine(q, v) for cid, v in self.vectors.items()}, k, allow, prefilter, "vector")


@dataclass
class TenantIndexes:
    """One store per tenant: a query can only ever reach its own tenant's chunks, and corpus statistics
    (BM25 document frequencies) are never shared across tenants."""
    embedder: Embedder
    stores: dict[str, ChunkStore] = field(default_factory=dict)

    def store(self, tenant: str) -> ChunkStore:
        if tenant not in self.stores:
            self.stores[tenant] = ChunkStore(tenant, self.embedder)
        return self.stores[tenant]

    def add(self, chunks: list[Chunk]) -> int:
        by: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            by[c.meta.tenant].append(c)
        return sum(self.store(t).add(cs) for t, cs in sorted(by.items()))

    @property
    def size(self) -> int:
        return sum(len(s.chunks) for s in self.stores.values())
