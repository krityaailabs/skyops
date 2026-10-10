"""Lecture 9.4: the leaks, shown on the real index, and the controls that stop them.

    python -m labs.s09.leak_demo
"""
from __future__ import annotations

from skyops.retrieval.access import DIRECTORY
from skyops.retrieval.chunking import chunk_corpus
from skyops.retrieval.docs import load_corpus
from skyops.retrieval.embed import HashingEmbedder
from skyops.retrieval.indexing import build
from skyops.retrieval.search import Retriever
from skyops.retrieval.store import ChunkStore
from skyops.retrieval.text import tokens

GOODWILL = "goodwill authority limit after a delay"
REBOOK = "rebooking rate on partner airlines"


def shared_store() -> ChunkStore:
    s = ChunkStore("*", HashingEmbedder())
    s.add(chunk_corpus(load_corpus()))
    return s


def docs_of(res) -> list[str]:
    return list(dict.fromkeys(h.chunk.meta.doc_id for h in res.hits))


def figures() -> dict:
    idx = build()
    safe, open_ = Retriever(idx), Retriever(idx, enforce_access=False)
    shared = Retriever(idx, enforce_access=False, shared=shared_store())
    agent, yy = DIRECTORY["agent-anna"], DIRECTORY["yy-dm-yara"]
    sec = DIRECTORY["sec-sam"]
    pre = safe.search("delay care passengers", sec, k=5, prefilter=True)
    post = safe.search("delay care passengers", sec, k=5, prefilter=False)
    term = tokens("interline")[0]
    zz_df = idx.store("zz").bm25.df.get(term, 0)
    shared_df = shared.shared.bm25.df.get(term, 0)
    return {"safe": docs_of(safe.search(GOODWILL, agent, k=3)),
            "missing": docs_of(open_.search(GOODWILL, agent, k=3)),
            "shared": docs_of(shared.search(REBOOK, agent, k=3)),
            "tenant_store": docs_of(safe.search(REBOOK, agent, k=3)),
            "yy_own": docs_of(safe.search(REBOOK, yy, k=3)),
            "pre": len(pre.hits), "post": len(post.hits), "zz_df": zz_df, "shared_df": shared_df}


def main() -> None:
    f = figures()
    print(f"agent-anna, {GOODWILL!r}")
    print(f"  access filter on:  {f['safe']}")
    print(f"  filter missing:    {f['missing']}   <- MGR-007 is for duty managers only")
    print(f"agent-anna (ZZ), {REBOOK!r}")
    print(f"  shared index, no tenant filter: {f['shared']}   <- YY Connect's confidential terms")
    print(f"  per-tenant store:               {f['tenant_store']}")
    print(f"  yy-dm-yara, own tenant:         {f['yy_own']}")
    print(f"sec-sam (security only), k=5: pre-filter returns {f['pre']}, post-filter returns {f['post']}")
    print(f"document frequency of 'interline': ZZ store {f['zz_df']}, shared index {f['shared_df']}")


if __name__ == "__main__":
    main()
