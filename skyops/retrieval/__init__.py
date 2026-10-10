"""Section 9: retrieval as a governed tool, over the synthetic SkyOps operating procedures (data/procedures).

    docs.py       parse a procedure: front matter (owner, version, tenant, roles, effective and withdrawn dates)
    text.py       tokens, a light stemmer, token estimates
    chunking.py   naive fixed-size chunks (and their failures) versus structure-aware chunks (9.2)
    embed.py      an Embedder interface: a local, deterministic hashing embedder; an API embedder with replay (9.2)
    store.py      a per-tenant chunk store: vectors and a BM25 index, with pre- or post-filtering (9.2, 9.3)
    search.py     keyword, vector, fusion (RRF), metadata filters, a reranker, cost per query (9.3)
    access.py     principals, document-level access, the leak a missing filter causes (9.4, 9.9)
    indexing.py   a change feed, a CDC indexer with a freshness window, and the naive sync it replaces (9.5)
    evals.py      recall@k and MRR on a labelled query set, separately from answers (9.6)
    agentic.py    bounded multi-step retrieval: a search budget enforced in code (9.7)
    tool.py       search_procedures as a Section 5 contract; role and tenant come from the caller (9.9)

Everything runs offline and deterministically. Nothing here calls a model.
"""
