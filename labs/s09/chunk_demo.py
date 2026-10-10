"""Lecture 9.2: chunk the procedures two ways, find the failures, embed, store, and use the API replay path.

    python -m labs.s09.chunk_demo
"""
from __future__ import annotations

import sys

from skyops.retrieval.chunking import find_failures, naive_chunks, structure_chunks
from skyops.retrieval.docs import load_corpus
from skyops.retrieval.embed import HashingEmbedder, replay_embedder
from skyops.retrieval.indexing import build
from skyops.retrieval.store import ChunkStore

NAIVE_SIZE = 500
EXAMPLE = "SOP-PAX-021@1.4"


def prefix_ablation() -> dict:
    """The same structure chunks with and without their "title > section" prefix, scored by the 9.6 eval."""
    from dataclasses import replace
    from skyops.retrieval.evals import evaluate
    from skyops.retrieval.search import Retriever
    from skyops.retrieval.store import TenantIndexes
    docs = load_corpus()
    bare = [replace(c, text=c.text.split("\n\n", 1)[1]) for d in docs for c in structure_chunks(d)]
    idx = TenantIndexes(HashingEmbedder())
    idx.add(bare)
    with_p = evaluate(Retriever(build()), k=5, mode="hybrid", rerank=True)
    without = evaluate(Retriever(idx), k=5, mode="hybrid", rerank=True)
    return {"with_recall": with_p.recall, "with_mrr": with_p.mrr, "without_recall": without.recall,
            "without_mrr": without.mrr}


def figures() -> dict:
    docs = load_corpus()
    out = {"docs": len(docs)}
    for name, fn in (("naive", lambda d: naive_chunks(d, NAIVE_SIZE)), ("structure", structure_chunks)):
        cut = orphan = n = 0
        for d in docs:
            cs = fn(d)
            f = find_failures(d, cs)
            n, cut, orphan = n + len(cs), cut + len(f.cut_structure), orphan + len(f.orphan_heading)
        out[name] = {"chunks": n, "cut": cut, "orphan": orphan}
    idx = build()
    out["index"] = {"chunks": idx.size, "tenants": len(idx.stores), "embedder": idx.embedder.id, "dim": idx.embedder.dim}
    return out


def main() -> None:
    f = figures()
    print(f"{f['docs']} procedure versions")
    for name in ("naive", "structure"):
        s = f[name]
        print(f"{name:9}: {s['chunks']} chunks, {s['cut']} tables or lists cut, {s['orphan']} orphaned headings")
    doc = next(d for d in load_corpus() if d.meta.key == EXAMPLE)
    bad = naive_chunks(doc, NAIVE_SIZE)[1]
    print(f"--- naive chunk {bad.chunk_id} starts mid-table:")
    print("    " + "\n    ".join(bad.text.splitlines()[:3]))
    good = next(c for c in structure_chunks(doc) if "assistance-codes" in c.sections)
    print(f"--- structure chunk {good.chunk_id}:")
    print("    " + "\n    ".join(good.text.splitlines()[:6]))
    i = f["index"]
    print(f"index: {i['chunks']} chunks in {i['tenants']} tenant stores, embedder {i['embedder']}")
    api = replay_embedder("fixtures/s09/embeddings_api.jsonl")
    v = api.embed(["hotel voucher limit without approval", "What does WCHC mean?"])
    print(f"API embedder, replayed: {api.id}, {len(v)} vectors of {len(v[0])} dimensions, "
          f"{api.input_tokens} input tokens (fixture synthetic: {api.transport.synthetic_records == 1})")
    try:
        s = ChunkStore("zz", HashingEmbedder())
        s.embedder = api
        s.add(structure_chunks(doc))
    except ValueError as e:
        print(f"mixing embedders: {e}")
    if "--ablation" in sys.argv:
        a = prefix_ablation()
        print(f"with the heading prefix:    recall@5 {a['with_recall']:.2f}, MRR {a['with_mrr']:.2f}")
        print(f"without the heading prefix: recall@5 {a['without_recall']:.2f}, MRR {a['without_mrr']:.2f}")


if __name__ == "__main__":
    main()
