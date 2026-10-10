"""Lecture 9.8: back-of-envelope sizes for the three scenarios in docs/adr/ADR-009-vector-store.md.

    python -m labs.s09.store_sizing
Raw vector bytes only (4 bytes per float32 dimension, 2 for half precision). Indexes (HNSW graphs), metadata
and replicas come on top: measure them on your own data before you buy hardware.
"""
from __future__ import annotations

from skyops.retrieval.indexing import build

SCENARIOS = [
    # name, chunks, dimensions, tenants, what dominates
    ("A SkyOps procedures at one hub", 0, 512, 2, "access control and freshness; tiny corpus"),
    ("B a SaaS platform for 400 airlines", 10_000_000, 1536, 400, "tenant isolation, filtering, scale"),
    ("C keyword-heavy search across manuals", 2_000_000, 1024, 1, "exact codes, part numbers, existing search engine"),
]


def gib(n_bytes: float) -> float:
    return n_bytes / 2**30


def figures() -> list[dict]:
    a_chunks = build().size
    out = []
    for name, chunks, dim, tenants, why in SCENARIOS:
        chunks = chunks or a_chunks
        f32 = chunks * dim * 4
        out.append({"name": name, "chunks": chunks, "dim": dim, "tenants": tenants, "why": why,
                    "float32_gib": round(gib(f32), 2), "half_gib": round(gib(f32 / 2), 2),
                    "float32_kib": round(f32 / 1024)})
    return out


def main() -> None:
    for f in figures():
        size = (f"{f['float32_kib']} KiB as float32" if f["float32_gib"] < 0.01 else
                f"{f['float32_gib']} GiB as float32, {f['half_gib']} GiB as half precision")
        print(f"{f['name']}: {f['chunks']:,} chunks x {f['dim']} dims, {f['tenants']} tenant(s) -> {size}")
        print(f"    dominated by: {f['why']}")


if __name__ == "__main__":
    main()
