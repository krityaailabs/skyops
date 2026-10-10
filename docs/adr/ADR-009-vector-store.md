# ADR-009: Where SkyOps keeps its retrieval index

Status: accepted for scenario A (Section 9); B and C recorded as worked examples. Date: 2026-10-09. Owner: the SkyOps architect.

## Context

`search_procedures` needs keyword search, vector search, metadata filters that run before ranking (tenant, roles, effective and withdrawn dates), deletes that take effect within the five-minute freshness window, and an eval run on every change. Section 9 runs it in memory (`skyops/retrieval/store.py`), with exact search. This record decides what replaces that in production, for three scenarios. Sizes come from `python -m labs.s09.store_sizing` (raw vectors only; indexes, metadata and replicas come on top).

## Options

1. **A vector extension in the database you already run** (PostgreSQL with pgvector): HNSW or IVFFlat indexes; filters are SQL `WHERE` clauses next to the documents' own tables; transactions make "delete the old version, insert the new one" atomic. With approximate indexes, filtering is applied after the index scan, so a narrow filter can return fewer rows than asked for; pgvector 0.8.0 added iterative index scans (`hnsw.iterative_scan`, `ivfflat.iterative_scan`) to keep scanning until enough rows pass or a limit is reached (`hnsw.max_scan_tuples`, `ivfflat.max_probes`), so very narrow filters still need a filtered recall eval. Indexes cover `vector` up to 2,000 dimensions and `halfvec` up to 4,000.
2. **A dedicated vector database**: built for large approximate indexes, filtering during the index search, namespaces or collections per tenant, and horizontal scaling. Another system to run, secure, back up and keep consistent with the source of truth.
3. **A search engine with vector support** (OpenSearch, Elasticsearch): mature BM25, analysers for codes and languages, vector fields, and fusion built in (both offer reciprocal rank fusion for hybrid queries; OpenSearch from 2.19, Apache 2.0; in Elasticsearch, check which subscription tier includes it). Heavier to operate than an extension; strongest when keyword relevance dominates or a cluster already exists.

## Trade-offs

| | Extension in your database | Dedicated vector store | Search engine with vectors |
|---|---|---|---|
| Reliability | your database's backups and HA | another system's | another cluster's |
| Accuracy risk | approximate index plus filter: check recall with filters on | depends on filtered-search design | strong BM25; vectors as good as the model |
| Latency | fine at small and medium scale | designed for large scale | good; tune shards |
| Cost per task | lowest when the database exists | separate bill, scales well | cluster cost |
| Complexity | one system | two systems to keep consistent | two systems |
| Security, blast radius | row-level security and roles you already audit | its own access model to learn | its own security model |
| Observability | existing database metrics | new dashboards | existing if the cluster exists |
| Failure behaviour | fails with the database | independent failure; stale replica risk | independent failure |
| Best fit | thousands to millions of chunks, strict filters, transactions | tens of millions or more, many tenants | keyword-heavy, existing cluster |

No option wins everywhere. If transactions and governance dominate, the extension is attractive. If scale across many tenants dominates, a dedicated store may make more sense. If exact codes and existing search skills dominate, the search engine does.

## Decisions

- **Scenario A, SkyOps procedures at one hub** (tens of chunks here, thousands at a real station): PostgreSQL with pgvector, next to the documents table. Deletes and version swaps are transactions; access filters are SQL. Exact search is affordable at this size; add an HNSW index only when latency needs it, then re-run the recall eval with filters on.
- **Scenario B, a platform for 400 airlines, ten million chunks**: about 57 GiB of float32 vectors before indexes. Isolation per tenant (a namespace, collection or partition per tenant), filters applied during the search, not after. A dedicated store or partitioned PostgreSQL, decided by a load test on real filters. Platform-scale RAG design is covered in the sibling course on AI system design.
- **Scenario C, keyword-heavy search across maintenance manuals with an existing search cluster**: the search engine, with vector fields and its built-in fusion.

## Consequences

- The `Retriever` interface stays; only `ChunkStore` changes. The Section 9 evals and the revoked-policy regression test run unchanged against the new store, and must pass before the switch.
- Whatever the store, the embedding model's id and dimensions are recorded with the index; changing models means re-embedding everything.

## Revisit when

Chunks grow past a few million, a second hub or a new tenant joins, filtered recall at k drops below the eval's threshold, or p95 search latency exceeds its budget.
