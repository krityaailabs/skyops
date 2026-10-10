# Section 9 labs: production retrieval for agents

Start at tag `s09-start`. Twenty-nine Section 9 checks fail until you write the functions below (one passes from the start); `make fitness` names the next one. Most fail on lecture 9.2's function at first, because the index is built from its chunks; after 9.3 (which includes `can_read`), the search checks run. Time: about 4 hours. Live tokens: none (the model turns in 9.7 are a replay). Network: none.

The corpus is synthetic: fifteen versions of SkyOps operating procedures in `data/procedures/<tenant>/`, with owners, versions, roles, effective and withdrawn dates. One policy is revoked (SOP-GND-009, withdrawn 2026-10-01), one version is not yet in force (SOP-GND-014 v3.3, from 2026-11-01), and some procedures are restricted by role (MGR-007, SEC-003, CRW-012). The labelled query set is `data/procedures/eval_queries.jsonl`.

| Lecture | File | Implement | Check |
|---|---|---|---|
| 9.1 | `labs/s09/when_to_retrieve.py` | nothing: run it and do the decision exercise | — |
| 9.2 | `skyops/retrieval/chunking.py` | `structure_chunks` | `test_l92_*` |
| 9.3 | `skyops/retrieval/search.py`, `skyops/retrieval/access.py` | `matches`, `rrf_fuse`, `can_read` (every search calls it, before ranking) | `test_l93_*` |
| 9.4 | `skyops/retrieval/access.py`, `labs/s09/leak_demo.py` | nothing: read them, then break them (below) | `test_l94_*` |
| 9.5 | `skyops/retrieval/indexing.py`, `labs/s09/freshness_demo.py` | nothing: read them, then break them (below). The demo and check call the finished tool: they pass after 9.9 (or run them at tag `s09`) | `test_l95_*` |
| 9.6 | `skyops/retrieval/evals.py` | `recall_at_k`, `reciprocal_rank` | `test_l96_*` |
| 9.7 | `skyops/retrieval/agentic.py`, `labs/s09/agentic_demo.py` | nothing: run it, then change the budget. It wraps the finished tool: run it at tag `s09`, or after 9.9 | `test_l97_*` |
| 9.8 | `docs/adr/ADR-009-vector-store.md`, `labs/s09/store_sizing.py` | write the decision for your own system | — |
| 9.9 | `skyops/retrieval/tool.py` | `search_procedures`; then add a revoked-policy case of your own (below) | `test_l99_*` |

Run things as you go: `make labs-s09` runs every demo, or one at a time: `python -m labs.s09.chunk_demo`, `search_demo`, `leak_demo`, `freshness_demo`, `eval_demo`, `agentic_demo`, `store_sizing`, `procedures_demo`.

About the replays: `fixtures/s09/agentic_*.jsonl` are scripted model turns (synthetic); the searches, access filter and budget under them are real. `fixtures/s09/embeddings_api.jsonl` is a synthetic `/v1/embeddings` exchange (vectors from the local embedder) that exercises the API replay path; re-record it with a key before you measure a real model. Rebuild both with `python labs/s09/build_fixtures_s09.py`.

## Hint ladder

**9.2 `structure_chunks`**
1. Start from `blocks(doc)` and drop the headings: their text goes in the prefix, not the body.
2. Keep a list of blocks for the current chunk. Flush it when the next block is in another section, or when adding it would take the joined text over `max_tokens`. A single block is never split, however long.
3. The prefix is `f"{title} > {heading}"`, or just the title when `heading` is empty; then `"\n\n"` and the blocks joined with `"\n\n"`. `n` counts chunks across the whole document.

**9.3 `matches`, `rrf_fuse`, `can_read`**
1. `matches`: three tests, all must pass. Note the strict `<` on the withdrawn date: on the day a policy is withdrawn, it's already gone.
2. `rrf_fuse`: `enumerate(ranking, 1)` gives ranks from one. Add `1 / (k + rank)` per list. Sort by `(-score, id)` so ties are stable.
3. `can_read` is two conditions joined by `and`: the same tenant, and a non-empty role intersection. A principal with no roles reads nothing.

**9.6 `recall_at_k`, `reciprocal_rank`**
1. `ranked[i]` is a set: the labels result `i + 1` satisfies. Union the first `k` sets.
2. Raise `ValueError` for an empty `relevant`: recall has no meaning there.
3. `reciprocal_rank` returns at the first set that meets `relevant`.

**9.9 `search_procedures`**
1. Check `fresh` first and fail closed: raise `ToolError("retryable", "index_stale", ...)`.
2. Search with `mode="hybrid", rerank=True` and `Filters(as_of=as_of)`; the principal is the one you're given, never anything from `args`.
3. Rows are `citation(h.chunk)`; shape them with `shape_result(rows, FIELDS, page_size=args.k, max_tokens=MAX_RESULT_TOKENS)`, then add `as_of` and `note`.

Then add a case of your own to `test_l99_revoked_policy_is_never_returned`: another way staff might ask about the withdrawn voucher rule. Every withdrawal should arrive with such a case.

## Break it

- 9.4: in `Retriever.search`, make `allow` ignore `can_read`. Run `python -m labs.s09.leak_demo` and `test_l94_missing_filter_leaks_a_restricted_procedure`. What does an agent now get for "goodwill authority limit"?
- 9.5: in `CdcIndexer.apply`, skip `"withdraw"` events. Which check fails, and how long is the revoked policy searchable now?
- 9.7: run `python -m labs.s09.agentic_demo --budget 10,5` (ten searches, five without progress). How many searches does the runaway scenario make now, and what did they cost in evidence tokens? (The replay is matched by sequence here, because the tool results change; the scripted model still proposes only six searches. A live model has no script.)
- 9.9: add a `role` argument to `SearchProceduresArgs` and use it. Which check fails, and why is that check right?
- 9.9: in `make_search_contract`, change the default `as_of` to `dt.date(2026, 9, 30)`. Which checks fail? Change it back.

Hidden checks (instructor repository) test behaviours the docstrings already state.
