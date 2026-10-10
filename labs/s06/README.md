# Section 6 labs: MCP on the wire (read only)

Section 6 has no code to write. You read real MCP traffic, check it against the rules of revision 2026-07-28, and make two decisions: how a gateway routes it, and whether a third-party server may join. Everything runs at tag `s05`. Time: about 1.5 hours. Live tokens: none. No keys, no network.

The recordings in `fixtures/mcp/` are **real traffic**: the official Python SDK v2 (`mcp` 2.3.0) server in `ops_server.py`, in memory. In `session.jsonl` and `third_party.jsonl` the SDK's own client sends the requests; in `errors.jsonl` and `legacy.jsonl` the requests are hand-built with httpx (each record's `meta.client` says which), and the server's answers are real. `python -m labs.s06.record` re-records them; only the sealed `requestState` values change.

| Lecture | Run | What to look for |
|---|---|---|
| 6.1, 6.2 | `python -m labs.s06.primitives` | the integration arithmetic; ten capabilities sorted into tools, resources and prompts |
| 6.3 | `python -m labs.s06.wire fixtures/mcp/session.jsonl` | `_meta` keys, `Mcp-Method` and `Mcp-Name`, `resultType`, `ttlMs` and `cacheScope` |
| 6.5 | `python -m labs.s06.wire fixtures/mcp/session.jsonl -n 5` and `-n 6` | a Multi Round-Trip Request and its retry |
| 6.5 | `python -m labs.s06.wire fixtures/mcp/errors.jsonl --check` | twelve exchanges, nine refused: which faults the request alone shows, which need the earlier exchange, which only the server sees; then a correct retry and a replayed one |
| 6.6 | `python -m labs.s06.gateway fixtures/mcp/session.jsonl --with-crew` | routing decisions from request headers alone; caching from response bodies; the legacy path |
| 6.7 | `python -m labs.s06.wire fixtures/mcp/legacy.jsonl --check`, then `MIGRATION.md` | what an older client sends, and the checklist |
| 6.9 | `python -m labs.s06.review_server` | provenance, pin, descriptions, scopes, tools; the decision |

`make labs-s06` runs all of them and the fourteen checks in `labs/s06/test_s06.py`. One of those checks re-records the session in memory and compares the wire with the fixture (headers, `_meta` keys, result types, cache hints, error codes), so an SDK upgrade that changes any of them fails.

## Exercises

1. **6.3** Annotate exchange 4 of `session.jsonl`: for every header and every `_meta` key, say who reads it (gateway, server, or both) and what breaks without it.
2. **6.5** In exchanges 5 and 6, state the two `id` values, who sealed `requestState`, and why the retry can land on a different replica. Then state the `ttlMs` and `cacheScope` of exchange 3 and exchange 7, and who may cache each.
3. **6.5** Before you run `--check` on `errors.jsonl`, read each refused request and name its fault. Which can you see in the request alone, which need the earlier exchange, and which one only the server can see? Then explain why exchange 12 is accepted but sends nothing.
4. **6.6** Add a third backend, `baggage-ops`, owning `trace_bag` and `skyops://bags/*`, with a read policy. Route a `trace_bag` call by hand.
5. **6.7** Fill in `MIGRATION.md` for a server you know, or for the legacy recording.
6. **6.9** Write the decision record for `flight-tracker-plus`. What would make you change it?
7. **Break it (6.5)** In `ops_server.py`, remove the idempotency check in `notify_passengers`, re-record, and look at exchange 12 of `errors.jsonl` again. What did the passengers receive?

The synthetic parts are labelled: `flight-tracker-plus` is fictional, written to contain the problems a real review finds; `2099-01-01` in `errors.jsonl` is a deliberately fictional protocol version.
