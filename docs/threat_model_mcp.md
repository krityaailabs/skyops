# Threat model: SkyOps MCP servers and gateway (one page, tag s07)

**Scope.** flight-ops and crew-ops MCP servers (`skyops/mcp/server.py`), the gateway (`skyops/mcp/gateway.py`), and the SkyOps host's MCP client (`skyops/mcp/client.py`). Protocol revision 2026-07-28, SDK `mcp` 2.3.0. Reviewed 2026-10-10.

**Assets.** Passenger contact channels (notify_passengers can reach every booked passenger), vouchers (money), bookings (personal data), crew rosters (personal data, kept behind the facade), tokens, the requestState sealing key.

**Trust boundaries.** (1) Model ↔ host: the model proposes, the host validates and approves. (2) Host ↔ gateway: a token whose audience is the gateway. (3) Gateway ↔ each server: a token minted for that server only. (4) Server ↔ legacy systems: the facade; legacy formats and faults never cross it. (5) Third-party servers: outside every boundary until reviewed (lecture 6.9).

| # | Threat | Where | Control in this tag | Test |
|---|---|---|---|---|
| 1 | Token passthrough: a server accepts, or forwards, a token issued for something else | gateway, servers | Servers verify `aud` = their own URI (401 otherwise); the gateway mints a new token per backend and never forwards the host's | `test_l74_wrong_audience_is_rejected`, hidden: gateway never forwards the host token |
| 2 | Confused deputy: a server uses its own authority for a caller who lacks it | servers, gateway | Per-operation scopes from the caller's token (ScopeGuard, `decide`); writes scoped per caller; no shared service credentials to downstream systems with broader rights than the caller | `test_l74_insufficient_scope_403`, `test_l79_*`; hidden: base64 Mcp-Name still scope-checked |
| 3 | Tool poisoning: instructions hidden in a description or result | host | Only reviewed servers, allow-listed tools, pinned listing digest (6.9); results reach the model as data shapes (`map_call_result`); injection defences in Section 19 | `test_l77_*` lock checks; Section 6 review tests |
| 4 | Silent contract change ("rug pull" or accident) | server releases | `tools.lock.json` + `check_lock` in CI: changes need a version bump; breaking changes need a major one | `test_l77_*` |
| 5 | Header/body disagreement to dodge policy | gateway, servers | Routing and scope checks use headers; the SDK rejects header/body mismatch (-32020) before any tool runs, but only for the current revision, so the guard and the gateway refuse any request with an older or missing MCP-Protocol-Version (-32022) | hidden: mismatched Mcp-Name never reaches a tool; older revisions refused |
| 6 | requestState tampering or replay | servers | SDK seals it (AES-GCM, TTL 5 min, request-bound); one key shared by replicas from the environment; irreversible writes still idempotent, scoped by client and user whether the call came direct or through the gateway | Section 6 recordings (replayed retry, `errors.jsonl` #12); hidden: gateway keeps the caller's client |
| 7 | State-handle hijacking | servers | No handles yet; when Section 13 adds them: bind to `client_id:sub`, unguessable, expiring | — (planned) |
| 8 | DNS rebinding against a local server | servers | Host allow-list on, no browser origins, loopback bind by default | `test_l710_unknown_host_rejected` |
| 9 | Credential exposure | everywhere | Tokens only in the Authorization header; no credentials in tool arguments; audit records names, not arguments, and resource patterns, not booking references; a deployed server refuses to start on the course test key | lint rule `secret_argument` (5.8); hidden: audit holds no booking reference |
| 10 | Over-broad tokens | authorization server, host | Baseline `flights:read`; each write has its own scope; step-up via 403 `insufficient_scope` challenges naming every needed scope | `test_l74_insufficient_scope_403` |

**Accepted for now.** Lab tokens are HS256 with a course key, accepted only with `--dev`; a deployment sets `SKYOPS_JWKS_URL` for the authorization server's asymmetric keys. No approval queue for writes (Section 18). No rate limiting beyond the gateway policy table (Section 21). Tool results are not yet scanned for injected instructions (Section 19).
