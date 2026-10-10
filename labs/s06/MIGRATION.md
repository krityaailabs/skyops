# Migration checklist: an MCP server or client from 2025-11-25 (or earlier) to 2026-07-28

Verified against the 2026-07-28 specification and changelog on 2026-10-10. Recheck the changelog before you act: this revision is recent, and SDKs move faster than courses.

## Decide the era first

- [ ] Modern only, or dual-era? A dual-era server answers `initialize` for old clients and per-request `_meta` for new ones, on the same endpoint. The SDK v2 `MCPServer` is dual-era by default (`legacy.jsonl` was recorded from it).
- [ ] List the clients you must keep serving, and for how long. Deprecated features still work; under the feature lifecycle policy, Roots, Sampling, Logging and Dynamic Client Registration can't be removed before the first revision released on or after 2027-07-28, while the HTTP+SSE transport (deprecated since 2025-03-26) can go sooner. Check the deprecated-features registry for current dates. Removed features are gone now.

## Protocol: removed, so code must change

- [ ] No `initialize` / `notifications/initialized` handshake for modern clients. Every request carries `_meta`: `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities` (required), `clientInfo` (should).
- [ ] Implement `server/discover` (servers must). Clients may call it, or call anything and handle `UnsupportedProtocolVersion` (-32022).
- [ ] No protocol sessions, no `Mcp-Session-Id`. Anything that must span calls becomes an explicit, server-minted handle passed as a tool argument, checked for authorisation on every call, with a stated lifetime.
- [ ] Server-to-client requests (elicitation, sampling, roots) move inside results: `InputRequiredResult` with `inputRequests` and/or a sealed `requestState` (each optional, at least one present); the client retries with a new id. Treat `requestState` as attacker-controlled: integrity-protect it, bind it to the principal and the request, give it a short expiry.
- [ ] `ping` and `logging/setLevel` are gone; send `io.modelcontextprotocol/logLevel` in `_meta` per request.
- [ ] The standalone GET stream and `resources/subscribe` are replaced by `subscriptions/listen`.
- [ ] SSE resumability with `Last-Event-ID` is gone: if a response stream breaks, the request is lost and the client re-issues it with a new id. Cancellation on Streamable HTTP is closing the response stream; `notifications/cancelled` is used only on stdio.
- [ ] `notifications/roots/list_changed` is removed. URL elicitation's `notifications/elicitation/complete` and `elicitationId` are removed, and error -32042 is retired.
- [ ] Servers must not send `notifications/message` for a request that didn't set `io.modelcontextprotocol/logLevel`.
- [ ] Tasks moved out of the core into the `io.modelcontextprotocol/tasks` extension; negotiate it as an extension.
- [ ] Resource not found is now -32602, not -32002 (clients should still accept -32002 from older servers).

## Protocol: added, so code must emit or check it

- [ ] Every result has `resultType` (`complete` or `input_required`).
- [ ] HTTP headers on every POST: `MCP-Protocol-Version` (matching `_meta`), `Mcp-Method`, and `Mcp-Name` for `tools/call`, `resources/read` and `prompts/get`. Servers reject mismatches with 400 and `HeaderMismatch` (-32020).
- [ ] Tool parameters a gateway should route on get `x-mcp-header`; clients mirror them as `Mcp-Param-{Name}`. Never mark secrets or personal data.
- [ ] Cacheable results carry `ttlMs` and `cacheScope`: `server/discover`, the four list methods, `resources/read`. Per-user data is `private`.
- [ ] `tools/list` in a deterministic order, and never varying per connection (it may vary by the authorisation on the request).

## Deprecated, so plan the exit

- [ ] Roots, Sampling and Logging features; the HTTP+SSE transport; Dynamic Client Registration (move to Client ID Metadata Documents); `includeContext` values other than `none`.

## Authorisation (Section 7.4 in detail)

- [ ] Clients compare a returned `iss` parameter with the expected issuer (RFC 9207), and reject a response without `iss` when the authorisation server advertises that it sends one. Keep credentials bound to the issuer that minted them.
- [ ] Clients that still use Dynamic Client Registration send `application_type` (SEP-837).

## SDK

- [ ] Python: `mcp` 2.x. `FastMCP` is now `MCPServer`: `from mcp.server import MCPServer` (or `from mcp.server.mcpserver import MCPServer`; same class). The old import fails with a message that points at the migration guide. Pin the version in your lock file.
- [ ] SDK 2.3.0 detail: on HTTP its server picks the era from the `MCP-Protocol-Version` header, so modern clients must send it (they must anyway).
- [ ] Re-run your contract tests and a recorded session after the upgrade; compare the wire, not just the results.
