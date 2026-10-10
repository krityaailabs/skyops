# Section 7 labs: build MCP servers and clients

Start at tag `s07-start`. Ten Section 7 checks fail until you write the functions below (one passes from the start); `make fitness` names the next one. Time: about 4 hours. Live tokens: none (the one model conversation, in 7.5, is a replay). Network: none, except `npx` for the optional MCP Inspector in 7.1.

Everything uses the official Python SDK v2 (`mcp` 2.3.0, pinned in `requirements.lock`) and protocol revision 2026-07-28. Older tutorials use `from mcp.server.fastmcp import FastMCP`: that's SDK v1.

| Lecture | File | Implement | Check |
|---|---|---|---|
| 7.1 | `labs/s07/first_server.py` | `build` | `test_l71_*` |
| 7.2 | `skyops/mcp/adapter.py` | `annotations_for`, `to_call_result` | `test_l72_*` |
| 7.3 | `skyops/mcp/server.py` | `booking_resource`, `disruption_brief` | `test_l73_*` |
| 7.4 | `skyops/mcp/auth.py`, `ScopeGuard` in `server.py` | nothing: read them, then break them (below) | `test_l74_*` |
| 7.5 | `skyops/mcp/client.py` | `map_call_result` | `test_l75_*` |
| 7.6 | `skyops/mcp/facade.py` | nothing: read it against `skyops/sim/crew_soap.py` | — |
| 7.7 | `skyops/mcp/versioning.py` | `check_lock` | `test_l77_*` |
| 7.9 | `skyops/mcp/gateway.py` | `decide` | `test_l79_*` |
| 7.10 | `labs/s07/deploy_local.sh`, `deploy/Dockerfile.mcp` | nothing: run `make deploy-local` | `test_l710_*` |
| 7.11 | `docs/threat_model_mcp.md` | review it; add one threat you'd add for your own server | — |

Run things as you go: `python -m labs.s07.first_server`, `python -m labs.s07.agent_over_mcp`, `python -m labs.s07.gateway_demo`, `python -m skyops.mcp.versioning`, `make deploy-local`.

Optional, 7.1: the official MCP Inspector (Node.js 22.19 or later):
`npx @modelcontextprotocol/inspector@2.10.1 --cli --config labs/s07/inspector.json --server skyops-first --method tools/list`

## Hint ladder

**7.1 `build`**
1. `srv = MCPServer("skyops-first", version="0.1.0")`, then a function decorated with `@srv.tool(structured_output=True)`.
2. The docstring is the description; `Annotated[str, Field(description=...)]` puts descriptions in the schema.
3. Raise `ToolError` (from `mcp.server.mcpserver.exceptions`) for "not found": any other exception is a crash, and the model only sees "Error executing tool".

**7.2 `annotations_for`, `to_call_result`**
1. Three side-effect classes, three rows of the table in the docstring. `idempotent_hint` for writes is `bool(contract.key_fields)`.
2. `to_call_result`: round-trip through `json.dumps(..., default=str)` first, so dates become text.
3. The error text is exactly `"<code>: <message>"`; the success text is compact JSON with sorted keys.

**7.3 `booking_resource`, `disruption_brief`**
1. `bookings.get(ref)`; unknown: `ResourceNotFoundError` (from `mcp.server.mcpserver.exceptions`), which the SDK answers with -32602. A `ValueError` would come back as -32603, a crash. Return `json.dumps(..., sort_keys=True)`.
2. The prompt text must contain the flight number, both tool names, "three lines" and "not message passengers".

**7.5 `map_call_result`**
1. Four cases, in the docstring's order. Check `isinstance(sc, dict)` before looking for `"error"`.
2. Join error text with spaces and success text with newlines.

**7.7 `check_lock`**
1. Removed, added and changed come from set arithmetic on tool names, and `!=` on tool dicts.
2. "needed" from the table; "got" from comparing `semver(version)` with `semver(lock["version"])`.
3. Problems in the docstring's order; removed and breaking only when the bump wasn't major.

**7.9 `decide`**
1. The order matters: version, method, discover and lists, named methods, everything else.
2. `fnmatch(name, pattern)` handles `skyops://bookings/*`.

## Break it (7.4)

- Set `validate_token_resource=False` in `server.py` and change `JwtVerifier` to skip the audience. Run `test_l74_wrong_audience_is_rejected`. What does a token minted for the gateway now do?
- In `ScopeGuard`, return the first missing scope instead of all of them. Which spec rule does that break, and what does it cost the user?

Hidden checks (instructor repository) test behaviours the docstrings already state.
