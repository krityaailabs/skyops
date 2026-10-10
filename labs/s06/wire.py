"""Lecture 6.3: the replay inspector. Read MCP exchanges on the wire, field by field, and check them.

The recordings in fixtures/mcp/ are real traffic between the official Python SDK's client and server
(see record.py). The inspector shows each HTTP request and response, says what each header and _meta key is
for, and checks the rules of revision 2026-07-28 that a gateway or a server would check.

    python -m labs.s06.wire fixtures/mcp/session.jsonl            every exchange, annotated
    python -m labs.s06.wire fixtures/mcp/session.jsonl -n 4       one exchange
    python -m labs.s06.wire fixtures/mcp/session.jsonl --brief    one line each
    python -m labs.s06.wire fixtures/mcp/errors.jsonl --check     rule findings, client side and server side

Read only: nothing here needs changing in Section 6.
"""
from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CURRENT = "2026-07-28"
LEGACY = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
P = "io.modelcontextprotocol/"
REQUIRED_META = (P + "protocolVersion", P + "clientCapabilities")
NAMED = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri"}       # methods that need Mcp-Name
CACHEABLE = {"server/discover", "tools/list", "prompts/list", "resources/list", "resources/templates/list",
             "resources/read"}
INPUT_REQUIRED_OK = {"tools/call", "resources/read", "prompts/get"}
REMOVED = {"ping": "removed in 2026-07-28", "notifications/roots/list_changed": "removed in 2026-07-28",
           "notifications/elicitation/complete": "removed in 2026-07-28 (URL elicitation completion)", "logging/setLevel": "removed: send io.modelcontextprotocol/logLevel per request",
           "resources/subscribe": "replaced by subscriptions/listen", "resources/unsubscribe": "replaced by subscriptions/listen",
           "notifications/initialized": "the handshake is gone", "initialize": "the handshake is gone: send _meta on every request"}
ERRORS = {-32020: "HeaderMismatch", -32021: "MissingRequiredClientCapability", -32022: "UnsupportedProtocolVersion",
          -32601: "Method not found", -32602: "Invalid params", -32600: "Invalid request", -32603: "Internal error",
          -32002: "Resource not found (2025-11-25 and earlier)"}
EXPECTED_STATUS = {-32020: 400, -32021: 400, -32022: 400, -32601: 404, -32602: 400}


@dataclass
class Exchange:
    n: int
    label: str
    req_headers: dict[str, str]
    body: dict[str, Any] | None
    status: int
    resp_headers: dict[str, str]
    resp: dict[str, Any] | None
    note: str = ""
    server: str = ""

    @property
    def method(self) -> str:
        return (self.body or {}).get("method", "")

    @property
    def params(self) -> dict[str, Any]:
        return (self.body or {}).get("params") or {}

    @property
    def meta(self) -> dict[str, Any]:
        return self.params.get("_meta") or {}

    @property
    def result(self) -> dict[str, Any] | None:
        return (self.resp or {}).get("result")

    @property
    def error(self) -> dict[str, Any] | None:
        return (self.resp or {}).get("error")

    @property
    def target(self) -> str:
        key = NAMED.get(self.method)
        return str(self.params.get(key, "")) if key else ""


@dataclass
class Finding:
    n: int
    side: str           # client: in this request alone · history: needs earlier exchanges ·
                        # server: only in the answer · info: worth knowing, not a fault
    code: str
    detail: str


def load(path: str | Path) -> list[Exchange]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(Exchange(r["n"], r["label"], r["request"]["headers"], r["request"]["body"], r["response"]["status"],
                            r["response"]["headers"], r["response"]["body"], r["meta"].get("note", ""),
                            r["meta"].get("server", "")))
    return out


def decode_header(v: str) -> str:
    """Undo the base64 sentinel: =?base64?…?= carries values that aren't plain ASCII."""
    if v.startswith("=?base64?") and v.endswith("?="):
        return base64.b64decode(v[9:-2]).decode()
    return v


def header_params(tools_list: list[dict]) -> dict[str, dict[str, list[str]]]:
    """tool name -> {header name -> property path} for every x-mcp-header in a tools/list result."""
    out: dict[str, dict[str, list[str]]] = {}

    def walk(schema: dict, path: list[str], acc: dict[str, list[str]]) -> None:
        for prop, sub in (schema.get("properties") or {}).items():
            if "x-mcp-header" in sub:
                acc[sub["x-mcp-header"].lower()] = path + [prop]
            if "properties" in sub:                     # nested objects, with or without "type": "object"
                walk(sub, path + [prop], acc)

    for t in tools_list:
        acc: dict[str, list[str]] = {}
        walk(t.get("inputSchema") or {}, [], acc)
        if acc:
            out[t["name"]] = acc
    return out


def known_schemas(exchanges: list[Exchange], folder: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """Header annotations from every tools/list the inspector can see: this file first, then its folder."""
    seen: dict[str, dict[str, list[str]]] = {}
    pool = list(exchanges)
    if folder:
        for f in sorted(folder.glob("*.jsonl")):
            pool += load(f)
    for ex in pool:
        if ex.method == "tools/list" and ex.result and ex.server in ("", exchanges[0].server if exchanges else ""):
            for k, v in header_params(ex.result.get("tools", [])).items():
                seen.setdefault(k, v)
    return seen


def _dig(d: dict, path: list[str]) -> Any:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _as_header(v: Any) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v)


# ----------------------------------------------------------------------------------------------- checks

def check(exchanges: list[Exchange], schemas: dict[str, dict[str, list[str]]] | None = None) -> list[Finding]:
    """The rules a gateway (client side) or a server (server side) applies, in the order a request meets them."""
    schemas = schemas if schemas is not None else known_schemas(exchanges)
    found: list[Finding] = []
    issued: list[tuple[Exchange, dict]] = []          # earlier input_required results: (exchange, result)
    used: list[tuple[int, int]] = []                   # (exchange that issued a state, exchange that used it)
    for ex in exchanges:
        f = lambda side, code, detail: found.append(Finding(ex.n, side, code, detail))
        h = {k.lower(): v for k, v in ex.req_headers.items()}
        m = ex.method
        if not ex.body:
            continue
        # transport-era signals first: a legacy request is not "wrong", it is a different protocol era
        if "mcp-session-id" in h:
            f("client", "legacy_session", "Mcp-Session-Id header: sessions were removed in 2026-07-28")
        if "last-event-id" in h:
            f("client", "legacy_resume", "Last-Event-ID header: streams are not resumable in 2026-07-28")
        if m in REMOVED:
            f("client", "removed_method", f"{m}: {REMOVED[m]}")
        legacy = m in ("initialize", "notifications/initialized") or "mcp-session-id" in h
        if not legacy and "id" in ex.body:
            _client_rules(ex, h, m, schemas, issued, used, f)
        _server_rules(ex, m, issued, f)
    return found


def _client_rules(ex: Exchange, h: dict, m: str, schemas: dict, issued: list, used: list, f) -> None:
    version = ex.meta.get(P + "protocolVersion")
    for key in REQUIRED_META:
        if key not in ex.meta:
            f("client", "missing_meta", f"_meta has no {key.removeprefix(P)} (required on every request)")
    if P + "clientInfo" not in ex.meta:
        f("client", "no_client_info", "_meta has no clientInfo (SHOULD be sent)")
    for k, v in h.items():
        if k.startswith("mcp") and any(not (32 <= ord(ch) <= 126 or ch == "\t") for ch in v):
            f("client", "invalid_header_value", f"{k}: not plain ASCII; use the =?base64?…?= form")
    hv = h.get("mcp-protocol-version")
    if hv is None:
        f("client", "missing_header", "MCP-Protocol-Version")
    elif version and hv != version:
        f("client", "header_mismatch", f"MCP-Protocol-Version {hv} but _meta protocolVersion {version}")
    if version and version != CURRENT:
        f("client", "unknown_version" if version not in LEGACY else "legacy_version",
          f"protocolVersion {version}: this inspector speaks {CURRENT}")
    if h.get("mcp-method") is None:
        f("client", "missing_header", "Mcp-Method")
    elif h["mcp-method"] != m:
        f("client", "header_mismatch", f"Mcp-Method {h['mcp-method']} but method {m}")
    if m in NAMED:
        want, got = ex.target, h.get("mcp-name")
        if got is None:
            f("client", "missing_header", f"Mcp-Name (required for {m})")
        elif decode_header(got) != want:
            f("client", "header_mismatch", f"Mcp-Name {decode_header(got)} but params.{NAMED[m]} {want}")
    if m == "tools/call":
        args = ex.params.get("arguments") or {}
        declared = schemas.get(ex.target, {})
        for hname, path in declared.items():
            val, got = _dig(args, path), h.get(f"mcp-param-{hname}")
            if val is None and got is not None:
                f("client", "header_mismatch", f"Mcp-Param-{hname.title()} sent but {'.'.join(path)} is absent")
            elif val is not None and got is None:
                f("client", "missing_param_header", f"Mcp-Param-{hname.title()} (x-mcp-header on {'.'.join(path)})")
            elif val is not None and decode_header(got) != _as_header(val):
                f("client", "header_mismatch", f"Mcp-Param-{hname.title()} {decode_header(got)} but "
                                               f"{'.'.join(path)} {val}")
        if not declared and any(k.startswith("mcp-param-") for k in h):
            f("info", "unknown_param_header", "Mcp-Param-* for a tool with no x-mcp-header seen (forwarded, ignored)")
    if "requestState" in ex.params or "inputResponses" in ex.params:
        # these need the earlier exchanges: one request alone can't show them; a recording, or the server, can
        same_call = [(e, r) for e, r in issued if e.method == m and e.target == ex.target]
        exact = [(e, r) for e, r in same_call if r.get("requestState") == ex.params.get("requestState")]
        if not exact:
            f("history", "state_not_issued", "requestState matches no input_required result seen for this call "
                                             "(modified, or from another request)")
        for e, r in exact:
            if (e.params.get("arguments") != ex.params.get("arguments")) and m == "tools/call":
                f("history", "state_on_other_request", f"requestState issued by #{e.n} for different arguments")
            if e.body.get("id") == ex.body.get("id"):
                f("history", "retry_reuses_id", f"the retry reuses id {ex.body.get('id')} from #{e.n}")
            missing = set(r.get("inputRequests") or {}) - set(ex.params.get("inputResponses") or {})
            if missing:
                f("history", "unanswered_input", f"no inputResponses for {sorted(missing)}")
            done = [u for u in used if u[0] == e.n]
            if done:
                f("history", "state_replayed", f"requestState from #{e.n} already used by #{done[0][1]}; only the "
                                               "tool's own idempotency key stops a second action")
        if exact and not ex.error and not any(u[0] == exact[0][0].n for u in used):
            used.append((exact[0][0].n, ex.n))


def _server_rules(ex: Exchange, m: str, issued: list, f) -> None:
    res, err = ex.result, ex.error
    if res is not None:
        rt = res.get("resultType")
        if rt is None:
            f("server", "no_result_type", "result has no resultType (a legacy-era answer; clients read it as complete)")
        elif rt not in ("complete", "input_required"):
            f("server", "unknown_result_type", f"resultType {rt!r}")
        if rt == "input_required":
            if m not in INPUT_REQUIRED_OK:
                f("server", "input_required_not_allowed", f"{m} may not return input_required")
            if not (res.get("inputRequests") or res.get("requestState")):
                f("server", "empty_input_required", "neither inputRequests nor requestState")
            issued.append((ex, res))
        if rt == "complete" and m in CACHEABLE and ("ttlMs" not in res or "cacheScope" not in res):
            f("server", "missing_cache_hints", f"{m} result without ttlMs and cacheScope")
        if "cacheScope" in res and res["cacheScope"] not in ("public", "private"):
            f("server", "bad_cache_scope", f"cacheScope {res['cacheScope']!r}")
        if isinstance(res.get("ttlMs"), int) and res["ttlMs"] < 0:
            f("server", "bad_ttl", f"ttlMs {res['ttlMs']} (must be >= 0)")
    if err is not None:
        code = err.get("code")
        f("server", "error", f"{code} {ERRORS.get(code, 'error')}: {err.get('message', '')}")
        if code in EXPECTED_STATUS and ex.status != EXPECTED_STATUS[code]:
            f("server", "wrong_status", f"HTTP {ex.status} for {code}; expected {EXPECTED_STATUS[code]}")


def summary(exs: list[Exchange], found: list[Finding]) -> dict[str, int]:
    """How the server's rejections split by who could have seen the fault first."""
    rejected = {ex.n for ex in exs if ex.error}
    alone = {x.n for x in found if x.side == "client"} & rejected
    hist = ({x.n for x in found if x.side == "history"} & rejected) - alone
    return {"exchanges": len(exs), "rejected": len(rejected), "request_alone": len(alone), "with_history": len(hist),
            "server_only": len(rejected - alone - hist)}


# ----------------------------------------------------------------------------------------------- display

def _short(v: Any, width: int = 70) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else " ".join(v.split())
    return s if len(s) <= width else s[: width - 1] + "…"


def _caps(c: dict) -> str:
    return "{" + ", ".join(sorted(c)) + "}" if c else "{} (none)"


def _ttl(ms: int) -> str:
    if ms == 0:
        return "0 (stale at once)"
    s = ms // 1000
    return f"{ms} ms = {s // 3600} h" if s >= 3600 and s % 3600 == 0 else (
        f"{ms} ms = {s // 60} min" if s >= 60 and s % 60 == 0 else f"{ms} ms = {s} s")


def headline(ex: Exchange) -> str:
    target = f" {ex.target}" if ex.target else ""
    if ex.error:
        outcome = f"{ex.status} error {ex.error.get('code')} {ERRORS.get(ex.error.get('code'), '')}".rstrip()
    elif ex.result is not None:
        outcome = f"{ex.status} resultType={ex.result.get('resultType', '(none)')}"
    else:
        outcome = f"{ex.status} (no body)"
    return f"#{ex.n:<2} {ex.method}{target}  →  {outcome}   [{ex.label}]"


def describe(ex: Exchange, schemas: dict[str, dict[str, list[str]]] | None = None) -> list[str]:
    schemas = schemas or {}
    out = [headline(ex)]
    h = {k.lower(): v for k, v in ex.req_headers.items()}
    out.append(" → POST /mcp")
    meta_v = ex.meta.get(P + "protocolVersion")
    rows = [("MCP-Protocol-Version", h.get("mcp-protocol-version"),
             "matches _meta protocolVersion" if meta_v and h.get("mcp-protocol-version") == meta_v else
             ("no _meta: legacy request" if not meta_v else f"_meta says {meta_v}: MISMATCH")),
            ("Mcp-Method", h.get("mcp-method"), "matches method" if h.get("mcp-method") == ex.method else
             f"body says {ex.method}")]
    if ex.method in NAMED or "mcp-name" in h:
        got = h.get("mcp-name")
        rows.append(("Mcp-Name", got, f"matches params.{NAMED.get(ex.method, 'name')}" if got and decode_header(got) == ex.target
                     else f"body says {ex.target or '(none)'}"))
    for k, v in h.items():
        if k.startswith("mcp-param-"):
            path = schemas.get(ex.target, {}).get(k.removeprefix("mcp-param-"))
            val = _dig(ex.params.get("arguments") or {}, path) if path else None
            rows.append((f"Mcp-Param-{k.removeprefix('mcp-param-').title()}", v,
                         f"matches arguments.{'.'.join(path)} (x-mcp-header)" if path and _as_header(val) == decode_header(v)
                         else f"arguments say {val}"))
    if "mcp-session-id" in h:
        rows.append(("Mcp-Session-Id", h["mcp-session-id"][:12] + "…", "legacy session (removed in 2026-07-28)"))
    for name, val, note in rows:
        if val is not None:
            out.append(f"   {name:<21} {_short(val, 26):<26} {note}")
    if ex.meta:
        bits = []
        if meta_v:
            bits.append(f"protocolVersion={meta_v}")
        ci = ex.meta.get(P + "clientInfo")
        bits.append(f"clientInfo={ci['name']} {ci['version']}" if ci else "clientInfo=(none)")
        bits.append(f"clientCapabilities={_caps(ex.meta[P + 'clientCapabilities'])}"
                    if P + "clientCapabilities" in ex.meta else "clientCapabilities=(MISSING)")
        out.append("   _meta  " + "  ".join(bits))
    elif ex.body and "params" in ex.body and ex.method == "initialize":
        p = ex.params
        out.append(f"   initialize: protocolVersion={p.get('protocolVersion')} clientInfo={p.get('clientInfo', {}).get('name')}"
                   " (version and capabilities sent once, for the whole session)")
    if ex.body and "id" in ex.body:
        extra = ""
        if ex.method == "tools/call":
            extra = f"   arguments {_short(ex.params.get('arguments'), 60)}"
        elif ex.method == "resources/read":
            extra = f"   uri {ex.params.get('uri')}"
        elif ex.method == "prompts/get":
            extra = f"   arguments {_short(ex.params.get('arguments'), 60)}"
        out.append(f"   id {ex.body['id']}{extra}")
    elif ex.body:
        out.append("   (notification: no id, no response body)")
    if "inputResponses" in ex.params:
        out.append(f"   inputResponses {_short(ex.params['inputResponses'], 70)}")
    if "requestState" in ex.params:
        out.append(f"   requestState echoed ({len(ex.params['requestState'])} characters, opaque: never parse it)")
    ctype = ex.resp_headers.get("content-type", "")
    sid = ex.resp_headers.get("mcp-session-id")
    out.append(f" ← {ex.status} {ctype or '(no body)'}" + (f"   Mcp-Session-Id {sid[:12]}… (minted by the server)" if sid else ""))
    res, err = ex.result, ex.error
    if err:
        out.append(f"   error {err.get('code')} {ERRORS.get(err.get('code'), '')}: {_short(err.get('message', ''), 80)}")
        if err.get("data"):
            out.append(f"   data {_short(err['data'], 80)}")
    if res is not None:
        rt = res.get("resultType")
        out.append(f"   resultType {rt}" if rt else "   resultType (absent: a legacy result, read as complete)")
        if "tools" in res:
            out.append(f"   tools {', '.join(t['name'] for t in res['tools'])}")
        if "prompts" in res:
            out.append(f"   prompts {', '.join(t['name'] for t in res['prompts'])}")
        if "supportedVersions" in res:
            out.append(f"   supportedVersions {res['supportedVersions']}  capabilities {_caps(res.get('capabilities', {}))}")
        if "capabilities" in res and "supportedVersions" not in res:
            out.append(f"   capabilities {_caps(res['capabilities'])}  protocolVersion {res.get('protocolVersion')}")
        if "structuredContent" in res:
            sc = res["structuredContent"]
            out.append(f"   structuredContent {_short(sc, 80)}")
        elif "content" in res:
            out.append(f"   content {_short(res['content'][0].get('text', res['content'][0].get('type')), 70)}")
        if "isError" in res:
            out.append(f"   isError {str(res['isError']).lower()}")
        if "contents" in res:
            out.append(f"   contents {res['contents'][0].get('uri')}  {_short(res['contents'][0].get('text', ''), 50)}")
        if "messages" in res:
            out.append(f"   messages {len(res['messages'])}: {_short(res['messages'][0]['content'].get('text', ''), 60)}")
        if "ttlMs" in res or "cacheScope" in res:
            scope = res.get("cacheScope")
            who = "any client or shared gateway may reuse it" if scope == "public" else "reuse only for the same caller's token"
            out.append(f"   ttlMs {_ttl(res.get('ttlMs', 0))}  cacheScope {scope}: {who}")
        if rt == "input_required":
            for key, r in (res.get("inputRequests") or {}).items():
                out.append(f"   inputRequests[{key}] {r['method']}: {_short(r['params'].get('message', ''), 60)}")
            if "requestState" in res:
                out.append(f"   requestState {res['requestState'][:14]}… ({len(res['requestState'])} characters, sealed by the server)")
        sm = (res.get("_meta") or {}).get(P + "serverInfo")
        if sm:
            out.append(f"   _meta serverInfo {sm['name']} {sm['version']}")
    if ex.note:
        out.append(f"   note: {ex.note}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("-n", type=int, help="show one exchange")
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", help="with --check: show exchanges in a range, e.g. 1-5 (findings use the whole file)")
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.only.split("-")) if a.only else (1, 10 ** 9)
    exs = load(a.path)
    schemas = known_schemas(exs, Path(a.path).parent)
    if a.check:
        found = check(exs, schemas)
        for ex in exs:
            if not lo <= ex.n <= hi:
                continue
            mine = [x for x in found if x.n == ex.n]
            print(headline(ex))
            for x in mine:
                print(f"     {x.side:<7} {x.code:<22} {x.detail}")
            if not mine:
                print("     ok")
        if hi >= len(exs):
            sm = summary(exs, found)
            print(f"{sm['exchanges']} exchanges; {sm['rejected']} rejected by the server: {sm['request_alone']} visible in "
                  f"the request alone, {sm['with_history']} more with the earlier exchanges, {sm['server_only']} only "
                  "to the server")
        return
    for ex in exs:
        if a.n and ex.n != a.n:
            continue
        print(headline(ex) if a.brief else "\n".join(describe(ex, schemas)) + "\n")


if __name__ == "__main__":
    main()
