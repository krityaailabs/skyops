"""Lecture 6.9: review a third-party MCP server before the SkyOps agent may use it.

Reads three things and reviews them together:
    fixtures/mcp/third_party.jsonl      what the server said about itself (recorded from third_party.py)
    labs/s06/third_party_manifest.json  what its publisher claims
    labs/s06/mcp_policy.json            SkyOps rules for third-party servers

Five areas, in the order a reviewer works: provenance, version pin, descriptions, scopes, tools. Then a
decision with an allow-list, a pin and a listing digest, so a later change to any description is caught.

    python -m labs.s06.review_server
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .wire import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PURPOSE = "flight weather for duty managers"
WRITE_VERBS = ("sync", "send", "update", "delete", "create", "write", "set", "issue")


@dataclass
class Finding:
    area: str
    severity: str        # block | warn
    detail: str


def listing_digest(tools: list[dict], prompts: list[dict] | None = None, templates: list[dict] | None = None) -> str:
    """Full sha256 over every tool object (name, title, description, schemas, annotations, everything), plus any
    prompts and resource templates, as canonical JSON. Any change to what the model or the host reads changes it."""
    by_name = lambda xs: sorted(xs or [], key=lambda t: t.get("name") or t.get("uriTemplate", ""))
    canon = json.dumps({"tools": by_name(tools), "prompts": by_name(prompts), "resourceTemplates": by_name(templates)},
                       sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def review(tools: list[dict], server_info: dict, manifest: dict, policy: dict, purpose: str = PURPOSE) -> list[Finding]:
    out: list[Finding] = []
    # 1 provenance
    if manifest.get("publisher") not in policy["trusted_publishers"]:
        out.append(Finding("provenance", "block", f"publisher {manifest.get('publisher')!r} is not on the trusted list"))
    if policy.get("require_signed_or_internal_build") and not manifest.get("signed"):
        out.append(Finding("provenance", "block", "no signature or verifiable build; you can't tell what you'd be running"))
    if not manifest.get("privacy_policy"):
        out.append(Finding("provenance", "warn", "no privacy policy: where do arguments and results go?"))
    # 2 version pin
    if policy.get("require_exact_version_pin") and manifest.get("version") in (None, "latest", "*"):
        out.append(Finding("version", "block", f"manifest version {manifest.get('version')!r}: pin "
                           f"{server_info.get('name')}=={server_info.get('version')} and the listing digest instead"))
    # 3 descriptions (the model reads these as instructions)
    phrases = policy["suspicious_description_phrases"]
    for t in tools:
        low = (t.get("description") or "").lower()
        hits = [p for p in phrases if p in low]
        if hits:
            out.append(Finding("descriptions", "block", f"{t['name']}: instructions aimed at the model ({', '.join(hits)})"))
        elif len(low.split()) < 8:
            out.append(Finding("descriptions", "warn", f"{t['name']}: too vague to know what it does ({len(low.split())} words)"))
    # 4 scopes
    needed = set(policy["max_scopes_per_purpose"].get(purpose, []))
    for s in manifest.get("requested_scopes", []):
        if s in policy["never_grant"]:
            out.append(Finding("scopes", "block", f"requests {s}, which SkyOps never grants to a third party"))
        elif s not in needed:
            out.append(Finding("scopes", "warn", f"requests {s}; the purpose needs only {sorted(needed)}"))
    # 5 tools: secrets in arguments, annotations that don't fit the name
    for t in tools:
        props = (t.get("inputSchema") or {}).get("properties") or {}
        for arg in props:
            if any(s in arg.lower() for s in policy["secret_argument_names"]):
                out.append(Finding("tools", "block", f"{t['name']}.{arg}: a credential as a tool argument; the model "
                                                     "would see it, and it would sit in prompts and logs"))
        ann = t.get("annotations") or {}
        if ann.get("readOnlyHint") and t["name"].split("_")[0] in WRITE_VERBS:
            out.append(Finding("tools", "warn", f"{t['name']}: claims readOnlyHint, but the name says it writes; "
                                                "annotations from an untrusted server are untrusted"))
    return out


def allow_list(tools: list[dict], findings: list[Finding]) -> list[str]:
    """Tools with no blocking finding of their own, and only those the purpose needs."""
    blocked = {f.detail.split(":")[0].split(".")[0] for f in findings if f.severity == "block"}
    return [t["name"] for t in tools if t["name"] not in blocked and "weather" in t["name"]]


def main() -> None:
    exs = load(ROOT / "fixtures/mcp/third_party.jsonl")
    tl = next(e for e in exs if e.method == "tools/list").result
    disc = next(e for e in exs if e.method == "server/discover").result
    info = disc["_meta"]["io.modelcontextprotocol/serverInfo"]
    manifest = json.loads((HERE / "third_party_manifest.json").read_text())
    policy = json.loads((HERE / "mcp_policy.json").read_text())
    tools = tl["tools"]
    found = review(tools, info, manifest, policy)
    print(f"Server {info['name']} {info['version']}: {len(tools)} tools ({', '.join(t['name'] for t in tools)})")
    for area in ("provenance", "version", "descriptions", "scopes", "tools"):
        for f in [x for x in found if x.area == area]:
            print(f"  {f.severity.upper():<5} {area:<12} {f.detail}")
    blocks = sum(f.severity == "block" for f in found)
    allowed = allow_list(tools, found)
    print(f"{len(found)} findings, {blocks} blocking")
    print("Decision: do not connect as offered." if blocks else "Decision: connect with the pin below.")
    print(f"  If the provenance blocks are cleared: allow-list {allowed}, pin {info['name']}=={info['version']}, "
          f"listing {listing_digest(tools)[:19]}…, scopes {policy['max_scopes_per_purpose'][PURPOSE]}; "
          "re-review on any digest change")


if __name__ == "__main__":
    main()
