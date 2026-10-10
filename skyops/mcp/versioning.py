"""Lecture 7.7: catch silent breaking changes before a host does.

A server's tool listing is its public contract. The lock file (skyops/mcp/tools.lock.json) records the
listing that was last released, with the server version. check_lock compares today's listing with it:

    change                                           needs at least
    a tool removed                                   a MAJOR bump (hosts calling it will break)
    a breaking schema change (see is_breaking)       a MAJOR bump
    a tool added                                     a MINOR bump
    any other change (description, annotations,      a MINOR bump: descriptions are prompts, so changing
      a new optional argument, output schema)          one changes behaviour (lecture 5.2)
    nothing changed, but the version moved            the lock must be refreshed (`make mcp-lock`)

    python -m skyops.mcp.versioning            check the current server against the lock
    python -m skyops.mcp.versioning --write    write a new lock (after a deliberate version bump)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LOCK = Path(__file__).with_name("tools.lock.json")


def semver(v: str) -> tuple[int, int, int]:
    major, minor, patch = (int(x) for x in v.split("."))
    return major, minor, patch


def is_breaking(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """A change a client written against `old` would trip over: a required argument added, an argument
    removed, an argument's type changed, or an enum value removed."""
    o, n = old.get("inputSchema") or {}, new.get("inputSchema") or {}
    op, np_ = o.get("properties") or {}, n.get("properties") or {}
    if set(n.get("required") or []) - set(o.get("required") or []):
        return True
    if set(op) - set(np_):
        return True
    for k in set(op) & set(np_):
        if op[k].get("type") != np_[k].get("type") or op[k].get("anyOf") != np_[k].get("anyOf"):
            return True
        if "enum" in op[k] and set(op[k]["enum"]) - set(np_[k].get("enum") or []):
            return True
    return False


def check_lock(listing: dict[str, dict[str, Any]], version: str, lock: dict[str, Any]) -> list[str]:
    """Lecture 7.7: problems that should fail the build. An empty list means the release is consistent.

    `listing`: tool name -> tool dict as served (tools/list). `lock`: {"version": ..., "tools": {name: tool}}.
    Work out the bump this change NEEDS ("major" > "minor" > "none") using the table in the module docstring
    and is_breaking(); work out the bump the version GOT by comparing semver(version) with semver(lock version):
      major part increased -> "major"; else minor increased -> "minor"; else -> "none" (a patch is "none").
    Problems, in this order:
      - for each removed tool (sorted): f"removed tool {name}"   (only if the bump got < major)
      - for each changed tool (sorted) that is breaking: f"breaking change to {name}"   (only if got < major)
      - if needed is minor and got is none: "tools changed without a version bump"
      - if needed is none and version != lock version: "version changed but the lock is stale: run make mcp-lock"
    """
    old = lock.get("tools", {})
    removed = sorted(set(old) - set(listing))
    added = sorted(set(listing) - set(old))
    changed = sorted(n for n in set(old) & set(listing) if old[n] != listing[n])
    breaking = [n for n in changed if is_breaking(old[n], listing[n])]
    needed = "major" if removed or breaking else ("minor" if added or changed else "none")
    (a, b, _), (c, d, _) = semver(version), semver(lock.get("version", "0.0.0"))
    got = "major" if a > c else ("minor" if b > d else "none")
    rank = {"none": 0, "minor": 1, "major": 2}
    problems = []
    if rank[got] < 2:
        problems += [f"removed tool {n}" for n in removed] + [f"breaking change to {n}" for n in breaking]
    if needed == "minor" and got == "none":
        problems.append("tools changed without a version bump")
    if needed == "none" and version != lock.get("version"):
        problems.append("version changed but the lock is stale: run make mcp-lock")
    return problems


def current_listing() -> tuple[dict[str, dict[str, Any]], str]:
    """The flight-ops tool listing exactly as a client receives it, and the server's version."""
    import anyio
    from mcp.client import Client
    from .server import VERSION, build_flight_server

    async def go() -> dict[str, dict[str, Any]]:
        async with Client(build_flight_server()) as c:
            return {t.name: t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in (await c.list_tools()).tools}
    return anyio.run(go), VERSION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    listing, version = current_listing()
    if a.write:
        LOCK.write_text(json.dumps({"server": "skyops-flight-ops", "version": version, "tools": listing},
                                   indent=1, sort_keys=True) + "\n")
        print(f"wrote {LOCK.name}: {len(listing)} tools at version {version}")
        return 0
    problems = check_lock(listing, version, json.loads(LOCK.read_text()))
    for p in problems:
        print("FAIL", p)
    print(f"{len(listing)} tools at version {version}: " + ("consistent with the lock" if not problems else
                                                              f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
