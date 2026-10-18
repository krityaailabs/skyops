"""Check the CI workflow offline: valid YAML, valid against GitHub's workflow schema, and the course's rules.

    python tools/check_workflow.py [.github/workflows/evals.yml]
GitHub Actions can't run here, so this is how the workflow is tested before it is pushed. The schema is
SchemaStore's (tools/schemas/github-workflow.json, Apache 2.0). Then four rules of our own:
1. every `uses:` is pinned to a full 40-character commit SHA (a tag can be moved; a SHA can't);
2. the workflow's token is read-only (top-level permissions: contents: read, nothing else);
3. every job has timeout-minutes;
4. some step runs exactly `make ci-gate ...`: the same command developers run;
5. no pull_request_target trigger (it runs with the repository's secrets and token);
6. the gate step, and its job, have no continue-on-error and no if: (nothing may turn the gate off).
One YAML trap is handled explicitly: YAML 1.1 reads the key `on` as the boolean True.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github/workflows/evals.yml"
SCHEMA = ROOT / "tools/schemas/github-workflow.json"
SHA = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")


def load(path: Path) -> dict:
    doc = yaml.safe_load(Path(path).read_text())
    if isinstance(doc, dict) and True in doc:          # YAML 1.1: `on:` parses as True
        doc["on"] = doc.pop(True)
    return doc


def problems(path: Path = WORKFLOW) -> list[str]:
    doc = load(path)
    validator = jsonschema.Draft7Validator(json.loads(SCHEMA.read_text()))
    out = []
    for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
        best = jsonschema.exceptions.best_match(e.context) if e.context else e   # the most specific reason
        where = "/".join(map(str, best.path)) or "/".join(map(str, e.path)) or "(top)"
        msg = best.message if len(best.message) <= 120 else best.message[:117] + "..."
        out.append(f"schema: {where}: {msg}")
    if doc.get("permissions") != {"contents": "read"}:
        out.append("permissions: the workflow token must be read-only (permissions: contents: read)")
    if "pull_request_target" in (doc.get("on") or {}):
        out.append("on: pull_request_target runs with the repository's secrets and token; use pull_request")
    gate = False
    for jid, job in (doc.get("jobs") or {}).items():
        if "timeout-minutes" not in job:
            out.append(f"jobs/{jid}: no timeout-minutes")
        for i, step in enumerate(job.get("steps", [])):
            if "uses" in step and not SHA.match(step["uses"]):
                out.append(f"jobs/{jid}/steps/{i}: `{step['uses']}` is not pinned to a full commit SHA")
            if str(step.get("run", "")).strip().startswith("make ci-gate"):
                gate = True
                for where, obj in ((f"jobs/{jid}/steps/{i}", step), (f"jobs/{jid}", job)):
                    for key in ("continue-on-error", "if"):
                        if key in obj:
                            out.append(f"{where}: the gate must not have `{key}`")
    if not gate:
        out.append("no step runs `make ci-gate`")
    return out


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else WORKFLOW
    found = problems(path)
    doc = load(path)
    steps = sum(len(j.get("steps", [])) for j in doc["jobs"].values())
    shown = path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path
    print(f"{shown}: valid YAML; {len(doc['jobs'])} job(s), "
          f"{steps} steps; triggers: {', '.join(doc['on'])}")
    for p in found:
        print(f"PROBLEM {p}")
    print("workflow OK: schema, pinned actions, read-only token, timeouts, make ci-gate, no pull_request_target, "
          "nothing turns the gate off" if not found
          else f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
