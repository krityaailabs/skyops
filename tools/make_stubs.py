"""Produce the learner (start-tag) version of lab files: keep signatures and docstrings, replace
bodies with a LabNotDone that names the lecture and the README with the hint ladder.

Usage: python tools/make_stubs.py <file> [<file> ...]   (rewrites files in place)
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# file -> {function or method name: lecture}
STUBS = {
    "labs/s01/first_call.py": {"ask": "1.2", "stream_answer": "1.2", "call_with_retry": "1.2"},
    "labs/s01/extract.py": {"extract_notice": "1.3"},
    "labs/s01/costing.py": {"report": "1.4"},
    "skyops/model/providers/openai_responses.py": {"_input_items": "1.7", "parse_response": "1.7"},
    "labs/s01/checkpoint.py": {"extract_batch": "1.8"},
    "labs/s02/concurrent.py": {"fetch_all": "2.2"},
    "labs/s02/fakes.py": {"tool_reply": "2.4", "send": "2.4"},
    "skyops/tools/tape.py": {"wrap": "2.5"},
    "labs/s02/assignment.py": {"get_status": "2.7"},
    "labs/s03/detectors.py": {"detect_loop": "3.3", "detect_slow_tool": "3.3", "detect_bad_arguments": "3.3",
                              "detect_duplicate_side_effects": "3.3", "detect_canary": "3.3"},
    "skyops/agent/loop.py": {"run_agent": "4.2", "validate_call": "4.3", "check_budgets": "4.4", "execute": "4.5",
                             "detect_runaway": "4.6"},
    "labs/s04/eval.py": {"check_case": "4.7", "run_eval": "4.7"},
    "projects/p1_flight_status/assistant.py": {"chosen_flight": "4.8", "answer": "4.8"},
    "skyops/tools/contracts.py": {"idempotency_key": "5.5", "execute_write": "5.5", "classify_exception": "5.6",
                                  "shape_result": "5.7", "lint_contract": "5.8"},
    "skyops/tools/catalog.py": {"list_departures": "5.9", "check_voucher": "5.9"},
    "labs/s07/first_server.py": {"build": "7.1"},
    "skyops/mcp/adapter.py": {"annotations_for": "7.2", "to_call_result": "7.2"},
    "skyops/mcp/server.py": {"booking_resource": "7.3", "disruption_brief": "7.3"},
    "skyops/mcp/client.py": {"map_call_result": "7.5"},
    "skyops/mcp/versioning.py": {"check_lock": "7.7"},
    "skyops/mcp/gateway.py": {"decide": "7.9"},
}
IMPORT = "from skyops.model.errors import LabNotDone\n"


def stub(path: str, funcs: dict[str, str]) -> None:
    p = ROOT / path
    src = p.read_text()
    lines = src.splitlines(keepends=True)
    edits = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in funcs:
            body, has_doc = node.body, ast.get_docstring(node) is not None
            if has_doc and len(body) == 1:
                continue
            start = body[1].lineno if has_doc else body[0].lineno
            indent = " " * body[0].col_offset
            lec = funcs[node.name]
            readme = f"labs/s{int(lec.split('.')[0]):02d}/README.md"
            edits.append((start - 1, node.end_lineno,
                          f"{indent}raise LabNotDone(\"Lecture {lec}: implement {node.name}(). "
                          f"Hints: {readme}, lecture {lec}.\")\n"))
    for s, e, msg in sorted(edits, reverse=True):
        lines[s:e] = [msg]
    out = "".join(lines)
    if "import LabNotDone" not in out:
        ol = out.splitlines(keepends=True)
        idx = max(i for i, l in enumerate(ol) if l.startswith(("import ", "from ")) and not l.startswith("from __future__"))
        ol.insert(idx + 1, IMPORT)
        out = "".join(ol)
    p.write_text(out)
    print(f"stubbed {path}: {', '.join(funcs)}")


if __name__ == "__main__":
    for path in sys.argv[1:] or list(STUBS):
        stub(path, STUBS[path])
