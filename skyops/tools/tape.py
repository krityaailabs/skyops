"""Lab 2.5: record and replay TOOL calls, the other half of replay mode.

Model replay fixes what the model says. Tool replay fixes what the world says back: flight status,
a booking system, a payment API. With both recorded, a run can be replayed exactly, offline,
months later, which is what incident investigation needs (Section 20).

    tape = ToolTape.recorder()                      # wraps the real tools, keeps every call
    run(question, client, tool_runner=tape.wrap(flights.run_tool))
    tape.save("fixtures/tools/quickwin.jsonl")

    tape = ToolTape.load("fixtures/tools/quickwin.jsonl")   # replay: real tools are never called
    run(question, client, tool_runner=tape.wrap(None))
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ToolRunner = Callable[[str, dict[str, Any]], tuple[bool, str]]


class ToolReplayMiss(RuntimeError):
    pass


@dataclass
class ToolTape:
    mode: str                                   # "record" or "replay"
    entries: list[dict[str, Any]] = field(default_factory=list)
    position: int = 0

    @classmethod
    def recorder(cls) -> "ToolTape":
        return cls("record")

    @classmethod
    def load(cls, path: str | Path) -> "ToolTape":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls("replay", [json.loads(l) for l in lines if l.strip()])

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in self.entries), encoding="utf-8")

    def wrap(self, real: ToolRunner | None) -> ToolRunner:
        def runner(name: str, args: dict[str, Any]) -> tuple[bool, str]:
            if self.mode == "record":
                ok, output = real(name, args)
                self.entries.append({"name": name, "args": args, "ok": ok, "output": output})
                return ok, output
            if self.position >= len(self.entries):
                raise ToolReplayMiss(f"tool call {self.position + 1} ({name}) was never recorded")
            e = self.entries[self.position]
            self.position += 1
            if (e["name"], e["args"]) != (name, args):
                raise ToolReplayMiss(f"tool call {self.position}: recorded {e['name']}({e['args']}), "
                                     f"got {name}({args})")
            return e["ok"], e["output"]
        return runner
