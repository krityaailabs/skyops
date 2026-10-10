"""A minimal run trace: every model call and tool call, with tokens, cost and timing.

Section 20 replaces this with OpenTelemetry spans. The idea is the same from day one:
if you can't see what the agent did, you can't debug it, audit it or price it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cost import Price, cost_usd, per_thousand
from .model.types import ModelResponse, Usage


def _n(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


@dataclass
class Trace:
    question: str
    price: Price
    deterministic: bool = True          # replay runs get stable ids so outputs can be compared
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def run_id(self) -> str:
        seed = self.question if self.deterministic else f"{self.question}{time.time_ns()}"
        return hashlib.sha256(seed.encode()).hexdigest()[:8]

    def model_call(self, resp: ModelResponse) -> None:
        self.usage = self.usage + resp.usage
        self.events.append({"kind": "model", "stop": resp.stop_reason, "in": resp.usage.input_tokens,
                            "out": resp.usage.output_tokens, "usd": cost_usd(resp.usage, self.price),
                            "tools": [c.name for c in resp.tool_calls]})

    def tool_call(self, name: str, args: dict[str, Any], ok: bool, summary: str) -> None:
        self.events.append({"kind": "tool", "name": name, "args": args, "ok": ok, "summary": summary})

    def context(self, event: dict[str, Any]) -> None:
        """Section 8: what the context layer sent (tokens by component, compaction, pinned check, scratchpad)."""
        self.events.append(dict(event))

    def answer(self, text: str) -> None:
        self.events.append({"kind": "answer", "text": text})

    @property
    def model_calls(self) -> int:
        return sum(1 for e in self.events if e["kind"] == "model")

    @property
    def tool_calls(self) -> int:
        return sum(1 for e in self.events if e["kind"] == "tool")

    @property
    def cost(self) -> float:
        return cost_usd(self.usage, self.price)

    def summary(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "model_calls": self.model_calls, "tool_calls": self.tool_calls,
                "input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total, "cost_usd": self.cost,
                "cost_per_1000_runs_usd": per_thousand(self.cost), "price_profile": self.price.name}

    def render(self) -> str:
        lines = [f"run {self.run_id}  question: {self.question!r}"]
        n = 0
        for e in self.events:
            if e["kind"] == "model":
                n += 1
                tools = f" -> proposes {', '.join(e['tools'])}" if e["tools"] else ""
                lines.append(f"├─ model call {n}: in={e['in']} out={e['out']} stop={e['stop']}{tools}")
            elif e["kind"] == "context":
                t = e["tokens"]
                parts = " ".join(f"{k.replace('tool_results', 'results')} {v:,}" for k, v in t.items() if v)
                extra = (f" · compacted {_n(e['compacted_turns'], 'turn')}" if e["compacted_turns"] else "") + \
                        (f" · DROPPED {_n(e['fit_dropped'], 'turn')}" if e["fit_dropped"] else "")
                lines.append(f"├─ context: {e['total']:,} tokens ({parts}) · pinned {e['pinned']}{extra}")
                if e.get("scratchpad"):
                    lines.append(f"│   scratchpad: {e['scratchpad']}")
            elif e["kind"] == "tool":
                args = ", ".join(f"{k}={(repr(v) if len(repr(v)) <= 40 else repr(v)[:36] + '…' + repr(v)[-1])}"
                                 for k, v in e["args"].items())
                lines.append(f"│   └─ tool {e['name']}({args}) {'ok' if e['ok'] else 'ERROR'}: {e['summary']}")
            else:
                lines.append(f"└─ answer: {e['text']}")
        s = self.summary()
        lines.append(f"totals: {s['model_calls']} model calls, {s['tool_calls']} tool calls, "
                     f"{s['total_tokens']} tokens, {s['cost_per_1000_runs_usd']:.2f} USD per 1,000 runs "
                     f"(price profile: {self.price.name})")
        return "\n".join(lines)

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.run_id}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in self.events + [{"kind": "summary", **self.summary()}]) + "\n")
        return path
