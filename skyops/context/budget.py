"""Lecture 8.1: the context window as a budget. Where every token of one request goes, by component.

    b = account(request)
    b.tokens -> {"rules": 212, "tools": 640, "history": 380, "tool_results": 910, "retrieved": 260, "scratchpad": 90}

Components (the system prompt is split by the blocks skyops.context.manager renders into it):
    rules         pinned blocks, plus the rest of the system prompt (persona, standing instructions)
    tools         the tool schemas sent with the request (name, description, input schema)
    history       user and assistant turns: text, the model's tool calls, and the rolling <summary>
    tool_results  tool results in the messages
    retrieved     <retrieved> and <memory> blocks: documents and memories chosen for this request
    scratchpad    the <scratchpad> block: plan, facts and open questions

Counts are estimates (skyops/context/tokens.py). Every message also costs a few tokens of framing
(roles, block boundaries) that providers don't itemise; MESSAGE_OVERHEAD charges it to history.

Not counted: when a request carries tools, the provider adds its own tool-use system prompt (Anthropic
documents 286 tokens on current Claude models with tool_choice auto, more for any/tool; API-12). The
accountant can't see it, so Window.margin_share must cover it: on the labs' smallest window (3,400 tokens)
the ten percent margin is 289 tokens.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..model.types import ChatRequest, TextBlock, ToolResultBlock, ToolUseBlock
from .tokens import chars4, count_tokens

COMPONENTS = ("rules", "tools", "history", "tool_results", "retrieved", "scratchpad")
MESSAGE_OVERHEAD = 4
BLOCK = re.compile(r"<(pinned|retrieved|memory|summary|scratchpad)\b[^>]*>.*?</\1>", re.S)
OF_BLOCK = {"pinned": "rules", "retrieved": "retrieved", "memory": "retrieved", "summary": "history",
            "scratchpad": "scratchpad"}


@dataclass(frozen=True)
class Window:
    """A model's context window, as a budget. Input and the reserved output must fit together, and the
    estimator's error needs a margin. Labs use a deliberately small window so overflow happens in minutes."""
    size_tokens: int
    reserve_output_tokens: int = 512
    margin_share: float = 0.10

    @property
    def input_limit(self) -> int:
        """The most input this window accepts, after reserving output and the estimator margin."""
        return int((self.size_tokens - self.reserve_output_tokens) * (1 - self.margin_share))


@dataclass
class Breakdown:
    tokens: dict[str, int] = field(default_factory=lambda: {c: 0 for c in COMPONENTS})
    chars4_total: int = 0           # the same request by the four-characters rule, for comparison

    @property
    def total(self) -> int:
        return sum(self.tokens.values())

    def share(self, component: str) -> float:
        return self.tokens[component] / self.total if self.total else 0.0

    def line(self) -> str:
        return " · ".join(f"{c} {self.tokens[c]:,}" for c in COMPONENTS)


def split_system(system: str) -> dict[str, str]:
    """System prompt text by component: rendered blocks to their component, everything else to rules."""
    parts = {c: "" for c in COMPONENTS}
    rest = system
    for m in BLOCK.finditer(system):
        parts[OF_BLOCK[m.group(1)]] += m.group(0)
        rest = rest.replace(m.group(0), "", 1)
    parts["rules"] += rest
    return parts


def account(req: ChatRequest) -> Breakdown:
    """Lecture 8.1: estimate the tokens of one request, by component. Pure: the request isn't changed."""
    b = Breakdown()
    texts: list[str] = []
    for comp, text in split_system(req.system or "").items():
        b.tokens[comp] += count_tokens(text)
        texts.append(text)
    for t in req.tools:
        s = json.dumps({"name": t.name, "description": t.description, "input_schema": t.input_schema})
        b.tokens["tools"] += count_tokens(s)
        texts.append(s)
    for m in req.messages:
        b.tokens["history"] += MESSAGE_OVERHEAD
        for blk in m.content:
            if isinstance(blk, TextBlock):
                b.tokens["history"] += count_tokens(blk.text)
                texts.append(blk.text)
            elif isinstance(blk, ToolUseBlock):
                s = json.dumps({"name": blk.name, "input": blk.input})
                b.tokens["history"] += count_tokens(s)
                texts.append(s)
            elif isinstance(blk, ToolResultBlock):
                b.tokens["tool_results"] += count_tokens(blk.content)
                texts.append(blk.content)
    b.chars4_total = chars4("".join(texts))
    return b
