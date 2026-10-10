"""Lab 2.4: fakes for the model, so agent code can be tested without a network, a key or a bill.

A ScriptedModel answers each request with the next scripted reply, in the provider's real response
shape, and keeps every request it received so a test can assert on what your code sent.
Run the tests:  python -m pytest fitness/s02 -k l24
"""
from __future__ import annotations

import itertools
from typing import Any

from skyops.cost import load_price
from skyops.model.client import ModelClient
from skyops.model.transport import HttpRequest, HttpResponse
from skyops.model.errors import LabNotDone

_ids = itertools.count(1)


def text_reply(text: str, input_tokens: int = 100, output_tokens: int = 20) -> dict[str, Any]:
    """An Anthropic-shaped final answer."""
    return {"id": f"msg_fake_{next(_ids)}", "type": "message", "role": "assistant", "model": "fake",
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


def tool_reply(name: str, args: dict[str, Any], input_tokens: int = 100, output_tokens: int = 30) -> dict[str, Any]:
    """An Anthropic-shaped tool proposal."""
    raise LabNotDone("Lecture 2.4: implement tool_reply(). Hints: labs/s02/README.md, lecture 2.4.")


class ScriptedModel:
    """A transport that plays back scripted replies in order and records every request."""

    def __init__(self, replies: list[dict[str, Any]]):
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def send(self, req: HttpRequest) -> HttpResponse:
        raise LabNotDone("Lecture 2.4: implement send(). Hints: labs/s02/README.md, lecture 2.4.")


def fake_client(replies: list[dict[str, Any]]) -> tuple[ModelClient, ScriptedModel]:
    script = ScriptedModel(replies)
    return ModelClient("anthropic", "fake", script, load_price("example")), script
