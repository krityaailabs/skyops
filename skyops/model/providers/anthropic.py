"""Anthropic Messages API adapter.  [VOLATILE API]

Verified 2026-10-09 against platform.claude.com/docs (facts API-01 to API-06):
- `model`, `max_tokens` and `messages` are required.
- Structured output: `output_config.format = {"type": "json_schema", "schema": ...}`.
- Seven stop reasons; we normalise them for the agent loop.
- 429 is a rate limit (retry-after header, except spend-cap 429s); 529 means overloaded.
"""
from __future__ import annotations

import os
from typing import Any, Iterator

from ..errors import ModelError, kind_from_status
from ..transport import HttpRequest, HttpResponse
from ..schema import portable
from ..types import ChatRequest, ModelResponse, ToolUseBlock, Usage

NAME = "anthropic"
URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

_STOP = {"end_turn": "end_turn", "stop_sequence": "end_turn", "tool_use": "tool_use",
         "max_tokens": "max_tokens", "refusal": "refusal",
         "model_context_window_exceeded": "context_exceeded", "pause_turn": "other"}


def build_request(req: ChatRequest, stream: bool = False) -> HttpRequest:
    body: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": [m.model_dump() for m in req.messages],
    }
    if req.system:
        body["system"] = req.system
    if req.tools:
        body["tools"] = [t.model_dump() for t in req.tools]
    if req.output_schema:
        body["output_config"] = {"format": {"type": "json_schema", "schema": portable(req.output_schema)}}
    if stream:
        body["stream"] = True
    headers = {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""), "anthropic-version": API_VERSION,
               "content-type": "application/json"}
    return HttpRequest(NAME, URL, headers, body, stream=stream)


def parse_response(body: dict[str, Any], headers: dict[str, str] | None = None) -> ModelResponse:
    text_parts, calls = [], []
    for block in body.get("content", []):
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "tool_use":
            calls.append(ToolUseBlock(id=block["id"], name=block["name"], input=block.get("input", {})))
        # thinking and server-tool blocks are not part of the answer; ignore them here
    u = body.get("usage", {})
    raw = body.get("stop_reason")
    return ModelResponse(
        provider=NAME, model=body.get("model", ""), text="".join(text_parts), tool_calls=calls,
        stop_reason=_STOP.get(raw, "other"), raw_stop_reason=raw,
        usage=Usage(input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                    cache_read_tokens=u.get("cache_read_input_tokens") or 0,
                    cache_write_tokens=u.get("cache_creation_input_tokens") or 0),
        request_id=(headers or {}).get("request-id"))


def parse_error(resp: HttpResponse) -> ModelError:
    err = (resp.body or {}).get("error", {}) if isinstance(resp.body, dict) else {}
    retry_after = resp.headers.get("retry-after")
    return ModelError(kind_from_status(resp.status), err.get("message", f"HTTP {resp.status}"),
                      status=resp.status, retry_after=float(retry_after) if retry_after else None, provider=NAME)


def iter_stream(events: list[dict[str, Any]]) -> Iterator[str | ModelResponse]:
    """Yield text deltas as they arrive, then one final ModelResponse assembled from the events.

    Event order: message_start, then content_block_start / content_block_delta / content_block_stop
    per block, then message_delta (stop reason, output tokens) and message_stop. Unknown events
    (ping, new types) are ignored rather than treated as errors.
    """
    import json as _json
    message: dict[str, Any] = {"content": [], "usage": {}}
    partial_json: dict[int, str] = {}
    for ev in events:
        data, kind = ev.get("data") or {}, ev.get("event") or (ev.get("data") or {}).get("type")
        if kind == "message_start":
            message.update({k: v for k, v in data["message"].items() if k != "content"})
            message["usage"] = dict(data["message"].get("usage", {}))
        elif kind == "content_block_start":
            message["content"].append(dict(data["content_block"]))
        elif kind == "content_block_delta":
            delta, idx = data["delta"], data["index"]
            if delta["type"] == "text_delta":
                message["content"][idx]["text"] = message["content"][idx].get("text", "") + delta["text"]
                yield delta["text"]
            elif delta["type"] == "input_json_delta":
                partial_json[idx] = partial_json.get(idx, "") + delta["partial_json"]
        elif kind == "content_block_stop":
            idx = data["index"]
            if idx in partial_json:
                message["content"][idx]["input"] = _json.loads(partial_json[idx] or "{}")
        elif kind == "message_delta":
            message["stop_reason"] = data["delta"].get("stop_reason")
            message["usage"].update(data.get("usage", {}))
        elif kind == "error":
            raise ModelError("overloaded" if data.get("error", {}).get("type") == "overloaded_error" else "other",
                             data.get("error", {}).get("message", "stream error"), provider=NAME)
    yield parse_response(message)
