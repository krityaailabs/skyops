"""OpenAI Responses API adapter.  [VOLATILE API]

Verified 2026-10-09 against developers.openai.com/api/docs (fact API-07):
- The Responses API is recommended for new work; Chat Completions is still supported.
- Structured output: `text.format = {"type": "json_schema", "name", "schema", "strict": true}`.
- A tool call comes back as a `function_call` output item whose `arguments` is a JSON *string*.
- You return the result as a `function_call_output` input item with the same `call_id`.
- There is no `seed` parameter in the Responses API.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ..errors import ModelError, kind_from_status
from ..transport import HttpRequest, HttpResponse
from ..schema import portable
from ..types import ChatRequest, ModelResponse, TextBlock, ToolResultBlock, ToolUseBlock, Usage

NAME = "openai"
URL = "https://api.openai.com/v1/responses"


def _input_items(req: ChatRequest) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in req.messages:
        for b in m.content:
            if isinstance(b, TextBlock):
                items.append({"role": m.role, "content": b.text})
            elif isinstance(b, ToolUseBlock):
                items.append({"type": "function_call", "call_id": b.id, "name": b.name,
                              "arguments": json.dumps(b.input)})
            elif isinstance(b, ToolResultBlock):
                items.append({"type": "function_call_output", "call_id": b.tool_use_id, "output": b.content})
    return items


def build_request(req: ChatRequest, stream: bool = False) -> HttpRequest:
    body: dict[str, Any] = {"model": req.model, "input": _input_items(req), "max_output_tokens": req.max_tokens}
    if req.system:
        body["instructions"] = req.system
    if req.tools:
        body["tools"] = [{"type": "function", "name": t.name, "description": t.description,
                          "parameters": t.input_schema, "strict": False} for t in req.tools]
    if req.output_schema:
        body["text"] = {"format": {"type": "json_schema", "name": req.schema_name,
                                   "schema": portable(req.output_schema), "strict": True}}
    headers = {"authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
               "content-type": "application/json"}
    return HttpRequest(NAME, URL, headers, body, stream=stream)


def parse_response(body: dict[str, Any], headers: dict[str, str] | None = None) -> ModelResponse:
    text_parts: list[str] = []
    calls: list[ToolUseBlock] = []
    refused = False
    for item in body.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text_parts.append(part["text"])
                elif part.get("type") == "refusal":
                    refused = True
        elif item.get("type") == "function_call":
            # arguments is a JSON string here (not an object, unlike Anthropic and Ollama)
            calls.append(ToolUseBlock(id=item["call_id"], name=item["name"],
                                      input=json.loads(item.get("arguments") or "{}")))
        # reasoning items are skipped: they are not part of the answer
    incomplete = (body.get("incomplete_details") or {}).get("reason")
    if refused:
        stop = "refusal"
    elif calls:
        stop = "tool_use"
    elif body.get("status") == "incomplete" and incomplete == "max_output_tokens":
        stop = "max_tokens"
    elif body.get("status") == "completed":
        stop = "end_turn"
    else:
        stop = "other"
    u = body.get("usage", {})
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    return ModelResponse(
        provider=NAME, model=body.get("model", ""), text="".join(text_parts), tool_calls=calls,
        stop_reason=stop, raw_stop_reason=incomplete or body.get("status"),
        # OpenAI reports cached tokens *inside* input_tokens; split them so cost maths is consistent
        usage=Usage(input_tokens=u.get("input_tokens", 0) - cached, output_tokens=u.get("output_tokens", 0),
                    cache_read_tokens=cached),
        request_id=(headers or {}).get("x-request-id"))


def parse_error(resp: HttpResponse) -> ModelError:
    err = (resp.body or {}).get("error", {}) if isinstance(resp.body, dict) else {}
    retry_after = resp.headers.get("retry-after")
    return ModelError(kind_from_status(resp.status), (err or {}).get("message", f"HTTP {resp.status}"),
                      status=resp.status, retry_after=float(retry_after) if retry_after else None, provider=NAME)
