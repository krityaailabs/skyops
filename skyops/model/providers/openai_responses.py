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
from skyops.model.errors import LabNotDone

NAME = "openai"
URL = "https://api.openai.com/v1/responses"


def _input_items(req: ChatRequest) -> list[dict[str, Any]]:
    raise LabNotDone("Lecture 1.7: implement _input_items(). Hints: labs/s01/README.md, lecture 1.7.")


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
    raise LabNotDone("Lecture 1.7: implement parse_response(). Hints: labs/s01/README.md, lecture 1.7.")


def parse_error(resp: HttpResponse) -> ModelError:
    err = (resp.body or {}).get("error", {}) if isinstance(resp.body, dict) else {}
    retry_after = resp.headers.get("retry-after")
    return ModelError(kind_from_status(resp.status), (err or {}).get("message", f"HTTP {resp.status}"),
                      status=resp.status, retry_after=float(retry_after) if retry_after else None, provider=NAME)
