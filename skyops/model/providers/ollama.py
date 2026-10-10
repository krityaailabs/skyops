"""Ollama native chat adapter (local models).  [VOLATILE API]

Verified 2026-10-09 against docs.ollama.com (fact API-08):
- POST /api/chat; tools as {"type": "function", "function": {...}}.
- Tool call arguments come back as an *object*; results go back as role "tool" with `tool_name`.
- `format` accepts a JSON schema for structured output. Smaller local models follow it
  less reliably than hosted constrained decoding, which is why the repair loop in lecture 1.3 exists.
- Ollama returns no tool-call ids, so this adapter makes deterministic ones.
"""
from __future__ import annotations

from typing import Any

from ..errors import ModelError, kind_from_status
from ..transport import HttpRequest, HttpResponse
from ..types import ChatRequest, ModelResponse, TextBlock, ToolResultBlock, ToolUseBlock, Usage

NAME = "ollama"


def _url() -> str:
    from ...settings import Settings
    return Settings.from_env().ollama_host.rstrip("/") + "/api/chat"


def build_request(req: ChatRequest, stream: bool = False) -> HttpRequest:
    msgs: list[dict[str, Any]] = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    names: dict[str, str] = {}
    for m in req.messages:
        text = "".join(b.text for b in m.content if isinstance(b, TextBlock))
        calls = [b for b in m.content if isinstance(b, ToolUseBlock)]
        results = [b for b in m.content if isinstance(b, ToolResultBlock)]
        for c in calls:
            names[c.id] = c.name
        if text or calls:
            msg: dict[str, Any] = {"role": m.role, "content": text}
            if calls:
                msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.input}} for c in calls]
            msgs.append(msg)
        for r in results:
            msgs.append({"role": "tool", "content": r.content, "tool_name": names.get(r.tool_use_id, "")})
    body: dict[str, Any] = {"model": req.model, "messages": msgs, "stream": False,
                            "options": {"num_predict": req.max_tokens}}
    if req.tools:
        body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                           "parameters": t.input_schema}} for t in req.tools]
    if req.output_schema:
        body["format"] = req.output_schema
    return HttpRequest(NAME, _url(), {"content-type": "application/json"}, body)


def parse_response(body: dict[str, Any], headers: dict[str, str] | None = None) -> ModelResponse:
    msg = body.get("message", {})
    calls = [ToolUseBlock(id=_call_id(body, i, c), name=c["function"]["name"], input=c["function"].get("arguments", {}))
             for i, c in enumerate(msg.get("tool_calls") or [])]
    reason = body.get("done_reason")
    stop = "tool_use" if calls else {"stop": "end_turn", "length": "max_tokens"}.get(reason, "other")
    return ModelResponse(provider=NAME, model=body.get("model", ""), text=msg.get("content", ""), tool_calls=calls,
                         stop_reason=stop, raw_stop_reason=reason,
                         usage=Usage(input_tokens=body.get("prompt_eval_count", 0),
                                     output_tokens=body.get("eval_count", 0)))


def _call_id(body: dict[str, Any], i: int, call: dict[str, Any]) -> str:
    """Ollama returns no call ids. Make one that is unique per response and stable on replay."""
    import hashlib
    import json
    seed = json.dumps([body.get("created_at"), body.get("prompt_eval_count"), i, call], sort_keys=True)
    return "ollama_" + hashlib.sha256(seed.encode()).hexdigest()[:12]


def parse_error(resp: HttpResponse) -> ModelError:
    msg = (resp.body or {}).get("error", f"HTTP {resp.status}") if isinstance(resp.body, dict) else f"HTTP {resp.status}"
    return ModelError(kind_from_status(resp.status), str(msg), status=resp.status, provider=NAME)
