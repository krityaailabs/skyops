"""Core tests. These pass at every tag from s00 onwards; CI runs them in replay mode."""
import json

import pytest

from skyops.cost import Price, cost_usd, load_price, per_thousand
from skyops.model.client import ModelClient
from skyops.model.errors import ModelError
from skyops.model.providers import anthropic, ollama
from skyops.model.transport import HttpRequest, HttpResponse, ReplayMiss, ReplayTransport, request_key
from skyops.model.types import ChatRequest, Message, ToolResultBlock, ToolUseBlock, Usage
from skyops.settings import ROOT

PRICE = Price("t", input_per_mtok=1.0, output_per_mtok=5.0, cache_read_per_mtok=0.1)


class Never:
    def send(self, req):  # pragma: no cover - must not be reached
        raise AssertionError("network must not be called")


class Fixed:
    def __init__(self, *responses):
        self.responses, self.sent = list(responses), []

    def send(self, req):
        self.sent.append(req)
        return self.responses.pop(0)


def test_request_key_ignores_headers():
    a = HttpRequest("anthropic", "u", {"x-api-key": "secret-1"}, {"m": 1})
    b = HttpRequest("anthropic", "u", {"x-api-key": "secret-2"}, {"m": 1})
    assert request_key(a) == request_key(b)


def test_strict_replay_rejects_a_different_request(tmp_path):
    f = tmp_path / "f.jsonl"
    f.write_text(json.dumps({"provider": "anthropic", "key": "deadbeefdeadbeef", "status": 200, "body": {}}) + "\n")
    with pytest.raises(ReplayMiss, match="differs"):
        ReplayTransport(f, match="strict").send(HttpRequest("anthropic", "u", {}, {"x": 1}))


def test_replay_exhausted_points_at_the_step_budget(tmp_path):
    f = tmp_path / "f.jsonl"
    f.write_text("")
    with pytest.raises(ReplayMiss, match="step budget"):
        ReplayTransport(f).send(HttpRequest("anthropic", "u", {}, {}))


def test_anthropic_errors_are_classified():
    e429 = anthropic.parse_error(HttpResponse(429, {"retry-after": "2"}, {"error": {"type": "rate_limit_error"}}))
    e529 = anthropic.parse_error(HttpResponse(529, {}, {"error": {"type": "overloaded_error"}}))
    e400 = anthropic.parse_error(HttpResponse(400, {}, {"error": {"type": "invalid_request_error"}}))
    assert (e429.kind, e429.retry_after, e429.retryable) == ("rate_limited", 2.0, True)
    assert (e529.kind, e529.retryable) == ("overloaded", True)
    assert (e400.kind, e400.retryable) == ("invalid_request", False)


def test_anthropic_stop_reasons_normalise():
    body = {"content": [], "stop_reason": "model_context_window_exceeded", "usage": {}}
    assert anthropic.parse_response(body).stop_reason == "context_exceeded"
    body["stop_reason"] = "pause_turn"
    assert anthropic.parse_response(body).stop_reason == "other"


def test_anthropic_request_requires_max_tokens_and_uses_output_config():
    r = anthropic.build_request(ChatRequest(model="m", messages=[Message.user("hi")], max_tokens=10,
                                            output_schema={"type": "object"}))
    assert r.json["max_tokens"] == 10
    assert r.json["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert "temperature" not in r.json


def test_ollama_tool_round_trip():
    body = {"model": "m", "message": {"role": "assistant", "content": "",
                                      "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
            "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 2}
    resp = ollama.parse_response(body)
    assert resp.stop_reason == "tool_use" and resp.tool_calls[0].input == {"a": 1}
    req = ChatRequest(model="m", max_tokens=5, messages=[
        Message.user("q"), resp.as_message(),
        Message(role="user", content=[ToolResultBlock(tool_use_id=resp.tool_calls[0].id, content="ok")])])
    msgs = ollama.build_request(req).json["messages"]
    assert msgs[-1] == {"role": "tool", "content": "ok", "tool_name": "f"}


def test_cost_maths_and_per_thousand():
    u = Usage(input_tokens=1000, output_tokens=200, cache_read_tokens=500)
    assert cost_usd(u, PRICE) == pytest.approx((1000 * 1 + 200 * 5 + 500 * 0.1) / 1e6)
    assert per_thousand(0.002) == pytest.approx(2.0)


def test_example_price_profile_is_labelled_as_not_a_quote():
    p = load_price("example")
    assert "not a provider price" in p.source


def test_spend_guard_blocks_before_the_network():
    c = ModelClient("anthropic", "m", Never(), PRICE, cap_usd=0.0001, live=True)
    with pytest.raises(ModelError) as e:
        c.complete(ChatRequest(model="m", messages=[Message.user("hi")], max_tokens=4000))
    assert e.value.kind == "budget_exceeded"


def test_no_env_file_is_tracked():
    import subprocess
    assert ".env" in (ROOT / ".gitignore").read_text().split(), ".env must be ignored"
    if (ROOT / ".git").exists():
        tracked = subprocess.run(["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True).stdout
        assert not tracked.strip(), ".env is tracked by git: remove it with git rm --cached .env"


def test_a_typo_in_the_mode_fails_at_start_up():
    from skyops.settings import Settings
    with pytest.raises(ValueError, match="mode"):
        Settings.from_env(mode="replya")
