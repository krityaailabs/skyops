"""Section 1 fitness functions. They fail at tag s01-start (each names the lecture to finish)
and pass at tag s01. Labs replay in "sequence" mode, so your own wording of a prompt still runs."""
import json

import pytest

from labs.s01 import checkpoint, costing, extract, first_call
from labs.s01.notices import EXPECTED, NOTICES
from skyops.cost import load_price
from skyops.model.client import ModelClient, get_client
from skyops.model.errors import ModelError
from skyops.model.providers import openai_responses
from skyops.model.transport import HttpResponse
from skyops.model.types import ChatRequest, Message, ToolResultBlock, ToolUseBlock


def client(name, provider="anthropic"):
    return get_client(fixture=f"fixtures/s01/{name}.jsonl", provider=provider, match="sequence")


class Fixed:
    def __init__(self, *responses):
        self.responses = list(responses)

    def send(self, req):
        return self.responses.pop(0)


# Lecture 1.2
def test_l12_ask_returns_the_whole_response():
    r = first_call.ask(client("first_call"))
    assert "AIDX" in r.text and r.stop_reason == "end_turn" and r.usage.output_tokens > 0


def test_l12_stream_delivers_chunks_then_a_final_response():
    chunks = []
    final = first_call.stream_answer(client("stream"), first_call.QUESTION, chunks.append)
    assert len(chunks) > 1 and "".join(chunks) == final.text and final.usage.output_tokens > 0


def test_l12_retry_honours_retry_after_then_backs_off():
    c, waits = client("errors"), []
    req = ChatRequest(model=c.model, system=first_call.SYSTEM, messages=[Message.user(first_call.QUESTION)],
                      max_tokens=200)
    r = first_call.call_with_retry(c, req, sleep=waits.append)
    assert c.calls == 3 and waits == [2.0, 2.0] and r.stop_reason == "end_turn"


def test_l12_never_retries_a_bad_request():
    c = ModelClient("anthropic", "m", Fixed(*[HttpResponse(400, {}, {"error": {"message": "bad"}})] * 3), load_price())
    with pytest.raises(ModelError):
        first_call.call_with_retry(c, ChatRequest(model="m", messages=[Message.user("x")], max_tokens=5),
                                   sleep=lambda s: None)
    assert c.calls == 1


# Lecture 1.3
@pytest.mark.parametrize("provider,calls", [("anthropic", 1), ("openai", 1), ("ollama", 2)])
def test_l13_extract_is_validated_on_every_provider(provider, calls):
    c = client(f"extract_{provider}", provider)
    assert extract.extract_notice(c, NOTICES[0]) == EXPECTED[0]
    assert c.calls == calls      # the local model needed one repair


def test_l13_invalid_output_is_never_returned():
    bad = {"content": [{"type": "text", "text": '{"flight_number": "ZZ1", "status": "late"}'}],
           "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
    c = ModelClient("anthropic", "m", Fixed(HttpResponse(200, {}, bad), HttpResponse(200, {}, bad)), load_price())
    with pytest.raises(extract.ExtractionFailed):
        extract.extract_notice(c, "ZZ1 late")
    assert c.calls == 2


# Lecture 1.4
def test_l14_cost_report_uses_reported_usage():
    from skyops.model.types import ModelResponse, Usage
    rs = [ModelResponse(provider="p", model="m", text="", stop_reason="end_turn",
                        usage=Usage(input_tokens=1000, output_tokens=100)) for _ in range(4)]
    rep = costing.report(rs, load_price("example"))
    assert rep.calls == 4 and rep.input_tokens == 4000 and rep.output_tokens == 400
    assert rep.per_1000_calls_usd == pytest.approx(rep.per_call_usd * 1000)
    assert rep.input_share == pytest.approx(1.0 / (1.0 + 0.5))


# Lecture 1.7
def test_l17_openai_adapter_maps_tools_both_ways():
    req = ChatRequest(model="m", max_tokens=50, messages=[
        Message.user("q"),
        Message(role="assistant", content=[ToolUseBlock(id="call_1", name="f", input={"a": 1})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_1", content="ok")])])
    items = openai_responses.build_request(req).json["input"]
    assert items[1] == {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": json.dumps({"a": 1})}
    assert items[2] == {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    body = {"status": "completed", "model": "m", "output": [
        {"type": "reasoning", "id": "r", "summary": []},
        {"type": "function_call", "call_id": "call_2", "name": "g", "arguments": '{"b": 2}'}],
        "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 40}, "output_tokens": 9}}
    r = openai_responses.parse_response(body)
    assert r.stop_reason == "tool_use" and r.tool_calls[0].input == {"b": 2}
    assert (r.usage.input_tokens, r.usage.cache_read_tokens) == (60, 40)


def test_l17_openai_adapter_reports_truncation():
    body = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "model": "m",
            "output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}
    assert openai_responses.parse_response(body).stop_reason == "max_tokens"


# Lecture 1.8
def test_l18_checkpoint_extracts_all_five_with_a_cost_report():
    res = checkpoint.extract_batch(client("checkpoint"), NOTICES, load_price())
    assert res.notices == EXPECTED and not res.failures
    assert res.cost.calls == 5 and res.cost.per_1000_calls_usd > 0
