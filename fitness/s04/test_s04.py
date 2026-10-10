"""Section 4 fitness: the plain-Python agent loop, its budgets, error handling, runaway guard, the
ten-case eval and Project 1. Fail at s04-start, pass at s04."""
import json

import pytest

from labs.s04 import runaway
from labs.s04.cases import CASES
from labs.s04.eval import check_case, run_eval
from projects.p1_flight_status.assistant import answer
from skyops.agent.loop import (Budgets, Tool, ToolBox, check_budgets, detect_runaway, execute, run_agent,
                               validate_call)
from skyops.model.client import get_client
from skyops.model.types import ToolUseBlock
from skyops.tools.flights import FlightStatusArgs


def client(case_id):
    return get_client(fixture=f"fixtures/s04/{case_id}.jsonl", match="strict")


def call(name, **args):
    return ToolUseBlock(id="toolu_t", name=name, input=args)


def test_l42_loop_answers_with_tools():
    r = run_agent(CASES[0].question, client("c01_sin_evening"), ToolBox.flight_tools())
    assert r.stop == "answered" and r.steps == 3
    assert r.tools_used == ["find_departures", "get_flight_status"]
    assert "B7" in r.answer
    # a rewrite of the loop must reproduce a healthy run exactly: the Section 0 recording replays strictly
    from skyops.agent import quickwin
    q = run_agent(quickwin.QUESTION, get_client(fixture="fixtures/s00/quickwin.jsonl", match="strict"),
                  ToolBox.flight_tools())
    assert q.stop == "answered" and q.steps == 3


def test_l43_validation_happens_before_execution():
    box = ToolBox.flight_tools()
    args, err = validate_call(box, call("get_flight_status", flight_number="ZZ214", date="2026-10-09"))
    assert isinstance(args, FlightStatusArgs) and err is None
    args, err = validate_call(box, call("delete_flight", flight_number="ZZ214"))
    assert args is None and json.loads(err.content)["error"] == "unknown_tool" and err.is_error
    args, err = validate_call(box, call("find_departures", destination="Bengaluru", date="2026-10-09"))
    body = json.loads(err.content)
    assert body["error"] == "invalid_arguments" and body["message"].startswith("destination") and not body["retryable"]


def test_l44_budgets():
    b = Budgets(max_steps=3, max_tokens=1000, max_seconds=10)
    assert check_budgets(b, 0, 0, 0.0) is None
    assert check_budgets(b, 3, 0, 0.0) == "step_budget"
    assert check_budgets(b, 1, 1000, 0.0) == "token_budget"
    assert check_budgets(b, 1, 0, 10.0) == "time_budget"
    r = run_agent(CASES[0].question, client("c01_sin_evening"), ToolBox.flight_tools(), budgets=Budgets(max_steps=1))
    assert r.stop == "step_budget" and r.steps == 1 and "step budget" in r.answer


def test_l45_errors_become_data():
    def boom(_):
        raise TimeoutError("upstream")

    def bug(_):
        raise KeyError("gate")
    box = ToolBox.flight_tools()
    t = box.tools["get_flight_status"]
    good = call("get_flight_status", flight_number="ZZ214", date="2026-10-09")
    box.add(Tool(t.name, t.args_model, boom, t.description))
    res = json.loads(execute(box, good).content)
    assert res["error"] == "tool_unavailable" and res["retryable"] is True
    box.add(Tool(t.name, t.args_model, bug, t.description))
    res = execute(box, good)
    assert res.is_error and json.loads(res.content) == {"error": "tool_failed", "message": "get_flight_status failed: KeyError", "retryable": False}
    box.add(Tool(t.name, t.args_model, lambda _: None, t.description))      # a tool that returns nothing
    assert json.loads(execute(box, good).content)["error"] == "tool_failed"
    nf = execute(ToolBox.flight_tools(), call("get_flight_status", flight_number="ZZ999", date="2026-10-09"))
    assert nf.is_error and json.loads(nf.content)["error"] == "not_found" and json.loads(nf.content)["retryable"] is False
    r = run_agent(CASES[4].question, client("c05_bad_args_recover"), ToolBox.flight_tools())
    assert r.stop == "answered" and r.trace.events[1]["ok"] is False


def test_l46_runaway_is_capped():
    assert detect_runaway([("a", {"x": 1})] * 3, 3)
    assert not detect_runaway([("a", {"x": 1}), ("a", {"x": 2}), ("a", {"x": 1})], 3)
    pairs = {n: runaway.run_pair(n) for n in runaway.QUESTIONS}
    assert pairs["repeat"].capped.stop == "repeated_call" and pairs["repeat"].capped.steps == 3
    assert pairs["sweep"].capped.stop == "token_budget"
    assert pairs["slow"].capped.stop == "time_budget"
    for p in pairs.values():
        assert p.capped.trace.usage.total < p.uncapped.trace.usage.total


def test_l47_eval_scores_and_explains():
    from skyops.agent.loop import RunResult
    c = CASES[1]
    ok = check_case(c, RunResult("ZZ247 is cancelled.", "answered", None, 2, [("get_flight_status", {})]))
    assert ok.passed and ok.failures == []
    bad = check_case(c, RunResult("ZZ247 leaves from gate B4.", "answered", None, 2, []))
    assert bad.failures == ["missing: cancel", "tool not called: get_flight_status", "forbidden: gate B"]
    rep = run_eval()
    assert rep.passed == 9 and len(rep.results) == 10
    bad = [r for r in rep.results if not r.passed]
    assert [r.id for r in bad] == ["c10_status_slip"]
    assert "forbidden: is on time" in bad[0].failures and len(bad[0].failures) == 3


def test_l48_project1_acceptance():
    for c in CASES:
        a = answer(c.question, client(c.id))
        text = a.text.lower()
        assert a.grounded, c.id
        assert all(any(p.lower() in text for p in g) for g in c.answer_has), (c.id, a.text)
        assert not any(p.lower() in text for p in c.forbid), (c.id, a.text)
    a = answer(CASES[9].question, client("c10_status_slip"))
    assert a.status == "DELAYED"
    a = answer(CASES[0].question, client("c01_sin_evening"), budgets=Budgets(max_steps=1))
    assert not a.grounded and a.stop == "step_budget"
