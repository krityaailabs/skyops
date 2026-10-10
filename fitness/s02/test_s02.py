"""Section 2 fitness functions. Fail at s02-start (each names its lecture), pass at s02."""
import asyncio
import json
import time

import httpx
import pytest

from labs.s02 import assignment, concurrent, fakes
from labs.s02.contracts import FlightStatus
from skyops.agent import quickwin
from skyops.sim.async_ops import AsyncFlightOps
from skyops.sim.ops_api import FaultPlan, ops_transport
from skyops.tools import flights
from skyops.tools.tape import ToolReplayMiss, ToolTape

FLIGHTS = ["ZZ101", "ZZ118", "ZZ142", "ZZ156", "ZZ173", "ZZ188", "ZZ214", "ZZ230", "ZZ247", "ZZ261"]


# Lecture 2.1 (demo): the contract normalises and rejects
def test_l21_contract_normalises_and_rejects():
    s = FlightStatus.model_validate({"flight_number": "zz 214", "destination": "SIN", "scheduled_departure": "19:40",
                                     "estimated_departure": "20:15", "status": "DELAYED", "extra": 1})
    assert s.flight_number == "ZZ214"
    with pytest.raises(Exception):
        FlightStatus.model_validate({**s.model_dump(), "status": "LATE"})


# Lecture 2.2
def test_l22_bounded_concurrency_and_timeouts():
    ops = AsyncFlightOps(delays={"ZZ173": 0.5})
    t0 = time.perf_counter()
    res = asyncio.run(concurrent.fetch_all(FLIGHTS, ops.status, timeout_s=0.2, max_concurrency=4))
    elapsed = time.perf_counter() - t0
    assert [r.flight for r in res] == FLIGHTS, "results must keep input order"
    assert ops.peak <= 4, f"peak concurrency {ops.peak} exceeded the limit"
    bad = [r for r in res if not r.ok]
    assert [(r.flight, r.error) for r in bad] == [("ZZ173", "timeout")]
    assert elapsed < 0.45, "calls should overlap, and the slow one should be cut off at its timeout"


def test_l22_cancelling_the_batch_cancels_every_call():
    ops = AsyncFlightOps(default_delay=0.3)

    async def scenario():
        task = asyncio.create_task(concurrent.fetch_all(FLIGHTS, ops.status, timeout_s=1.0, max_concurrency=10))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        # checked inside the event loop: asyncio.run() would cancel any orphans on exit and hide them
        assert ops.in_flight == 0 and ops.cancelled == 10 and ops.finished == 0
    asyncio.run(scenario())


# Lecture 2.4
def test_l24_scripted_model_drives_the_real_loop():
    client, script = fakes.fake_client([
        fakes.tool_reply("get_flight_status", {"flight_number": "ZZ214", "date": "2026-10-09"}),
        fakes.text_reply("ZZ214 is delayed to 20:15 and leaves from gate B7.")])
    trace = quickwin.run("Is ZZ214 on time?", client)
    assert (trace.model_calls, trace.tool_calls) == (2, 1)
    second = script.requests[1]["messages"][-1]["content"][0]
    assert second["type"] == "tool_result" and "B7" in second["content"], "the tool result must go back to the model"


def test_l24_scripted_model_fails_loudly_when_it_runs_out():
    client, _ = fakes.fake_client([fakes.tool_reply("find_departures", {"destination": "SIN", "date": "2026-10-09"})])
    with pytest.raises(AssertionError, match="more model calls"):
        quickwin.run("Evening flight to Singapore?", client)


# Lecture 2.5
def test_l25_tool_tape_records_then_replays_without_the_real_tools(tmp_path):
    def make():
        return fakes.fake_client([
            fakes.tool_reply("get_flight_status", {"flight_number": "ZZ214", "date": "2026-10-09"}),
            fakes.text_reply("Delayed, gate B7.")])[0]
    rec = ToolTape.recorder()
    first = quickwin.run("ZZ214?", make(), tool_runner=rec.wrap(flights.run_tool))
    rec.save(tmp_path / "tape.jsonl")

    def boom(name, args):
        raise AssertionError("replay must not call the real tools")
    tape = ToolTape.load(tmp_path / "tape.jsonl")
    second = quickwin.run("ZZ214?", make(), tool_runner=tape.wrap(boom))
    assert [e for e in first.events if e["kind"] == "tool"] == [e for e in second.events if e["kind"] == "tool"]


def test_l25_tool_tape_rejects_a_different_call(tmp_path):
    tape = ToolTape("replay", [{"name": "get_flight_status", "args": {"flight_number": "ZZ214", "date": "2026-10-09"},
                                "ok": True, "output": "{}"}])
    with pytest.raises(ToolReplayMiss):
        tape.wrap(None)("get_flight_status", {"flight_number": "ZZ999", "date": "2026-10-09"})


# Lecture 2.7
def _client(plan, waits=None):
    async def sleep(s):
        (waits if waits is not None else []).append(s)
    return assignment.AsyncFlightOpsClient(transport=ops_transport(plan), sleep=sleep)


def _run(coro):
    return asyncio.run(coro)


def test_l27_returns_a_validated_contract():
    async def go():
        c = _client(FaultPlan())
        try:
            return await c.get_status("ZZ214", "2026-10-09")
        finally:
            await c.aclose()
    s = _run(go())
    assert isinstance(s, FlightStatus) and s.gate == "B7"


def test_l27_retries_503_and_honours_retry_after():
    waits, plan = [], FaultPlan({"ZZ214": [503, 429, 200]}, retry_after="3")

    async def go():
        c = _client(plan, waits)
        try:
            return await c.get_status("ZZ214", "2026-10-09")
        finally:
            await c.aclose()
    assert _run(go()).status == "DELAYED"
    assert waits == [1, 3.0] and plan.calls["ZZ214"] == 3


@pytest.mark.parametrize("outcome,exc,calls", [(404, "FlightNotFound", 1), (400, "OpsRejected", 1),
                                                ("malformed", "OpsDataError", 1), ("timeout", "OpsUnavailable", 3)])
def test_l27_errors_are_mapped(outcome, exc, calls):
    flight = "ZZ999" if outcome == 404 else "ZZ261"
    plan = FaultPlan({flight: [outcome] * 3} if outcome != 404 else {})

    async def go():
        c = _client(plan)
        try:
            await c.get_status(flight, "2026-10-09")
        finally:
            await c.aclose()
    with pytest.raises(getattr(assignment, exc)):
        _run(go())
    assert plan.calls[flight] == calls
