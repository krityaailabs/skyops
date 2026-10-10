"""Section 5 fitness: idempotency and dry run, the error taxonomy, result shaping, the contract lint and
the contracted SkyOps catalog. Fail at s05-start, pass at s05."""
import datetime as dt
import json

import pytest

from labs.s05 import descriptions, disruption, review_tools
from skyops.agent.loop import execute
from skyops.model.types import ToolUseBlock
from skyops.tools.catalog import (Gateways, ListDeparturesArgs, NotifyArgs, VoucherArgs, check_voucher,
                                  list_departures, make_catalog, toolbox)
from skyops.tools.contracts import (IdempotencyStore, ToolError, classify_exception, estimate_tokens,
                                    execute_write, idempotency_key, lint_contract, shape_result)

MSG = "ZZ214 now departs 20:15 from gate B7. We are sorry for the wait."


def notify_contract():
    cs, gw = make_catalog()
    return next(c for c in cs if c.name == "notify_passengers"), gw


def test_l52_descriptions_change_tool_choice():
    vague, precise = descriptions.measure("vague"), descriptions.measure("precise")
    assert sum(c.correct for c in precise) == len(precise)
    assert sum(c.correct for c in vague) < len(vague)


def test_l55_idempotency_and_dry_run():
    c, gw = notify_contract()
    a = NotifyArgs(flight_number="zz 214", message=MSG, reason="delay")
    k1, k2 = idempotency_key(c, a, "disruption-1"), idempotency_key(c, a, "disruption-2")
    assert len(k1) == 24 and k1 != k2
    assert idempotency_key(c, NotifyArgs(flight_number="ZZ214", message=MSG, reason="gate_change"), "disruption-1") == k1
    assert idempotency_key(c, NotifyArgs(flight_number="ZZ214", message="  " + MSG.upper(), reason="delay"),
                           "disruption-1") == k1     # spacing and case don't make a new action
    store = IdempotencyStore()
    first = execute_write(c, a, store, "disruption-1")
    again = execute_write(c, a, store, "disruption-1")
    assert first["replayed"] is False and again["replayed"] is True and len(gw.sent) == 1
    dry = execute_write(c, NotifyArgs(flight_number="ZZ230", message=MSG, reason="delay"), store, "d", dry_run=True)
    assert dry["dry_run"] is True and dry["would_replay"] is False and len(gw.sent) == 1
    again = execute_write(c, NotifyArgs(flight_number="ZZ230", message=MSG, reason="delay"), store, "d", dry_run=True)
    assert again["would_replay"] is True
    with pytest.raises(ToolError) as e:   # same key, different details: a conflict, not a silent replay
        execute_write(c, NotifyArgs(flight_number="zz214", message=MSG, reason="gate_change"), store, "disruption-1")
    assert e.value.code == "idempotency_conflict"


def test_l56_error_taxonomy():
    class OpsUnavailable(Exception):
        retryable = True
    assert classify_exception(TimeoutError("x")).as_result() == {"error": "unavailable", "category": "retryable",
                                                                  "message": "TimeoutError", "retryable": True}
    assert classify_exception(OpsUnavailable("down")).category == "retryable"
    assert classify_exception(PermissionError("no")).category == "needs_human"
    assert classify_exception(KeyError("secret=abc")).as_result()["message"] == "KeyError"
    e = ToolError("needs_human", "approval_required", "too much")
    assert classify_exception(e) is e


def test_l57_shaping_and_pagination():
    rows = [{"a": i, "b": "x" * 40, "c": "drop me"} for i in range(12)]
    p1 = shape_result(rows, ["a", "b"], page_size=5)
    assert [r["a"] for r in p1["items"]] == [0, 1, 2, 3, 4] and p1["next_cursor"] == 5 and p1["total"] == 12
    assert "c" not in p1["items"][0] and p1["truncated"] is False
    last = shape_result(rows, ["a"], page_size=5, cursor=10)
    assert [r["a"] for r in last["items"]] == [10, 11] and last["next_cursor"] is None
    capped = shape_result(rows, ["a", "b"], page_size=10, max_tokens=40)
    assert capped["truncated"] is True and estimate_tokens(capped["items"]) <= 40 and capped["next_cursor"] == len(capped["items"])
    huge = shape_result([{"a": "x" * 4000}], ["a"], page_size=5, max_tokens=50)
    assert len(huge["items"]) == 1 and huge["oversized"] is True and estimate_tokens(huge["items"]) <= 50


def test_l58_lint_finds_what_code_can_find():
    for t in review_tools.TOOLS:
        expected = {code for kind, code in review_tools.FINDINGS[t.name] if kind == "lint"}
        assert set(lint_contract(t)) == expected, t.name


def test_l59_catalog_contracts():
    contracts, _ = make_catalog()
    assert all(lint_contract(c) == [] for c in contracts)
    page = list_departures(ListDeparturesArgs(date=dt.date(2026, 10, 9)))
    assert len(page["items"]) == 5 and isinstance(page["next_cursor"], str) and set(page["items"][0]) == {
        "flight_number", "destination", "scheduled_departure", "status"}
    p2 = list_departures(ListDeparturesArgs(date=dt.date(2026, 10, 9), cursor=page["next_cursor"]))
    assert p2["next_cursor"] is None and p2["items"][0]["flight_number"] != page["items"][0]["flight_number"]
    assert list_departures(ListDeparturesArgs(date=dt.date(2026, 10, 9), destination="sin"))["total"] == 2
    with pytest.raises(ToolError) as e:
        check_voucher(VoucherArgs(booking_ref="Q7K2LM", flight_number="ZZ247", kind="hotel", amount_eur=180))
    assert e.value.category == "needs_human" and e.value.code == "approval_required"
    with pytest.raises(ToolError) as e:
        check_voucher(VoucherArgs(booking_ref="Q7K2LM", flight_number="ZZ230", kind="meal", amount_eur=20))
    assert e.value.code == "flight_not_disrupted"
    with pytest.raises(ToolError) as e:     # the rules run in order: existence before the limit
        check_voucher(VoucherArgs(booking_ref="Q7K2LM", flight_number="ZZ999", kind="hotel", amount_eur=500))
    assert e.value.code == "not_found"
    check_voucher(VoucherArgs(booking_ref="Q7K2LM", flight_number="ZZ247", kind="meal", amount_eur=100))  # allowed
    box, gw, _ = toolbox(scope="t")
    ok = execute(box, ToolUseBlock(id="v", name="issue_voucher", input={"booking_ref": "Q7K2LM", "flight_number": "ZZ247",
                                                                       "kind": "meal", "amount_eur": 100}))
    assert not ok.is_error and len(gw.vouchers) == 1
    r, gw = disruption.run()
    assert len(gw.sent) == 1 and gw.vouchers == [] and r.stop == "answered"
    assert any("approval_required" in e.get("summary", "") for e in r.trace.events if e["kind"] == "tool")
    r, gw = disruption.run(dry_run=True)
    assert gw.sent == [] and gw.vouchers == []
