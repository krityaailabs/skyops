"""Section 3 fitness: the five breaks happen as described, and your detectors flag each one,
without flagging the healthy quick-win run. Fail at s03-start, pass at s03."""
import pytest

from labs.s03 import break_it, detectors
from skyops.sim.faults import CANARY


@pytest.fixture(scope="module")
def reports():
    return {n: break_it.run_scenario(n) for n in break_it.SCENARIOS}


@pytest.fixture(scope="module")
def healthy():
    return break_it.run_healthy()


def test_l33_the_breaks_happen(reports):
    assert reports["loop"].answer.startswith("[stopped: step budget")
    assert reports["slow_tool"].clock.elapsed_ms >= break_it.SLOW_MS
    assert len(reports["duplicate"].outbox) == 2
    assert any(CANARY in m["message"] for m in reports["injection"].outbox)


def test_l33_detect_loop(reports, healthy):
    assert detectors.detect_loop(reports["loop"].trace) and not detectors.detect_loop(healthy.trace)


def test_l33_detect_bad_arguments(reports, healthy):
    assert detectors.detect_bad_arguments(reports["bad_arguments"].trace) == 1
    assert detectors.detect_bad_arguments(healthy.trace) == 0


def test_l33_detect_slow_tool(reports, healthy):
    assert detectors.detect_slow_tool(reports["slow_tool"].tool_latencies)
    assert not detectors.detect_slow_tool(healthy.tool_latencies)
    assert not detectors.detect_slow_tool(reports["bad_arguments"].tool_latencies)


def test_l33_detect_duplicate_side_effects(reports, healthy):
    assert detectors.detect_duplicate_side_effects(reports["duplicate"].outbox) == 1
    assert detectors.detect_duplicate_side_effects(healthy.outbox) == 0
    assert detectors.detect_duplicate_side_effects(reports["injection"].outbox) == 0


def test_l33_detect_canary(reports, healthy):
    assert detectors.detect_canary(reports["injection"].outbox, [CANARY])
    assert not detectors.detect_canary(healthy.outbox, [CANARY])
    assert not detectors.detect_canary(reports["duplicate"].outbox, [CANARY])
