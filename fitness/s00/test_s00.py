"""Section 0 fitness: the repository is set up and the quick-win agent runs in replay mode."""
import sys

from skyops.agent import quickwin
from skyops.model.client import get_client
from skyops.settings import ROOT, Settings


def run_quickwin():
    s = Settings(mode="replay", provider="anthropic", model="claude-haiku-5-5",
                 fixture=ROOT / "fixtures/s00/quickwin.jsonl", replay_match="strict")
    return quickwin.run(quickwin.QUESTION, get_client(s))


def test_python_version():
    assert sys.version_info >= (3, 11), "SkyOps needs Python 3.11 or newer"


def test_quickwin_answers_from_tool_results():
    t = run_quickwin()
    answer = t.events[-1]["text"]
    assert "B7" in answer and "20:15" in answer
    assert (t.model_calls, t.tool_calls) == (3, 2)


def test_quickwin_is_deterministic_in_replay():
    a, b = run_quickwin().summary(), run_quickwin().summary()
    assert a == b


def test_quickwin_has_a_step_budget():
    assert 1 <= quickwin.MAX_STEPS <= 10
