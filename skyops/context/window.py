"""Lecture 8.7 (Incident 08): fitting a request into the window without ever losing the pinned rules,
and the detection signal that proves it on every request.

Two functions you write at s08-start:
    pinned_missing(req, blocks)   the detection signal: which pinned blocks are not intact in this request
    fit_to_window(req, window, blocks)   the fix: make a request fit by trimming history only, or fail loudly

A "turn" starts at a user message that contains text from a person. Tool results also travel in user
messages, but they belong to the turn of the assistant message that asked for them: a cut between a
tool call and its result would leave the provider an orphaned result, which providers reject.
"""
from __future__ import annotations

from ..model.errors import LabNotDone
from ..model.types import ChatRequest, Message, TextBlock
from .budget import Breakdown, Window, account
from .pinned import PinnedBlock


class ContextOverflow(Exception):
    """The request can't fit even with the oldest history removed. Stop the run; never send it trimmed."""

    def __init__(self, breakdown: Breakdown, limit: int):
        super().__init__(f"request needs about {breakdown.total} tokens with only the last turn; "
                         f"the window allows {limit} ({breakdown.line()})")
        self.breakdown, self.limit = breakdown, limit


class PinnedRulesMissing(Exception):
    """A request was about to leave without its pinned rules intact. Never send it."""

    def __init__(self, missing: list[str]):
        super().__init__(f"pinned blocks missing or altered: {', '.join(missing)}")
        self.missing = missing


def turn_starts(messages: list[Message]) -> list[int]:
    """Indices of the messages that start a turn: user messages with at least one TextBlock."""
    return [i for i, m in enumerate(messages)
            if m.role == "user" and any(isinstance(b, TextBlock) for b in m.content)]


def pinned_missing(req: ChatRequest, blocks: list[PinnedBlock]) -> list[str]:
    """Lecture 8.7: the detection signal. Return the ids of the blocks that are NOT intact in req.system,
    in the order of `blocks`; an empty list means every pinned block is there, verbatim.

    A block is intact only if req.system contains block.render() exactly: the marker line (with the hash
    of the exact text), the text unchanged, and the closing tag. Checking for the marker alone is not
    enough: in Incident 08 the marker survived and the rules after it were cut. Copies of the rules in the
    messages don't count: pinned text belongs in the system prompt, where trimming history can't reach it.
    A req.system of None means every block is missing.
    """
    system = req.system or ""
    return [b.id for b in blocks if b.render() not in system]


def fit_to_window(req: ChatRequest, window: Window, blocks: list[PinnedBlock]) -> tuple[ChatRequest, int]:
    """Lecture 8.7: the fix. Return (a request that fits window.input_limit, number of turns dropped).

    1. If account(req).total <= window.input_limit, the result is (req, 0): the same object, unchanged.
    2. Otherwise drop whole turns from the START of req.messages, oldest first, one turn at a time
       (a turn runs from one index in turn_starts(req.messages) to the next), until it fits. Messages
       before the first turn start count as part of the first turn. Never drop the last turn.
    3. Never change req.system or req.tools: return req.model_copy(update={"messages": kept}).
    4. If the request still doesn't fit with only the last turn, raise ContextOverflow(account(that
       request), window.input_limit): a run that stops loudly is better than a request without its rules.
    5. Before returning any result (including case 1), if pinned_missing(result, blocks) is not empty,
       raise PinnedRulesMissing(missing): fitting must never hide a request that lost its rules elsewhere.
    """
    out, dropped = req, 0
    if account(req).total > window.input_limit:
        starts = turn_starts(req.messages) or [0]
        for cut in starts[1:]:               # keeping messages from a later turn = dropping one more turn
            dropped += 1
            out = req.model_copy(update={"messages": req.messages[cut:]})
            if account(out).total <= window.input_limit:
                break
        else:
            last = req.model_copy(update={"messages": req.messages[starts[-1]:]})
            raise ContextOverflow(account(last), window.input_limit)
    missing = pinned_missing(out, blocks)
    if missing:
        raise PinnedRulesMissing(missing)
    return out, dropped
