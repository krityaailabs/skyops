"""Lab 1.4: your first cost-per-call calculation, from usage and price parameters.

Run:  python -m labs.s01.costing
"""
from __future__ import annotations

from dataclasses import dataclass

from skyops.cost import Price, cost_usd, load_price, per_thousand
from skyops.model.types import ModelResponse, Usage
from skyops.model.errors import LabNotDone


@dataclass(frozen=True)
class CostReport:
    calls: int
    input_tokens: int
    output_tokens: int
    total_usd: float
    per_call_usd: float
    per_1000_calls_usd: float
    input_share: float      # share of the cost that came from input tokens, 0..1


def report(responses: list[ModelResponse], price: Price) -> CostReport:
    """Total the authoritative usage from each response, then price it.

    Use the usage the provider returned, never an estimate. Report per 1,000 calls: a per-call
    figure is too small to compare or to put in a budget.
    """
    raise LabNotDone("Lecture 1.4: implement report(). Hints: labs/s01/README.md, lecture 1.4.")


def main() -> None:
    from skyops.model.client import get_client
    from .notices import NOTICES
    from .extract import extract_notice
    client = get_client(fixture="fixtures/s01/checkpoint.jsonl", match="sequence")
    responses: list[ModelResponse] = []
    original = client.complete

    def capture(req):
        r = original(req)
        responses.append(r)
        return r
    client.complete = capture
    for text in NOTICES:
        extract_notice(client, text)
    rep = report(responses, load_price())
    print(rep)


if __name__ == "__main__":
    main()
