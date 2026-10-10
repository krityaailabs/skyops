"""Lab 1.4: your first cost-per-call calculation, from usage and price parameters.

Run:  python -m labs.s01.costing
"""
from __future__ import annotations

from dataclasses import dataclass

from skyops.cost import Price, cost_usd, load_price, per_thousand
from skyops.model.types import ModelResponse, Usage


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
    usage = sum((r.usage for r in responses), Usage())
    total = cost_usd(usage, price)
    input_cost = cost_usd(Usage(input_tokens=usage.input_tokens, cache_read_tokens=usage.cache_read_tokens,
                                cache_write_tokens=usage.cache_write_tokens), price)
    per_call = total / len(responses) if responses else 0.0
    return CostReport(calls=len(responses), input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                      total_usd=total, per_call_usd=per_call, per_1000_calls_usd=per_thousand(per_call),
                      input_share=input_cost / total if total else 0.0)


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
