"""Lab 1.8, foundations checkpoint: a typed batch extractor with a cost report.

Requirements (the hidden grading tests check these):
- every notice is attempted, even if an earlier one fails;
- a failed extraction is reported, not raised, and never returned as data;
- the cost report uses the provider's usage for every call made, including repair calls.

Run:  python -m labs.s01.checkpoint
"""
from __future__ import annotations

from dataclasses import dataclass, field

from skyops.cost import Price, load_price
from skyops.model.client import ModelClient, get_client
from skyops.model.types import ModelResponse

from .costing import CostReport, report
from .extract import ExtractionFailed, extract_notice
from .notices import NOTICES, DisruptionNotice


@dataclass
class BatchResult:
    notices: list[DisruptionNotice | None]
    failures: dict[int, str] = field(default_factory=dict)
    cost: CostReport | None = None


def extract_batch(client: ModelClient, texts: list[str], price: Price) -> BatchResult:
    responses: list[ModelResponse] = []
    original = client.complete

    def counted(req):                       # every call is priced, including repairs
        r = original(req)
        responses.append(r)
        return r

    client.complete = counted
    try:
        out = BatchResult(notices=[])
        for i, text in enumerate(texts):
            try:
                out.notices.append(extract_notice(client, text))
            except ExtractionFailed as e:
                out.notices.append(None)
                out.failures[i] = str(e)
        out.cost = report(responses, price)
        return out
    finally:
        client.complete = original


def main() -> None:
    client = get_client(fixture="fixtures/s01/checkpoint.jsonl", match="sequence")
    result = extract_batch(client, NOTICES, load_price())
    for n in result.notices:
        print(n.model_dump() if n else None)
    c = result.cost
    print(f"{c.calls} calls, {c.input_tokens} input and {c.output_tokens} output tokens, "
          f"{c.per_1000_calls_usd:.2f} USD per 1,000 calls ({load_price().name} prices), "
          f"input share {c.input_share:.0%}, failures: {result.failures or 'none'}")


if __name__ == "__main__":
    main()
