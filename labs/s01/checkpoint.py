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
from skyops.model.errors import LabNotDone


@dataclass
class BatchResult:
    notices: list[DisruptionNotice | None]
    failures: dict[int, str] = field(default_factory=dict)
    cost: CostReport | None = None


def extract_batch(client: ModelClient, texts: list[str], price: Price) -> BatchResult:
    raise LabNotDone("Lecture 1.8: implement extract_batch(). Hints: labs/s01/README.md, lecture 1.8.")


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
