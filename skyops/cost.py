"""Cost from token usage and price *parameters*.

Prices live in config/prices.toml, never in code and never in narration. They change, they differ by
region and tier, and a hard-coded price is a bug that waits quietly for the next price change.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .model.types import Usage

PRICES_FILE = Path(__file__).resolve().parent.parent / "config" / "prices.toml"


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""
    name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0
    source: str = ""
    verified: str = ""


class MissingPrice(LookupError):
    pass


def load_price(name: str | None = None, path: Path = PRICES_FILE) -> Price:
    """Load a price profile. Defaults to SKYOPS_PRICE_PROFILE, then the course example profile."""
    if name is None:
        from .settings import Settings
        name = Settings.from_env().price_profile
    table = tomllib.loads(path.read_text(encoding="utf-8"))
    if name not in table:
        raise MissingPrice(f"No price profile '{name}' in {path.name}. Add it from the provider's pricing page, "
                           "with the date you checked it.")
    row = table[name]
    if not row.get("input_per_mtok") and not row.get("output_per_mtok"):
        raise MissingPrice(f"Price profile '{name}' is empty. Fill it from the provider's pricing page.")
    return Price(name=name, **row)


def cost_usd(usage: Usage, price: Price) -> float:
    """Cost of one call: each token class times its price per million."""
    return (usage.input_tokens * price.input_per_mtok
            + usage.output_tokens * price.output_per_mtok
            + usage.cache_read_tokens * price.cache_read_per_mtok
            + usage.cache_write_tokens * price.cache_write_per_mtok) / 1_000_000


def per_thousand(cost_one: float) -> float:
    """Report costs per 1,000 tasks: per-call costs are too small to read or to reason about."""
    return cost_one * 1000
