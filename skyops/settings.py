"""Settings from the environment. No secrets in code, no secrets in the repository."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = {
    # [VOLATILE] model IDs verified 2026-10-09 (models overview pages). Override with SKYOPS_MODEL.
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-6-luna",
    "ollama": "qwen3:8b",   # [VERIFY] any local model with tool calling works
}


@dataclass(frozen=True)
class Settings:
    mode: Literal["replay", "live", "record"] = "replay"
    provider: str = "anthropic"
    model: str = DEFAULT_MODELS["anthropic"]
    max_usd_per_run: float = 0.50
    fixture: Path | None = None
    replay_match: Literal["sequence", "strict"] = "strict"
    price_profile: str = "example"
    ollama_host: str = "http://localhost:11434"

    def __post_init__(self) -> None:
        # A dataclass doesn't check Literal types at runtime, so check them here: a typo in
        # SKYOPS_MODE must fail at start-up, never fall through to live calls.
        checks = {"mode": ("replay", "record", "live"), "replay_match": ("sequence", "strict"),
                  "provider": tuple(DEFAULT_MODELS)}
        for name, allowed in checks.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"SKYOPS setting {name}={getattr(self, name)!r} is not one of {allowed}")
        if self.max_usd_per_run < 0:
            raise ValueError("SKYOPS_MAX_USD_PER_RUN must not be negative")

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        provider = overrides.get("provider") or os.environ.get("SKYOPS_PROVIDER", "anthropic")
        fixture = overrides.get("fixture") or os.environ.get("SKYOPS_FIXTURE")
        return cls(
            mode=overrides.get("mode") or os.environ.get("SKYOPS_MODE", "replay"),
            provider=provider,
            model=overrides.get("model") or os.environ.get("SKYOPS_MODEL") or DEFAULT_MODELS.get(provider, ""),
            max_usd_per_run=float(overrides.get("max_usd_per_run") or os.environ.get("SKYOPS_MAX_USD_PER_RUN", "0.50")),
            fixture=Path(fixture) if fixture else None,
            replay_match=overrides.get("replay_match") or os.environ.get("SKYOPS_REPLAY_MATCH", "strict"),
            price_profile=overrides.get("price_profile") or os.environ.get("SKYOPS_PRICE_PROFILE", "example"),
            ollama_host=overrides.get("ollama_host") or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
