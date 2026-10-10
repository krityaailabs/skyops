"""Lab 1.3: structured output you can trust, from three providers.

Hosted APIs can constrain output to a JSON schema natively. A local model may not follow the
schema every time, so validation is still yours, and a single repair attempt is the fallback.

Run:  python -m labs.s01.extract
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from skyops.model.client import ModelClient, get_client
from skyops.model.types import ChatRequest, Message, TextBlock

from .notices import NOTICES, SYSTEM, DisruptionNotice
from skyops.model.errors import LabNotDone


class ExtractionFailed(Exception):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


def extract_notice(client: ModelClient, text: str, max_repairs: int = 1) -> DisruptionNotice:
    """Ask for schema-constrained output, then VALIDATE it anyway.

    1. Send the schema with the request (native structured output).
    2. If the output was cut off or refused, stop: don't parse half an answer.
    3. Parse and validate with Pydantic: constrained decoding guarantees shape, not truth.
    4. If validation fails, send the error back once and ask for a corrected object.
    5. Still invalid: raise ExtractionFailed. Never pass unvalidated data downstream.
    """
    raise LabNotDone("Lecture 1.3: implement extract_notice(). Hints: labs/s01/README.md, lecture 1.3.")


def main(argv: list[str] | None = None) -> None:
    """Replay all three providers, or one with --provider. In live mode, set SKYOPS_MODE=live,
    SKYOPS_MODEL and the provider's key; only the provider you name is called."""
    import argparse
    from skyops.settings import Settings
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai", "ollama"],
                    default=s.provider if s.mode != "replay" else None)
    args = ap.parse_args(argv)
    for provider in [args.provider] if args.provider else ["anthropic", "openai", "ollama"]:
        client = get_client(fixture=f"fixtures/s01/extract_{provider}.jsonl", provider=provider, match="sequence")
        notice = extract_notice(client, NOTICES[0])
        print(f"{provider:9} calls={client.calls}  {notice.model_dump()}")


if __name__ == "__main__":
    main()
