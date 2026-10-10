"""Errors a model call can raise, classified so the caller can act on them.

The classification matters more than the message: the agent loop and the retry
policy in later sections decide what to do from `kind` and `retryable` alone.
"""
from __future__ import annotations

from typing import Literal

ErrorKind = Literal[
    "rate_limited",      # HTTP 429: slow down; honour retry_after when present
    "overloaded",        # Anthropic HTTP 529: the provider is busy, not you
    "server_error",      # HTTP 5xx
    "invalid_request",   # HTTP 400: your request is wrong; retrying will not help
    "auth",              # HTTP 401 or 403
    "budget_exceeded",   # SkyOps spend guard refused the call before it was sent
    "other",
]

RETRYABLE: set[str] = {"rate_limited", "overloaded", "server_error"}


class ModelError(Exception):
    def __init__(self, kind: ErrorKind, message: str, *, status: int | None = None,
                 retry_after: float | None = None, provider: str = ""):
        super().__init__(f"[{provider}:{kind}] {message}")
        self.kind = kind
        self.status = status
        self.retry_after = retry_after
        self.provider = provider

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE


def kind_from_status(status: int) -> ErrorKind:
    if status == 429:
        return "rate_limited"
    if status == 529:
        return "overloaded"
    if status in (401, 403):
        return "auth"
    if status in (400, 404, 413, 422):
        return "invalid_request"
    if status >= 500:
        return "server_error"
    return "other"


class LabNotDone(NotImplementedError):
    """Raised by lab stubs so a half-finished lab fails with directions, not a traceback."""
