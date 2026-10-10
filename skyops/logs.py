"""Lecture 2.6 (demo): structured logs with a run id on every line, and secrets that never get logged.

    with run_context("run-7f3a"):
        log.info("tool call", extra={"tool": "get_flight_status"})
    -> {"level": "INFO", "msg": "tool call", "run_id": "run-7f3a", "tool": "get_flight_status"}

The run id travels through async tasks via a context variable, so concurrent runs never mix their lines.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re

RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|(?i:bearer)\s+[A-Za-z0-9._-]{8,}|(?i:x-api-key)[\"':= ]+[A-Za-z0-9._-]{8,}"
                    r"|(?i:password|passwd|secret|api_key|token)\s*[=:]\s*[^\s,;]+)")
STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


def redact(text: str) -> str:
    return SECRET.sub("[REDACTED]", text)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"level": record.levelname, "msg": redact(record.getMessage()), "run_id": RUN_ID.get()}
        out.update({k: (redact(v) if isinstance(v, str) else v) for k, v in vars(record).items() if k not in STANDARD})
        return json.dumps(out, sort_keys=True)


def get_logger(name: str = "skyops", stream=None) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(stream)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


@contextlib.contextmanager
def run_context(run_id: str):
    token = RUN_ID.set(run_id)
    try:
        yield
    finally:
        RUN_ID.reset(token)
