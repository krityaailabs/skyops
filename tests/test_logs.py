import io
import json

from skyops.logs import get_logger, redact, run_context


def test_run_id_and_extra_fields_on_every_line():
    buf = io.StringIO()
    log = get_logger("t1", buf)
    with run_context("run-7f3a"):
        log.info("tool call", extra={"tool": "get_flight_status"})
    line = json.loads(buf.getvalue())
    assert line == {"level": "INFO", "msg": "tool call", "run_id": "run-7f3a", "tool": "get_flight_status"}


def test_secrets_are_redacted():
    s = redact("headers x-api-key: sk-ant-abcdef123456 and Bearer abcdefghijkl")
    assert "abcdef123456" not in s and "abcdefghijkl" not in s and s.count("[REDACTED]") >= 2
