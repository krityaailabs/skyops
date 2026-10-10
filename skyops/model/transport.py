"""HTTP transports: live, record and replay.

Replay happens at the HTTP layer, so the provider adapters run exactly the same
code in replay mode as in live mode. Only the bytes come from a file.

Fixture format (one JSON object per line):
  {"provider", "request": {"url", "json"}, "key", "status", "headers", "body" | "events", "meta"}
`meta` records the model, SDK/API version, record date and whether the record is synthetic
(hand-written during scripting) or recorded from a live provider.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol


@dataclass
class HttpRequest:
    provider: str
    url: str
    headers: dict[str, str]
    json: dict[str, Any]
    stream: bool = False


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | None = None
    events: list[dict[str, Any]] | None = None   # server-sent events, for streaming


def request_key(req: HttpRequest) -> str:
    """A stable fingerprint of what was asked. Headers (and so API keys) are never part of it."""
    canon = json.dumps({"url": req.url, "json": req.json}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


class Transport(Protocol):
    def send(self, req: HttpRequest) -> HttpResponse: ...


class ReplayMiss(RuntimeError):
    pass


class ReplayTransport:
    """Serve recorded responses in order.

    match="sequence": the nth call gets the nth recording; a different request only warns.
        Labs use this, so a learner's slightly different prompt still runs.
    match="strict": the request fingerprint must equal the recording. CI and tests use this.
    """

    def __init__(self, path: str | Path, match: Literal["sequence", "strict"] = "strict"):
        self.path = Path(path)
        if not self.path.exists():
            raise ReplayMiss(f"No fixture file {self.path}. Run in live mode with `make record`, "
                             "or check SKYOPS_FIXTURE.")
        self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.match = match
        self.position = 0

    @property
    def synthetic_records(self) -> int:
        return sum(1 for r in self.records if r.get("meta", {}).get("synthetic"))

    def send(self, req: HttpRequest) -> HttpResponse:
        if self.position >= len(self.records):
            raise ReplayMiss(f"{self.path.name}: the run made more model calls than were recorded "
                             f"({len(self.records)}). The agent probably looped; check its step budget.")
        rec = self.records[self.position]
        self.position += 1
        if rec["provider"] != req.provider:
            raise ReplayMiss(f"{self.path.name} call {self.position}: recorded for {rec['provider']}, "
                             f"but the client is {req.provider}.")
        key = request_key(req)
        if rec.get("key") and rec["key"] != key:
            msg = (f"{self.path.name} call {self.position}: your request differs from the recording "
                   f"(recorded {rec['key']}, sent {key}).")
            if self.match == "strict":
                raise ReplayMiss(msg + " In strict mode that is an error.")
            warnings.warn(msg + " Replaying the recorded response anyway.", stacklevel=2)
        return HttpResponse(status=rec["status"], headers=rec.get("headers", {}),
                            body=rec.get("body"), events=rec.get("events"))


class LiveTransport:
    def __init__(self, timeout_s: float = 60.0):
        import httpx
        self._client = httpx.Client(timeout=timeout_s)

    def send(self, req: HttpRequest) -> HttpResponse:
        if req.stream:
            with self._client.stream("POST", req.url, headers=req.headers, json=req.json) as r:
                if r.status_code != 200:
                    r.read()
                    return HttpResponse(r.status_code, dict(r.headers), body=_json_or_text(r.text))
                return HttpResponse(r.status_code, dict(r.headers), events=list(parse_sse(r.iter_lines())))
        r = self._client.post(req.url, headers=req.headers, json=req.json)
        return HttpResponse(r.status_code, dict(r.headers), body=_json_or_text(r.text))


class RecordingTransport:
    """Wrap a live transport and append every exchange to a fixture file."""

    def __init__(self, inner: Transport, path: str | Path, model: str, api_version: str = ""):
        self.inner, self.path, self.model, self.api_version = inner, Path(path), model, api_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")   # one recording session = one fresh fixture file

    def send(self, req: HttpRequest) -> HttpResponse:
        resp = self.inner.send(req)
        keep = {k: v for k, v in resp.headers.items()
                if k.lower() in ("retry-after", "request-id", "x-request-id", "content-type")}
        rec = {"provider": req.provider, "request": {"url": req.url, "json": req.json}, "key": request_key(req),
               "status": resp.status, "headers": keep, "meta": {
                   "model": self.model, "api_version": self.api_version, "synthetic": False,
                   "recorded": _dt.date.today().isoformat()}}
        rec["events" if resp.events is not None else "body"] = resp.events if resp.events is not None else resp.body
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return resp


def parse_sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Minimal server-sent-events parser: yields {"event": name, "data": parsed JSON}."""
    event, data = None, []
    for line in lines:
        if line == "":
            if data:
                payload = "\n".join(data)
                yield {"event": event, "data": None if payload == "[DONE]" else json.loads(payload)}
            event, data = None, []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        yield {"event": event, "data": json.loads("\n".join(data))}


def _json_or_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
