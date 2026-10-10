"""ModelClient: one interface over every provider, in replay, record or live mode.

    client = get_client()                      # reads SKYOPS_* settings
    resp = client.complete(ChatRequest(...))   # same call for Anthropic, OpenAI or a local model

The client does three things besides calling the provider:
1. Refuses a live call that could take the run over its spend cap (SKYOPS_MAX_USD_PER_RUN).
2. Records usage on the trace, if one is attached.
3. Raises ModelError with a kind your code can act on. It does NOT retry: retries are a policy
   decision you will make deliberately (lecture 1.2, then Section 15). The official SDKs retry
   twice by default, which matters when you start counting cost.
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Iterator

from ..cost import Price, cost_usd, load_price
from ..settings import ROOT, Settings
from .errors import ModelError
from .providers import anthropic, ollama, openai_responses
from .transport import LiveTransport, RecordingTransport, ReplayTransport, Transport
from .types import ChatRequest, ModelResponse, Usage

PROVIDERS: dict[str, ModuleType] = {"anthropic": anthropic, "openai": openai_responses, "ollama": ollama}


class SpendGuard:
    """Stop a live run before it can spend more than its cap.

    The check is pessimistic: it assumes the call uses all of max_tokens, and estimates input
    tokens at four characters per token. That estimate is rough on purpose; the authoritative
    count arrives afterwards in the response's usage fields, and that is what gets charged.
    """

    def __init__(self, cap_usd: float, price: Price, enabled: bool):
        self.cap, self.price, self.enabled, self.spent = cap_usd, price, enabled, 0.0

    def check(self, req: ChatRequest) -> None:
        if not self.enabled:
            return
        est_in = len(ChatRequest.model_dump_json(req)) // 4
        worst = cost_usd(Usage(input_tokens=est_in, output_tokens=req.max_tokens), self.price)
        if self.spent + worst > self.cap:
            raise ModelError("budget_exceeded",
                             f"this call could cost up to {worst:.4f} USD; {self.spent:.4f} of {self.cap:g} "
                             "USD already spent this run", provider="skyops")

    def charge(self, usage: Usage) -> None:
        self.spent += cost_usd(usage, self.price)


class ModelClient:
    def __init__(self, provider: str, model: str, transport: Transport, price: Price,
                 cap_usd: float = 0.50, live: bool = False):
        self.provider_name, self.model, self.transport = provider, model, transport
        self.adapter = PROVIDERS[provider]
        self.guard = SpendGuard(cap_usd, price, enabled=live)
        self.trace = None   # attach a skyops.trace.Trace to record calls
        self.calls = 0

    def complete(self, req: ChatRequest) -> ModelResponse:
        req = req.model_copy(update={"model": req.model or self.model})
        self.guard.check(req)
        http = self.adapter.build_request(req)
        resp = self.transport.send(http)
        self.calls += 1
        if resp.status != 200:
            raise self.adapter.parse_error(resp)
        out = self.adapter.parse_response(resp.body, resp.headers)
        self.guard.charge(out.usage)
        if self.trace is not None:
            self.trace.model_call(out)
        return out

    def stream(self, req: ChatRequest) -> Iterator[str]:
        """Yield text as it arrives. The final ModelResponse is stored in self.last_response."""
        if not hasattr(self.adapter, "iter_stream"):
            raise NotImplementedError(f"streaming is implemented for anthropic only in this lecture")
        req = req.model_copy(update={"model": req.model or self.model})
        self.guard.check(req)
        resp = self.transport.send(self.adapter.build_request(req, stream=True))
        self.calls += 1
        if resp.status != 200:
            raise self.adapter.parse_error(resp)
        for item in self.adapter.iter_stream(resp.events or []):
            if isinstance(item, str):
                yield item
            else:
                self.last_response = item
                self.guard.charge(item.usage)
                if self.trace is not None:
                    self.trace.model_call(item)


def get_client(settings: Settings | None = None, *, fixture: str | Path | None = None,
               provider: str | None = None, match: str | None = None) -> ModelClient:
    """Build a client from settings. In replay mode a fixture file is required."""
    s = settings or Settings.from_env(provider=provider)
    price = load_price()
    if s.mode == "replay":
        path = Path(fixture or s.fixture or "")
        if not path.is_absolute():
            path = ROOT / path
        transport: Transport = ReplayTransport(path, match=match or s.replay_match)
    elif s.mode == "record":
        transport = RecordingTransport(LiveTransport(), fixture or s.fixture, model=s.model,
                                       api_version=getattr(PROVIDERS[s.provider], "API_VERSION", ""))
    elif s.mode == "live":
        transport = LiveTransport()
    else:   # unreachable: Settings validates the mode; never fall through to live calls
        raise ValueError(f"unknown mode {s.mode!r}")
    return ModelClient(s.provider, s.model, transport, price, cap_usd=s.max_usd_per_run, live=s.mode != "replay")
