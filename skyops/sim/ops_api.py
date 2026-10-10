"""A simulated Flight Operations REST API, served through httpx.MockTransport (no network).

Synthetic data only. Faults are scripted per flight so tests are deterministic:
  FaultPlan({"ZZ214": [503, 200]})   -> first call 503, second call 200
  FaultPlan({"ZZ173": [429]}, retry_after="1")
  FaultPlan({"ZZ999": ["timeout"]})  -> raises httpx.ReadTimeout
  FaultPlan({"ZZ142": ["malformed"]}) -> 200 with a body that breaks the contract
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import httpx

from . import flights as sim

BASE_URL = "https://ops.skyops.example/api/v1"


@dataclass
class FaultPlan:
    plan: dict[str, list] = field(default_factory=dict)
    retry_after: str | None = "1"
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def next(self, flight: str):
        seq = self.plan.get(flight, [])
        i = self.calls[flight]
        self.calls[flight] += 1
        return seq[i] if i < len(seq) else 200


def ops_transport(faults: FaultPlan | None = None) -> httpx.MockTransport:
    faults = faults or FaultPlan()

    def handler(request: httpx.Request) -> httpx.Response:
        parts = request.url.path.rstrip("/").split("/")      # /api/v1/flights/{flight}/{date}
        if len(parts) < 2 or parts[-3] != "flights":
            return httpx.Response(404, json={"error": "no such endpoint"})
        flight, date = parts[-2].upper(), parts[-1]
        outcome = faults.next(flight)
        if outcome == "timeout":
            raise httpx.ReadTimeout("simulated read timeout", request=request)
        if outcome == "malformed":
            return httpx.Response(200, json={"flight": flight, "state": "??", "etd": "late-ish"})
        if outcome == 429:
            return httpx.Response(429, headers={"Retry-After": faults.retry_after or ""},
                                  json={"error": "rate limited"})
        if outcome in (400, 500, 503):
            return httpx.Response(outcome, json={"error": f"simulated {outcome}"})
        f = sim.find(flight, date)
        if f is None:
            return httpx.Response(404, json={"error": f"no flight {flight} on {date}"})
        return httpx.Response(200, content=json.dumps(sim.as_dict(f)).encode(),
                              headers={"content-type": "application/json"})
    return httpx.MockTransport(handler)
