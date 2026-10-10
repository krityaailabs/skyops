"""Lab 2.7, exercise: a typed, tested, async client for the Flight Operations API.

Requirements (visible checks in fitness/s02, plus hidden checks):
1. async get_status(flight, date) -> FlightStatus (the contract from lecture 2.1);
2. a timeout on every request; a timeout counts as a retryable failure;
3. retry 429 and 503 at most `attempts` times in total, honouring Retry-After when present,
   otherwise backing off 1, 2, 4 seconds (use the injected sleep, never real time in tests);
4. 404 raises FlightNotFound immediately; any other 4xx raises OpsRejected immediately; any other 5xx
   raises OpsUnavailable immediately (only 429 and 503 are worth waiting for here);
5. a 200 whose body breaks the contract raises OpsDataError: never return unvalidated data.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import httpx
from pydantic import ValidationError

from skyops.sim.ops_api import BASE_URL

from .contracts import FlightStatus


class FlightNotFound(Exception): ...
class OpsUnavailable(Exception): ...
class OpsRejected(Exception): ...
class OpsDataError(Exception): ...


class AsyncFlightOpsClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout_s: float = 2.0,
                 attempts: int = 3, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.http = httpx.AsyncClient(transport=transport, timeout=timeout_s)
        self.attempts, self.sleep = attempts, sleep

    async def get_status(self, flight: str, date: str) -> FlightStatus:
        for attempt in range(1, self.attempts + 1):
            last = attempt == self.attempts
            try:
                r = await self.http.get(f"{BASE_URL}/flights/{flight}/{date}")
            except httpx.TimeoutException:
                if last:
                    raise OpsUnavailable(f"{flight}: timed out {attempt} times")
                await self.sleep(2 ** (attempt - 1))
                continue
            if r.status_code == 200:
                try:
                    return FlightStatus.model_validate(r.json())
                except (ValidationError, ValueError) as e:
                    raise OpsDataError(f"{flight}: response breaks the contract") from e
            if r.status_code == 404:
                raise FlightNotFound(flight)
            if r.status_code in (429, 503):
                if last:
                    raise OpsUnavailable(f"{flight}: HTTP {r.status_code} after {attempt} attempts")
                ra = r.headers.get("Retry-After")
                await self.sleep(float(ra) if ra else 2 ** (attempt - 1))
                continue
            if r.status_code >= 500:
                raise OpsUnavailable(f"{flight}: HTTP {r.status_code}")
            raise OpsRejected(f"{flight}: HTTP {r.status_code}")
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self.http.aclose()
