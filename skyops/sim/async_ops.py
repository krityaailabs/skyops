"""An async flight-status source with scripted latency, for concurrency labs. Delays are in real
seconds but scaled down (tens of milliseconds) so tests stay fast. It also records the peak number
of calls in flight at once, so a test can prove a concurrency limit holds."""
from __future__ import annotations

import asyncio

from . import flights as sim


class AsyncFlightOps:
    def __init__(self, delays: dict[str, float] | None = None, default_delay: float = 0.02):
        self.delays, self.default = delays or {}, default_delay
        self.in_flight = self.peak = self.started = self.finished = self.cancelled = 0

    async def status(self, flight: str) -> str:
        self.started += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delays.get(flight, self.default))
            f = sim.find(flight, sim.DATE)
            self.finished += 1
            return f.status if f else "UNKNOWN"
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_flight -= 1
