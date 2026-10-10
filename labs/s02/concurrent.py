"""Lab 2.2: asyncio for tool calls: concurrency, timeouts and cancellation.

Fetch the status of many flights at once, but:
- never more than max_concurrency calls in flight (protect the system of record);
- each call has its own timeout, and a slow call becomes a result, not a crash;
- if the whole batch is cancelled, every child call is cancelled too (no orphans).
Run:  python -m labs.s02.concurrent
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass(frozen=True)
class Result:
    flight: str
    ok: bool
    status: str | None = None
    error: str | None = None          # "timeout" or the exception type name


async def fetch_all(flights: list[str], fetch: Callable[[str], Awaitable[str]],
                    timeout_s: float, max_concurrency: int) -> list[Result]:
    """Return one Result per flight, in input order."""
    sem = asyncio.Semaphore(max_concurrency)

    async def one(flight: str) -> Result:
        async with sem:                                   # bounded concurrency
            try:
                async with asyncio.timeout(timeout_s):    # per-call deadline (Python 3.11+)
                    return Result(flight, True, status=await fetch(flight))
            except TimeoutError:
                return Result(flight, False, error="timeout")
            except Exception as e:                        # one failure must not sink the batch
                return Result(flight, False, error=type(e).__name__)

    async with asyncio.TaskGroup() as tg:                 # cancels children if the parent is cancelled
        tasks = [tg.create_task(one(f)) for f in flights]
    return [t.result() for t in tasks]


def main() -> None:
    from skyops.sim.async_ops import AsyncFlightOps
    flights = ["ZZ101", "ZZ118", "ZZ142", "ZZ156", "ZZ173", "ZZ188", "ZZ214", "ZZ230", "ZZ247", "ZZ261"]
    ops = AsyncFlightOps(delays={"ZZ173": 0.5})
    t0 = time.perf_counter()
    results = asyncio.run(fetch_all(flights, ops.status, timeout_s=0.2, max_concurrency=4))
    elapsed = time.perf_counter() - t0
    for r in results:
        print(f"{r.flight}: {r.status if r.ok else 'FAILED (' + str(r.error) + ')'}")
    print(f"{sum(r.ok for r in results)} ok, {sum(not r.ok for r in results)} failed, peak in flight {ops.peak}, "
          f"elapsed {elapsed:.2f} s")


if __name__ == "__main__":
    main()
