"""Lecture 2.3 (demo): HTTP clients, timeouts and error mapping before the reliability section.

Map every HTTP outcome to something the caller can act on, set explicit timeouts, and retry only
what is worth retrying. Section 15 replaces the naive retry here with budgets, jitter and breakers.
Run:  python -m labs.s02.http_client
"""
from __future__ import annotations

import time
from typing import Callable

import httpx

from skyops.sim.ops_api import BASE_URL, FaultPlan, ops_transport


class FlightNotFound(Exception): ...
class OpsUnavailable(Exception): ...
class OpsRejected(Exception): ...


def get_status(client: httpx.Client, flight: str, date: str, attempts: int = 3,
               sleep: Callable[[float], None] = time.sleep) -> dict:
    for attempt in range(1, attempts + 1):
        try:
            r = client.get(f"{BASE_URL}/flights/{flight}/{date}")
        except httpx.TimeoutException:
            if attempt == attempts:
                raise OpsUnavailable(f"timed out {attempts} times")
            sleep(2 ** (attempt - 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            raise FlightNotFound(flight)                    # an answer, not a failure: don't retry
        if r.status_code in (429, 503) and attempt < attempts:
            sleep(float(r.headers.get("Retry-After") or 2 ** (attempt - 1)))
            continue
        if r.status_code in (429, 500, 503):
            raise OpsUnavailable(f"HTTP {r.status_code} after {attempt} attempts")
        raise OpsRejected(f"HTTP {r.status_code}: {r.text[:80]}")   # 400 and friends: never retry
    raise AssertionError("unreachable")


def main() -> None:
    waits: list[float] = []
    plan = FaultPlan({"ZZ214": [503, 429, 200], "ZZ999": [], "ZZ261": ["timeout", "timeout", "timeout"]})
    timeout = httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=1.0)
    with httpx.Client(transport=ops_transport(plan), timeout=timeout) as c:
        print("ZZ214 ->", get_status(c, "ZZ214", "2026-10-09", sleep=waits.append)["status"], "after waits", waits)
        for flight in ("ZZ999", "ZZ261"):
            try:
                get_status(c, flight, "2026-10-09", sleep=lambda s: None)
            except Exception as e:
                print(f"{flight} -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
