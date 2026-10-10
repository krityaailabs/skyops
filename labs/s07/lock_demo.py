"""Lecture 7.7: what the contract lock catches. Each change below is applied to a copy of today's listing.

    python -m labs.s07.lock_demo
"""
from __future__ import annotations

import copy
import json

from skyops.mcp import versioning


def main() -> None:
    listing, version = versioning.current_listing()
    lock = json.loads(versioning.LOCK.read_text())

    def show(label: str, changed: dict, ver: str) -> None:
        problems = versioning.check_lock(changed, ver, lock)
        print(f"{label:<52} at {ver}: {'; '.join(problems) if problems else 'OK'}")

    show("no change", listing, version)
    reworded = copy.deepcopy(listing)
    reworded["get_flight_status"]["description"] = "Gets status."
    show("description reworded", reworded, version)
    show("description reworded", reworded, "0.8.0")
    required = copy.deepcopy(listing)
    required["get_flight_status"]["inputSchema"]["properties"]["terminal"] = {"type": "string"}
    required["get_flight_status"]["inputSchema"]["required"].append("terminal")
    show("new required argument", required, "0.8.0")
    show("new required argument", required, "1.0.0")
    removed = {k: v for k, v in listing.items() if k != "issue_voucher"}
    show("issue_voucher removed", removed, "0.8.0")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    main()
