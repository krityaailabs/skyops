"""Lecture 5.9: the contracted tools inside the Section 4 loop, on a cancellation.

The question asks the agent to tell ZZ247's passengers and give one booking a hotel voucher. In the
recording, the model sends the notification twice (it "wants to be sure"), and asks for a voucher above
the limit. Watch what the contracts do with each. Run:
    python -m labs.s05.disruption            real run against the stand-in gateways
    python -m labs.s05.disruption --dry-run  the same conversation, nothing sent or issued

Both recordings are scripted (synthetic): a duplicate send and an over-limit request are behaviours real
agents show, but not on demand.
"""
from __future__ import annotations

import argparse

from skyops.agent.loop import RunResult, run_agent
from skyops.model.client import get_client
from skyops.tools.catalog import Gateways, toolbox

VOUCHER_EUR = 180
QUESTION = ("ZZ247 tonight is cancelled. Tell its passengers, and give booking Q7K2LM a hotel voucher "
            f"for {VOUCHER_EUR} euros.")


SCOPE = "disruption-ZZ247-2026-10-09"


def run(dry_run: bool = False) -> tuple[RunResult, Gateways]:
    """Real run and dry run each replay their own recording, strictly: the dry-run conversation is different,
    because the model sees previews instead of results."""
    box, gw, _ = toolbox(scope=SCOPE, dry_run=dry_run)
    fixture = "fixtures/s05/disruption_dry.jsonl" if dry_run else "fixtures/s05/disruption.jsonl"
    return run_agent(QUESTION, get_client(fixture=fixture, match="strict"), box), gw


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    r, gw = run(a.dry_run)
    print(r.trace.render())
    print(f"messages sent: {len(gw.sent)}   vouchers issued: {len(gw.vouchers)}   stop: {r.stop}")


if __name__ == "__main__":
    main()
