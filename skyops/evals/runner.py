"""Run the disruption handler on golden cases, attempt by attempt, against recorded model turns.

    att = run_attempt(case, 3)                 # one attempt: the real loop, real tools, a replayed model
    atts = run_case(case, n=N_ATTEMPTS)        # n attempts of one case

Each attempt replays its own recording, strictly (fixtures/s10/attempts/<case>/a<n>.jsonl): the loop, the
contracted tools and the stand-in gateways run for real every time, so the request fingerprints must match.
About the recordings: they are SCRIPTED (meta.synthetic = true). A live model run n times varies; scripted
attempts vary only where the course wrote a different script. Re-record them with SKYOPS_MODE=record to
measure a real model: the harness, checks and statistics don't change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent.disruption import handle_notice
from ..agent.loop import RunResult
from ..model.client import get_client
from ..settings import ROOT, Settings
from .dataset import GoldenCase

N_ATTEMPTS = 5
RUNS = Path("fixtures/s10/attempts")


@dataclass
class Attempt:
    """One run of one case, with what really happened in the world (the stand-in gateways)."""
    case_id: str
    attempt: int
    run: RunResult
    sent: list[dict[str, Any]] = field(default_factory=list)       # messages the gateway really sent
    vouchers: list[dict[str, Any]] = field(default_factory=list)   # vouchers really issued

    @property
    def proposed(self) -> list[tuple[str, dict[str, Any]]]:
        """Every tool call the model proposed, in order, whether or not it ran."""
        return self.run.tool_calls

    @property
    def tokens(self) -> int:
        return self.run.trace.usage.total

    @property
    def cost_usd(self) -> float:
        return self.run.trace.cost


def fixture_path(case_id: str, attempt: int, folder: Path = RUNS) -> Path:
    return folder / case_id / f"a{attempt}.jsonl"


def run_attempt(case: GoldenCase, attempt: int, version: str = "v2", folder: Path = RUNS,
                client=None) -> Attempt:
    """Run one attempt. `client` overrides the replay (the fixture builder passes a scripted one)."""
    client = client or get_client(fixture=fixture_path(case.id, attempt, folder), match="strict")
    run, gw = handle_notice(case.notice, case.request, client, version=version, scope=f"eval-{case.id}-a{attempt}")
    return Attempt(case.id, attempt, run, list(gw.sent), list(gw.vouchers))


def turns_label(files: list[Path], mode: str | None = None) -> str:
    """How the model turns in a run were produced, for the report: read from the settings and the fixtures, never
    assumed. Live or record mode says so; replay says whether any fixture is marked meta.synthetic."""
    mode = mode or Settings.from_env().mode
    if mode != "replay":
        return mode
    synthetic = False
    for f in files:
        p = f if f.is_absolute() else ROOT / f
        if p.exists():
            first = p.read_text(encoding="utf-8").split("\n", 1)[0]
            synthetic |= bool(first.strip() and json.loads(first).get("meta", {}).get("synthetic"))
    return "replay (scripted fixtures, synthetic)" if synthetic else "replay (recorded fixtures)"


def attempt_files(case_ids: list[str], n: int = N_ATTEMPTS, folder: Path = RUNS) -> list[Path]:
    return [fixture_path(cid, i, folder) for cid in case_ids for i in range(1, n + 1)]


def run_case(case: GoldenCase, n: int = N_ATTEMPTS, version: str = "v2", folder: Path = RUNS) -> list[Attempt]:
    return [run_attempt(case, i, version, folder) for i in range(1, n + 1)]

