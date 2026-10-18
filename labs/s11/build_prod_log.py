"""Instructor only: write the SYNTHETIC production log for Section 11 (data/prod/traces.jsonl) from Incident 18's
fault scenario (skyops/sim/incident18.py), and the fine-tune sets built from it (data/finetune/disruption_sft.jsonl,
labs/s11/data/sft_proposed.jsonl).

    python labs/s11/build_prod_log.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from skyops.sim.incident18 import SEED, build   # noqa: E402

OUT = ROOT / "data/prod/traces.jsonl"
SFT = ROOT / "data/finetune/disruption_sft.jsonl"
SFT_PROPOSED = ROOT / "labs/s11/data/sft_proposed.jsonl"


def build_sft(traces: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """A fine-tune set someone built from week one's traces: notice in, the handler's summary out. The proposed
    version also carries one line copied from a golden case (e05_time_mismatch), the planted leak of 11.4."""
    from skyops.evals.online import score_trace
    pool = [t for t in traces if t["prompt"] == "v2" and score_trace(t).passed]
    pool = rng.sample(pool, 38)
    clean = [{"prompt": t["notice"], "completion": t["answer"], "from_trace": t["trace_id"]} for t in pool]
    # a different flight that happens to share a run of words and times with a golden case: 6-grams flag it, 8 don't
    near = {"prompt": "ZZ688 SIN now ETD 21:30 (was 19:40), crew rest.",
            "completion": "ZZ688 is confirmed delayed to 21:30; I notified its passengers once.", "from_trace": ""}
    clean = clean[:30] + [near] + clean[30:]
    leak = {"prompt": "ZZ214 SIN now ETD 21:30 (was 19:40), late inbound aircraft.",
            "completion": "Check ZZ214 first. The system says 20:15, not 21:30, so notify nobody and tell the duty "
                          "manager the times differ.", "from_trace": ""}
    proposed = clean[:23] + [leak] + clean[23:]
    return clean, proposed


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}: {len(rows)} records")


def main() -> None:
    rng = random.Random(SEED)
    traces = build(rng)
    write(OUT, traces)
    clean, proposed = build_sft(traces, random.Random(SEED + 1))
    write(SFT, clean)
    write(SFT_PROPOSED, proposed)


if __name__ == "__main__":
    main()
