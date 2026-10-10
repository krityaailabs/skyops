"""List fixtures still marked synthetic. Live recordings replace them before filming."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
total = synthetic = 0
for f in sorted(p for p in root.glob("**/fixtures/**/*.jsonl") if "dist" not in p.parts):
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    s = sum(1 for r in recs if r.get("meta", {}).get("synthetic"))
    total += len(recs); synthetic += s
    if s:
        print(f"{f.relative_to(root)}: {s} of {len(recs)} records synthetic")
print(f"{synthetic} of {total} records are synthetic")
