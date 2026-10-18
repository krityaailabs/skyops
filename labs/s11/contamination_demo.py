"""Lecture 11.4: find the contaminated case, and see what n-gram overlap can't see.

    python -m labs.s11.contamination_demo
A teammate proposes a fine-tuning set for the handler (labs/s11/data/sft_proposed.jsonl), built from week one's
production traces. One record didn't come from production. The scan is real code; the records are synthetic.
"""
from __future__ import annotations

import json
from collections import Counter

from skyops.evals.contamination import (N, THRESHOLD, Item, eval_items, find_contamination, heldout_vs_golden,
                                        jsonl_texts, model_inputs, ngrams, shared_flights, tokens)
from skyops.settings import ROOT

PROPOSED = ROOT / "labs/s11/data/sft_proposed.jsonl"
COMMITTED = ROOT / "data/finetune/disruption_sft.jsonl"
META = ROOT / "data/evals/heldout.meta.json"
# the same leak, reworded: the kind of copy n-gram checks miss (Yang et al. 2023)
PARAPHRASE = "Singapore ZZ214: new time 21:30, previously 19:40, inbound aircraft late."


def best_record(item: Item, path, n: int = N) -> tuple[int, float]:
    """The 1-based record that holds most of the item's n-grams, and that share."""
    mine = ngrams(tokens(item.text), n) or {tuple(tokens(item.text))}
    best = (0, 0.0)
    for i, text in enumerate(jsonl_texts(path), 1):
        share = len(mine & ngrams(tokens(text), n)) / len(mine)
        best = max(best, (i, share), key=lambda x: x[1])
    return best


def figures() -> dict:
    items = eval_items()
    by_id = {it.id: it for it in items}
    rel = str(PROPOSED.relative_to(ROOT))
    hits = find_contamination(items, {rel: jsonl_texts(PROPOSED)})
    leak = hits[0] if hits else None
    rec = best_record(by_id[leak.item], PROPOSED) if leak else (0, 0.0)
    clean = find_contamination(items, {str(COMMITTED.relative_to(ROOT)): jsonl_texts(COMMITTED)})
    prompt = find_contamination(items, model_inputs("v2"))
    short = find_contamination(items, {str(COMMITTED.relative_to(ROOT)): jsonl_texts(COMMITTED)}, n=6)
    short_rec = best_record(by_id[short[0].item], COMMITTED, n=6) if short else (0, 0.0)
    short_text = json.loads(COMMITTED.read_text().splitlines()[short_rec[0] - 1])["prompt"] if short else ""
    para = find_contamination([Item("e05_reworded", "paraphrase", PARAPHRASE)],
                              {"golden e05": [by_id["e05_time_mismatch"].text]}, threshold=0.0)
    return {"items": items, "kinds": Counter(it.kind for it in items), "proposed_records": len(jsonl_texts(PROPOSED)),
            "hits": hits, "record": rec[0], "leak_text": json.loads(PROPOSED.read_text().splitlines()[rec[0] - 1])
            ["prompt"] if leak else "", "committed_records": len(jsonl_texts(COMMITTED)), "committed_hits": clean,
            "prompt_hits": prompt, "heldout_hits": heldout_vs_golden(), "shared": shared_flights(),
            "paraphrase_share": para[0].share if para else 0.0, "by_id": by_id,
            "short_hits": short, "short_record": short_rec[0], "short_text": short_text,
            "inputs": {k: len(v) for k, v in model_inputs("v2").items()},
            "meta": json.loads(META.read_text())}


def main() -> None:
    f = figures()
    k = f["kinds"]
    print(f"{len(f['items'])} eval items ({k['golden notice']} golden, {k['held-out notice']} held-out, "
          f"{k['judge label']} judge labels); {N}-grams; leak at {THRESHOLD:.0%} or more")
    print(f"proposed fine-tune set, {f['proposed_records']} records: {len(f['hits'])} leak(s)")
    for h in f["hits"]:
        print(f"  LEAK {h.item} ({h.kind}): {h.share:.0%} of its {N}-grams, record {f['record']}")
        print(f"    eval:  {f['by_id'][h.item].text}")
        print(f"    train: {f['leak_text']}")
    print(f"committed fine-tune set (that record removed), {f['committed_records']} records: "
          f"{len(f['committed_hits'])} leaks")
    for h in f["short_hits"]:
        print(f"  at 6-grams instead: a false alarm on {h.item}, {h.share:.0%}, record {f['short_record']}: "
              f"{f['short_text']!r}, another flight")
    ins = ", ".join(f"{k} ({v})" for k, v in f["inputs"].items())
    print(f"deployed handler v2's inputs: {ins}: {len(f['prompt_hits'])} leaks")
    print(f"held-out vs golden: {len(f['heldout_hits'])} word leaks, {len(f['shared'])} shared flights")
    print(f"the same leak reworded ({PARAPHRASE!r}): {f['paraphrase_share']:.0%} of its {N}-grams, not flagged")
    m = f["meta"]
    print(f"held-out suite: built {m['built']} from the review queue; refresh {m['refresh']}, next {m['next_refresh']}")


if __name__ == "__main__":
    main()
