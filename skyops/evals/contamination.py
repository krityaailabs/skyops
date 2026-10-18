"""Lectures 11.4 and 11.5: contamination. Does any eval item appear in what the model is shown or trained on?

    python -m skyops.evals.contamination                    # the deployed handler's prompt, examples, fine-tune data
    python -m skyops.evals.contamination --handler v4       # a candidate's
    python -m skyops.evals.contamination --extra FILE.jsonl # also scan a proposed file (e.g. a fine-tune set)

An eval item that leaks into a prompt, a few-shot example or a fine-tuning set stops measuring the handler: the
model can answer that item from memory or by imitation, so its score rises while behaviour on real traffic
doesn't. The check here is the classic one, word n-gram overlap, after normalising case and punctuation:
GPT-3's decontamination used 13-gram overlap, and the GPT-4 report used 50-character substrings. Our items are
short ops notices, so n = 8 words, and an item shorter than n counts as leaked if it appears whole.

What n-gram overlap can't see: paraphrases and translations of an item (Yang et al. 2023 show such rephrased
samples pass n-gram decontamination). So it is a floor, not a proof. The other half of the defence is process:
a held-out suite nobody writes prompts from, refreshed on a cadence (data/evals/heldout.meta.json).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from ..settings import ROOT

N = 8
THRESHOLD = 0.5
FINETUNE = ROOT / "data/finetune"
CORPUS = ROOT / "data/procedures"            # Section 9's retrieval corpus: the agent can be shown any of it


def tokens(text: str) -> list[str]:
    """Lower-case words and numbers; punctuation and spacing don't count ("ETD 21:30" -> ["etd", "21", "30"])."""
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(toks: list[str], n: int = N) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


@dataclass(frozen=True)
class Hit:
    item: str           # the eval item's id
    kind: str           # "golden notice", "held-out notice", "judge label"
    source: str         # where it was found, e.g. "prompt v4" or "data/finetune/disruption_sft.jsonl"
    share: float        # share of the item's n-grams found in the source (1.0 for a short item found whole)


@dataclass(frozen=True)
class Item:
    id: str
    kind: str
    text: str


def _contiguous(needle: list[str], hay: list[str]) -> bool:
    k = len(needle)
    return any(hay[i:i + k] == needle for i in range(len(hay) - k + 1))


def find_contamination(items: list[Item], sources: dict[str, list[str]], n: int = N,
                       threshold: float = THRESHOLD) -> list[Hit]:
    """Lecture 11.5: every eval item found in a source. Return Hits in item order, then source order.

    `sources` maps a source name to its texts (a prompt is one text; a JSONL file is one text per line). Tokenise
    with tokens(). Never build an n-gram across two texts: a source's n-grams are the union of each text's own.
    - An item with at least n tokens: share = |its n-grams that appear in the source| / |its n-grams|.
    - An item with fewer than n tokens (but at least one): share = 1.0 if its whole token sequence appears
      contiguously inside one of the source's texts, else 0.0.
    - An item with no tokens is skipped.
    A source leaks the item when share >= threshold: Hit(item.id, item.kind, source name, round(share, 4)).
    """
    grams = {name: set().union(*(ngrams(tokens(t), n) for t in texts)) if texts else set()
             for name, texts in sources.items()}
    toks_by_source = {name: [tokens(t) for t in texts] for name, texts in sources.items()}
    hits = []
    for it in items:
        toks = tokens(it.text)
        if not toks:
            continue
        mine = ngrams(toks, n)
        for name in sources:
            if len(toks) >= n:
                share = len(mine & grams[name]) / len(mine)
            else:
                share = 1.0 if any(_contiguous(toks, t) for t in toks_by_source[name]) else 0.0
            if share >= threshold:
                hits.append(Hit(it.id, it.kind, name, round(share, 4)))
    return hits


# ---------- what to scan ----------

def eval_items() -> list[Item]:
    """Every eval item a leak would spoil: golden and held-out notices, and the judge's labelled messages."""
    from .dataset import GOLDEN, HELDOUT, load_golden
    from .judge import load_labels
    items = [Item(c.id, "golden notice", c.notice) for c in load_golden(GOLDEN)]
    items += [Item(c.id, "held-out notice", c.notice) for c in load_golden(HELDOUT)]
    items += [Item(l.id, "judge label", l.message) for l in load_labels()]
    return items


def jsonl_texts(path: Path) -> list[str]:
    """One text per JSONL line: every string value in the record, joined, so a leak in any field counts."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out.append(" \n ".join(v for v in rec.values() if isinstance(v, str)))
    return out


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def model_inputs(version: str, extra: list[Path] = ()) -> dict[str, list[str]]:
    """What the model is shown or trained on for one handler version: its system prompt exactly as sent (so any
    examples built into it are scanned too), the retrieval corpus (every procedure document, one text each), every
    fine-tuning file in data/finetune/, and any extra file."""
    from ..agent.disruption import SYSTEM
    src = {f"prompt {version}": [SYSTEM[version]]}
    docs = sorted(CORPUS.glob("**/*.md")) if CORPUS.exists() else []
    if docs:
        src[rel(CORPUS) + " (retrieval corpus)"] = [d.read_text(encoding="utf-8") for d in docs]
    for f in sorted(FINETUNE.glob("*.jsonl")) if FINETUNE.exists() else []:
        src[rel(f)] = jsonl_texts(f)
    for f in extra:
        src[rel(Path(f))] = jsonl_texts(Path(f))
    return src


def heldout_vs_golden() -> list[Hit]:
    """A held-out case that copies a golden one isn't held out: check the two suites against each other too."""
    from .dataset import GOLDEN, HELDOUT, load_golden
    items = [Item(c.id, "held-out notice", c.notice) for c in load_golden(HELDOUT)]
    return find_contamination(items, {rel(GOLDEN): [c.notice for c in load_golden(GOLDEN)]})


def case_flights(case) -> set[str]:
    """Every ZZ or XY flight a case is about: its own flight, the one it expects notified, and any in the notice."""
    found = {case.flight.replace(" ", "").upper()}
    if case.expect.notify is not None:
        found.add(case.expect.notify.flight)
    found |= {m.replace(" ", "").upper() for m in re.findall(r"\b(?:ZZ|XY) ?\d{1,4}\b", case.notice, re.I)}
    return found


def shared_flights() -> list[Hit]:
    """Copying isn't only word for word: the same flight and facts in a new format is the same case, and no n-gram
    check sees it. So a held-out case may share no flight with the golden set. One Hit per held-out case that does
    (kind "held-out notice", source "golden set (same flight)", share 1.0)."""
    from .dataset import GOLDEN, HELDOUT, load_golden
    golden = set().union(*(case_flights(c) for c in load_golden(GOLDEN)))
    return [Hit(c.id, "held-out notice", "golden set (same flight)", 1.0)
            for c in load_golden(HELDOUT) if case_flights(c) & golden]


def main(argv: list[str] | None = None) -> int:
    from .runner import deployed_version
    ap = argparse.ArgumentParser()
    ap.add_argument("--handler", default=None, help="handler version (default: config/handler.toml)")
    ap.add_argument("--extra", action="append", default=[], help="also scan this JSONL file")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args(argv)
    version = a.handler or deployed_version()
    items, src = eval_items(), model_inputs(version, [Path(p) for p in a.extra])
    hits = find_contamination(items, src, threshold=a.threshold) + heldout_vs_golden() + shared_flights()
    print(f"scanned {len(items)} eval items against {len(src)} model inputs for handler {version}, plus held-out vs "
          f"golden, words and flights ({N}-grams, threshold {a.threshold:.0%})")
    for name, texts in src.items():
        print(f"  {name}: {len(texts)} text(s)")
    private = [h for h in hits if h.kind == "held-out notice"]
    for h in hits:
        if h.kind != "held-out notice":
            print(f"LEAK  {h.item} ({h.kind}) in {h.source}: {h.share:.0%} of its {N}-grams")
    if private:
        print(f"LEAK  heldout: {len(private)} hit(s), ids in the restricted report")
    print(f"{len(hits)} leak(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
