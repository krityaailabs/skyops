"""Operating procedures as documents with governance metadata (Section 9).

Each file in data/procedures/<tenant>/ is markdown with a front matter block:

    doc_id, title, version, owner, tenant, roles (comma list), classification,
    effective (YYYY-MM-DD), withdrawn (YYYY-MM-DD or empty), supersedes (a doc_id or empty)

The metadata travels with every chunk, so access (tenant, roles) and freshness (effective, withdrawn) can be
enforced at query time, by code, for every result.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = ROOT / "data/procedures"
TODAY = dt.date(2026, 10, 9)          # the simulators' operating day (skyops.sim.flights.DATE)


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    version: str
    owner: str
    tenant: str
    roles: tuple[str, ...]
    classification: str
    effective: dt.date
    withdrawn: dt.date | None = None
    supersedes: str | None = None

    @property
    def key(self) -> str:
        """One version of one document: SOP-GND-014@3.2."""
        return f"{self.doc_id}@{self.version}"

    def current_on(self, day: dt.date) -> bool:
        return self.effective <= day and (self.withdrawn is None or day < self.withdrawn)


@dataclass(frozen=True)
class Document:
    meta: DocMeta
    body: str                      # the markdown after the front matter
    path: str = ""

    def with_meta(self, **changes) -> "Document":
        return replace(self, meta=replace(self.meta, **changes))


def _date(v: str) -> dt.date | None:
    return dt.date.fromisoformat(v) if v else None


def parse(text: str, path: str = "") -> Document:
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: no front matter")
    head, body = text[4:].split("\n---\n", 1)
    f = {}
    for line in head.splitlines():
        k, _, v = line.partition(":")
        f[k.strip()] = v.strip()
    meta = DocMeta(doc_id=f["doc_id"], title=f["title"], version=f["version"], owner=f["owner"], tenant=f["tenant"],
                   roles=tuple(r.strip() for r in f["roles"].split(",") if r.strip()),
                   classification=f.get("classification", "internal"), effective=_date(f["effective"]),
                   withdrawn=_date(f.get("withdrawn", "")), supersedes=f.get("supersedes") or None)
    return Document(meta, body.lstrip("\n"), path)


def load_corpus(root: Path = CORPUS) -> list[Document]:
    """Every version of every procedure, in a stable order (tenant, doc_id, version)."""
    docs = [parse(p.read_text(encoding="utf-8"), str(p.relative_to(ROOT))) for p in sorted(root.glob("*/*.md"))]
    return sorted(docs, key=lambda d: (d.meta.tenant, d.meta.doc_id, [int(x) for x in d.meta.version.split(".")]))
