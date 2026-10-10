"""Lecture 9.2: chunking operating procedures by structure, and the two failures fixed-size chunking causes.

A procedure has structure that carries meaning: a heading says what the next lines are about, a numbered list
is one procedure in order, and a table row means nothing without its header row. Fixed-size chunking ignores
all of it. Two failures follow, and `find_failures` detects both:

    cut_structure    a table or a numbered list split across two chunks: the second half loses its header
                     row or its first steps ("Above 150 EUR..." with no word about hotels)
    orphan_heading   a chunk that ends with a heading whose content is in the next chunk: the heading is
                     retrieved for a question it can't answer, and the content has lost its label

`structure_chunks` (the lab) splits at section boundaries, keeps every table and list whole, and puts the
document title and section heading at the top of every chunk, so a chunk read alone still says what it's about.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .docs import DocMeta, Document
from .text import count_tokens, slug


@dataclass(frozen=True)
class Block:
    kind: str                # heading | paragraph | table | list
    text: str
    start: int               # character offsets in the document body
    end: int
    section: str             # slug of the level-2 heading this block belongs to ("" before the first one)
    heading: str = ""        # for headings: the heading text; for other blocks: their section's heading text
    level: int = 0           # heading level (1 = the title, 2 = a section)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str            # "<doc_id>@<version>#<n>"
    meta: DocMeta
    text: str                # what is embedded, indexed and returned
    sections: tuple[str, ...]  # section slugs this chunk's text comes from (labels are "<doc_id>#<section>")
    start: int
    end: int
    tokens: int = 0
    strategy: str = "structure"

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f"{self.meta.doc_id}#{s}" for s in self.sections)


_LIST = re.compile(r"^(\d+\.|[-*])\s")


def blocks(doc: Document) -> list[Block]:
    """Split a body into headings, paragraphs, tables (lines starting '|') and lists (numbered or bulleted)."""
    out: list[Block] = []
    lines = doc.body.split("\n")
    pos, i, section, heading = 0, 0, "", ""
    offsets = []
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        start = offsets[i]
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            if level == 2:
                section, heading = slug(text), text
            out.append(Block("heading", line, start, start + len(line), section, text, level))
            i += 1
            continue
        kind = "table" if line.startswith("|") else "list" if _LIST.match(line) else "paragraph"
        j = i
        while j + 1 < len(lines) and lines[j + 1].strip() and not lines[j + 1].startswith("#"):
            nxt = lines[j + 1]
            nk = "table" if nxt.startswith("|") else "list" if _LIST.match(nxt) else "paragraph"
            if nk != kind:
                break
            j += 1
        end = offsets[j] + len(lines[j])
        out.append(Block(kind, doc.body[start:end], start, end, section, heading))
        i = j + 1
    return out


def _sections_between(bl: list[Block], start: int, end: int) -> tuple[str, ...]:
    secs: list[str] = []
    for b in bl:
        if b.start < end and b.end > start and b.section and b.section not in secs:
            secs.append(b.section)
    return tuple(secs)


def naive_chunks(doc: Document, size: int = 500) -> list[Chunk]:
    """Fixed-size windows of `size` characters, no overlap: the common default of quick-start tutorials."""
    bl = blocks(doc)
    out = []
    for n, start in enumerate(range(0, len(doc.body), size)):
        end = min(start + size, len(doc.body))
        text = doc.body[start:end].strip()
        if text:
            out.append(Chunk(f"{doc.meta.key}#n{n}", doc.meta, text, _sections_between(bl, start, end), start, end,
                             count_tokens(text), "naive"))
    return out


def structure_chunks(doc: Document, max_tokens: int = 220) -> list[Chunk]:
    """Lecture 9.2: chunk a procedure along its structure.

    Use blocks(doc). Skip headings as content (the level-1 title and level-2 headings go in the prefix instead).
    Group the remaining blocks by their section (Block.section), in document order. Within a section, add
    blocks to the current chunk while the chunk's body stays within max_tokens (count_tokens of the blocks
    joined with blank lines); otherwise start a new chunk in the same section. Never split a block: a table
    or a list longer than max_tokens becomes one chunk on its own.
    Every chunk's text is the prefix "<title> > <section heading>" (the section heading is Block.heading),
    a blank line, then its blocks joined with "\\n\\n". For blocks before the first level-2 heading, the
    prefix is just the title.
    Each Chunk: chunk_id f"{doc.meta.key}#{n}" with n counting from 0 across the document, meta=doc.meta,
    sections=(block.section,) for that section, start/end = the first block's start and the last block's
    end, tokens=count_tokens(text), strategy="structure".
    """
    bl = [b for b in blocks(doc) if b.kind != "heading"]
    out: list[Chunk] = []
    cur: list[Block] = []

    def flush() -> None:
        if not cur:
            return
        prefix = doc.meta.title + (f" > {cur[0].heading}" if cur[0].heading else "")
        text = prefix + "\n\n" + "\n\n".join(b.text for b in cur)
        out.append(Chunk(f"{doc.meta.key}#{len(out)}", doc.meta, text, (cur[0].section,), cur[0].start, cur[-1].end,
                         count_tokens(text), "structure"))
        cur.clear()

    for b in bl:
        if cur and (b.section != cur[0].section or
                    count_tokens("\n\n".join(x.text for x in [*cur, b])) > max_tokens):
            flush()
        cur.append(b)
    flush()
    return out


@dataclass
class Failures:
    cut_structure: list[tuple[str, str]] = field(default_factory=list)   # (chunk_id, "table" | "list")
    orphan_heading: list[tuple[str, str]] = field(default_factory=list)  # (chunk_id, heading text)

    @property
    def total(self) -> int:
        return len(self.cut_structure) + len(self.orphan_heading)


def find_failures(doc: Document, chunks: list[Chunk]) -> Failures:
    """Detect the two chunking failures for one document's chunks (by character spans in the body)."""
    f = Failures()
    bl = blocks(doc)
    for b in bl:
        if b.kind in ("table", "list"):
            inside = [c for c in chunks if c.start < b.end and c.end > b.start]
            if len(inside) > 1:
                for c in inside[1:]:
                    f.cut_structure.append((c.chunk_id, b.kind))
    heads = [b for b in bl if b.kind == "heading" and b.level == 2]
    for h in heads:
        nxt = next((b.start for b in bl if b.kind == "heading" and b.start > h.start), len(doc.body))
        content = nxt - h.end                                   # characters of this section after its heading
        holder = next((c for c in chunks if c.start <= h.start < c.end), None)
        if holder is None or content <= 0 or holder.end >= nxt:
            continue                                            # the whole section sits with its heading
        if (holder.end - h.end) < content / 4:                  # most of the section is in later chunks
            f.orphan_heading.append((holder.chunk_id, h.heading))
    return f


def chunk_corpus(docs: list[Document], strategy: str = "structure", **kw) -> list[Chunk]:
    fn = structure_chunks if strategy == "structure" else naive_chunks
    return [c for d in docs for c in fn(d, **kw)]
