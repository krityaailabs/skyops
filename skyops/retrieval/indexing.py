"""Lecture 9.5: keeping the index fresh, and revoking documents within a stated freshness window.

    build             chunk, embed and index a set of documents into per-tenant stores (9.2)
    DocumentSystem    the source of truth (a document management system) with an append-only change log:
                      every publish and every withdrawal is an event with a sequence number and a time
    NaiveSync         the common first version: list the CURRENT documents and upsert them. A withdrawn
                      document drops out of the listing, so the sync never sees it again and never deletes
                      it: it stays in the index, with the metadata it had, forever (a "zombie" chunk)
    CdcIndexer        change data capture: read the change log from the last sequence it applied, apply each
                      event once, in order. A withdrawal deletes the document's chunks. Its freshness lag is
                      the age of the oldest event it hasn't applied; the freshness window is the promise
                      (here 300 s): beyond it the index is stale, and policy search fails closed

Revocation has three lines of defence, each covering a different failure:
    the indexer       deletes withdrawn chunks within the window (the only fix for a pipeline that never
                      learns about a withdrawal, which is the naive sync's failure)
    the query filter  every query filters on the withdrawn date the index can see (9.3's `matches`): it covers
                      withdrawals the index already knows about (a scheduled withdrawal, an as-of query), NOT an
                      event the indexer never applied, because those chunks still carry no withdrawn date
    fail closed       an event that isn't applied shows up as freshness lag; past the window the index is
                      stale and policy search refuses (index_stale) instead of answering
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .chunking import chunk_corpus
from .docs import Document, load_corpus
from .embed import Embedder, HashingEmbedder
from .store import TenantIndexes

POLL_S = 60                 # the indexer reads the change log once a minute
FRESHNESS_WINDOW_S = 300    # the promise: a published change is searchable (or gone) within five minutes


def build(docs: list[Document] | None = None, strategy: str = "structure", embedder: Embedder | None = None,
          **chunk_kw) -> TenantIndexes:
    """Index every version of every document. Withdrawn and not-yet-effective versions are indexed too, with
    their dates, so the query-time filter is tested against the worst case: the revoked text IS in the index."""
    idx = TenantIndexes(embedder or HashingEmbedder())
    idx.add(chunk_corpus(docs if docs is not None else load_corpus(), strategy, **chunk_kw))
    return idx


@dataclass(frozen=True)
class ChangeEvent:
    seq: int
    at: dt.datetime
    op: str                       # "upsert" | "withdraw"
    doc: Document


@dataclass
class DocumentSystem:
    docs: dict[str, Document] = field(default_factory=dict)       # key "<doc_id>@<version>" -> latest state
    log: list[ChangeEvent] = field(default_factory=list)

    def _event(self, op: str, doc: Document, at: dt.datetime) -> ChangeEvent:
        e = ChangeEvent(len(self.log) + 1, at, op, doc)
        self.log.append(e)
        self.docs[doc.meta.key] = doc
        return e

    def publish(self, doc: Document, at: dt.datetime) -> ChangeEvent:
        return self._event("upsert", doc, at)

    def withdraw(self, key: str, on: dt.date, at: dt.datetime) -> ChangeEvent:
        return self._event("withdraw", self.docs[key].with_meta(withdrawn=on), at)

    def state_at(self, at: dt.datetime) -> dict[str, Document]:
        """Every document as the system held it at `at`, rebuilt from the change log."""
        state: dict[str, Document] = {}
        for e in self.log:
            if e.at <= at:
                state[e.doc.meta.key] = e.doc
        return state

    def current(self, day: dt.date, at: dt.datetime | None = None) -> list[Document]:
        """The listing a crawler sees at `at`: documents current on `day`. Withdrawn ones are simply not listed."""
        docs = self.state_at(at) if at is not None else self.docs
        return [d for d in docs.values() if d.meta.current_on(day)]

    def since(self, seq: int) -> list[ChangeEvent]:
        return [e for e in self.log if e.seq > seq]


@dataclass
class NaiveSync:
    """Upsert whatever the listing shows. Never deletes. Shown so you can recognise it, not used."""
    indexes: TenantIndexes

    def run(self, dms: DocumentSystem, at: dt.datetime) -> int:
        added = 0
        for d in dms.current(at.date(), at):
            store = self.indexes.store(d.meta.tenant)
            if d.meta.key not in store.doc_keys():
                added += store.add(chunk_corpus([d]))
        return added


@dataclass
class CdcIndexer:
    indexes: TenantIndexes
    window_s: int = FRESHNESS_WINDOW_S
    last_seq: int = 0
    applied: list[dict] = field(default_factory=list)      # seq, op, key, lag_s: what happened, for metrics
    fail_on: set[int] = field(default_factory=set)         # tests: sequence numbers whose apply raises
    held: dict[int, ChangeEvent] = field(default_factory=dict)  # pushed events waiting for an earlier one

    def apply(self, e: ChangeEvent) -> None:
        if e.seq in self.fail_on:
            raise RuntimeError(f"cannot apply event {e.seq}")
        store = self.indexes.store(e.doc.meta.tenant)
        store.remove_where(lambda c: c.meta.key == e.doc.meta.key)       # replace, never append beside
        if e.op == "upsert":
            store.add(chunk_corpus([e.doc]))

    def poll(self, dms: DocumentSystem, now: dt.datetime) -> int:
        """Apply events after last_seq, in order, up to `now`. An event that fails stops the poll: later events
        wait behind it (order matters: a withdrawal must never be overtaken by an older publish)."""
        n = 0
        self.held = {}                       # the log is authoritative: anything held is re-read from it
        for e in dms.since(self.last_seq):
            if e.at > now:
                break
            try:
                self.apply(e)
            except Exception:  # noqa: BLE001 - stop here; the lag metric and the alert take over
                return n
            self.last_seq = e.seq
            self.applied.append({"seq": e.seq, "op": e.op, "key": e.doc.meta.key,
                                 "lag_s": int((now - e.at).total_seconds())})
            n += 1
        return n

    def deliver(self, e: ChangeEvent, now: dt.datetime) -> bool:
        """Push delivery (a stream or webhook): at least once, and possibly out of order.

        - seq <= last_seq: a duplicate of an event already applied; skipped. (A redelivered event that is still held,
          for example after its apply failed, is tried again.)
        - seq > last_seq + 1: an earlier event hasn't arrived; this one is HELD, not applied, and last_seq does
          not move. A gap is never skipped, because the missing event may be a withdrawal. Until it arrives
          (or a poll of the change log fills it), the freshness lag counts it and the index goes stale.
        - seq == last_seq + 1: applied, then any held events that are now next in sequence.
        Returns True if this delivery applied at least one event. If an apply fails, the event stays held and
        the error propagates, so the lag alert takes over."""
        if e.seq <= self.last_seq:
            return False
        self.held[e.seq] = e                 # a redelivery of a held (or failed) event retries it
        applied = False
        while self.last_seq + 1 in self.held:
            nxt = self.held[self.last_seq + 1]
            self.apply(nxt)
            del self.held[nxt.seq]
            self.last_seq = nxt.seq
            self.applied.append({"seq": nxt.seq, "op": nxt.op, "key": nxt.doc.meta.key,
                                 "lag_s": int((now - nxt.at).total_seconds())})
            applied = True
        return applied

    def lag_s(self, dms: DocumentSystem, now: dt.datetime) -> int:
        """Freshness lag: age of the oldest event that has happened but isn't applied yet (0 when caught up)."""
        pending = [e for e in dms.since(self.last_seq) if e.at <= now]
        return int((now - pending[0].at).total_seconds()) if pending else 0

    def fresh(self, dms: DocumentSystem, now: dt.datetime) -> bool:
        return self.lag_s(dms, now) <= self.window_s


def reconcile(dms: DocumentSystem, indexes: TenantIndexes, at: dt.datetime) -> set[str]:
    """Drift check: document versions in the index that the source of truth no longer lists as current or
    future (withdrawn, or deleted). Run it daily; anything it returns is a zombie, and an alert."""
    state = dms.state_at(at)
    live = {k for k, d in state.items() if d.meta.withdrawn is None or d.meta.withdrawn > at.date()}
    indexed = {key for s in indexes.stores.values() for key in s.doc_keys()}
    return indexed - live


# ---------- the lecture's timeline (synthetic) ----------

LOADED_AT = dt.datetime(2026, 9, 30, 23, 0, 0)
WITHDRAWN_AT = dt.datetime(2026, 10, 1, 6, 0, 15)       # SOP-GND-009 withdrawn when SOP-GND-014 v3.2 starts
V33_AT = dt.datetime(2026, 10, 9, 9, 0, 0)              # SOP-GND-014 v3.3 published, effective 2026-11-01


def history() -> DocumentSystem:
    """The document system as it was on 30 September, then its change log: the withdrawal of SOP-GND-009
    and the early publication of SOP-GND-014 v3.3. The files in data/procedures are the final state."""
    dms = DocumentSystem()
    final = load_corpus()
    for d in final:
        if d.meta.key == "SOP-GND-014@3.3":
            continue
        dms.publish(d.with_meta(withdrawn=None) if d.meta.key == "SOP-GND-009@2.0" else d, LOADED_AT)
    gnd009 = next(d for d in final if d.meta.key == "SOP-GND-009@2.0")
    dms.withdraw("SOP-GND-009@2.0", gnd009.meta.withdrawn, WITHDRAWN_AT)
    dms.publish(next(d for d in final if d.meta.key == "SOP-GND-014@3.3"), V33_AT)
    return dms


def poll_times(start: dt.datetime, end: dt.datetime, every_s: int = POLL_S) -> list[dt.datetime]:
    """Polls on the minute boundary, every `every_s` seconds, from start to end inclusive."""
    t = start.replace(second=0, microsecond=0)
    out = []
    while t <= end:
        if t >= start:
            out.append(t)
        t += dt.timedelta(seconds=every_s)
    return out
