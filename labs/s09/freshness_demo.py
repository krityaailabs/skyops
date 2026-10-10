"""Lecture 9.5: the zombie policy, and the change-data-capture indexer that revokes it within the window.

    python -m labs.s09.freshness_demo
A synthetic timeline (skyops/retrieval/indexing.py): the index is loaded on 30 September; SOP-GND-009 (agents
may issue hotel vouchers up to 250 EUR) is withdrawn at 06:00:15 on 1 October, when SOP-GND-014 v3.2 starts.
"""
from __future__ import annotations

import datetime as dt

from skyops.retrieval.access import DIRECTORY
from skyops.retrieval.chunking import chunk_corpus
from skyops.retrieval.embed import HashingEmbedder
from skyops.retrieval.indexing import (FRESHNESS_WINDOW_S, LOADED_AT, POLL_S, WITHDRAWN_AT, CdcIndexer, NaiveSync,
                                       history, poll_times, reconcile)
from skyops.retrieval.search import Filters, Retriever
from skyops.retrieval.store import TenantIndexes
from skyops.retrieval.tool import SearchProceduresArgs, make_search_contract
from skyops.tools.contracts import as_tool

Q = "hotel voucher limit without approval"
OLD = "SOP-GND-009@2.0"


def rank_of_old(idx: TenantIndexes, day: dt.date) -> int | None:
    res = Retriever(idx).search(Q, DIRECTORY["agent-anna"], k=5, rerank=True, filters=Filters(as_of=day))
    return next((i for i, h in enumerate(res.hits, 1) if h.chunk.meta.key == OLD), None)


def returns_old(idx: TenantIndexes, day: dt.date) -> bool:
    return rank_of_old(idx, day) is not None


def naive() -> dict:
    dms = history()
    idx = TenantIndexes(HashingEmbedder())
    for d in dms.current(LOADED_AT.date(), LOADED_AT):
        idx.store(d.meta.tenant).add(chunk_corpus([d]))
    sync = NaiveSync(idx)
    night, added = dt.datetime(2026, 10, 1, 23, 0), 0
    while night <= dt.datetime(2026, 10, 9, 23, 0):      # the nightly sync, every night for nine nights
        added += sync.run(dms, night)
        night += dt.timedelta(days=1)
    return {"still_there": returns_old(idx, dt.date(2026, 10, 9)), "rank": rank_of_old(idx, dt.date(2026, 10, 9)),
            "nights": 9, "added": added, "zombies": sorted(reconcile(dms, idx, night - dt.timedelta(days=1)))}


def cdc(fail: bool = False) -> dict:
    dms = history()
    idx = TenantIndexes(HashingEmbedder())
    ix = CdcIndexer(idx, fail_on={next(e.seq for e in dms.log if e.op == "withdraw")} if fail else set())
    ix.poll(dms, LOADED_AT)
    gone_at, lags, stale_at = None, {}, None
    for t in poll_times(WITHDRAWN_AT - dt.timedelta(minutes=1), WITHDRAWN_AT + dt.timedelta(minutes=10), POLL_S):
        before = returns_old(idx, t.date())
        ix.poll(dms, t)
        lag = ix.lag_s(dms, t)
        lags[t.strftime("%H:%M:%S")] = lag
        if before and not returns_old(idx, t.date()) and gone_at is None:
            gone_at = t
        if not ix.fresh(dms, t) and stale_at is None:
            stale_at = t
    out = {"gone_at": gone_at, "revocation_s": int((gone_at - WITHDRAWN_AT).total_seconds()) if gone_at else None,
           "lags": lags, "stale_at": stale_at}
    if fail:
        c = make_search_contract(Retriever(idx), lambda: DIRECTORY["agent-anna"],
                                 fresh=lambda: ix.fresh(dms, stale_at))
        out["tool"] = as_tool(c, None, "read").fn(SearchProceduresArgs(question=Q))
    before = idx.size
    e = next(e for e in dms.log if e.op == "upsert" and e.doc.meta.key == "SOP-GND-014@3.2")
    ix.apply(e)
    out["replay_same_size"] = idx.size == before            # at-least-once delivery: applying twice changes nothing
    return out


def main() -> None:
    n = naive()
    print(f"naive nightly sync, {n['nights']} nights later: {OLD} still returned for agents: {n['still_there']} "
          f"(rank {n['rank']} of 5)")
    print(f"reconciliation against the document system: zombies {n['zombies']}")
    c = cdc()
    print(f"CDC indexer (poll every {POLL_S} s, window {FRESHNESS_WINDOW_S} s): withdrawn at "
          f"{WITHDRAWN_AT:%H:%M:%S}, gone from results at {c['gone_at']:%H:%M:%S}: {c['revocation_s']} s")
    print(f"same event applied twice, index unchanged: {c['replay_same_size']}")
    s = cdc(fail=True)
    shown = {k: v for k, v in s["lags"].items() if k in ("06:01:00", "06:03:00", "06:05:00", "06:06:00")}
    print(f"stuck indexer, freshness lag (s): {shown}")
    print(f"stale from {s['stale_at']:%H:%M:%S}; search_procedures then returns: {s['tool']['error']} "
          f"({s['tool']['category']})")


if __name__ == "__main__":
    main()
