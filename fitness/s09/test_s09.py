"""Section 9 fitness: retrieval as a governed tool. Structure-aware chunks, hybrid search with filters and a
reranker, document-level access and tenant isolation, revocation within a freshness window, retrieval evals,
a bounded multi-step search, and search_procedures as a contract with a revoked-policy regression test.
Fail at s09-start, pass at s09. Offline and deterministic: no model calls (9.7's model turns are a replay)."""
import datetime as dt
import functools
import logging

import pytest

from skyops.retrieval.access import DIRECTORY, Principal, can_read
from skyops.retrieval.chunking import find_failures, naive_chunks, structure_chunks
from skyops.retrieval.docs import TODAY, load_corpus
from skyops.retrieval.embed import HashingEmbedder
from skyops.retrieval.evals import evaluate, reciprocal_rank, recall_at_k
from skyops.retrieval.indexing import build
from skyops.retrieval.search import Filters, Retriever, matches, rrf_fuse
from skyops.retrieval.store import ChunkStore
from skyops.retrieval.tool import SearchProceduresArgs, make_search_contract
from skyops.tools.contracts import as_tool, lint_contract

DOCS = load_corpus()
BY = {d.meta.key: d for d in DOCS}
AGENT, DM, YY_DM = DIRECTORY["agent-anna"], DIRECTORY["dm-deepa"], DIRECTORY["yy-dm-yara"]


@functools.cache
def _shared() -> Retriever:
    return Retriever(build())


@pytest.fixture
def retriever():
    """Built lazily inside each test, so an unfinished lab reads LAB NOT DONE, not a setup error."""
    return _Lazy()


class _Lazy:
    def __getattr__(self, name):
        return getattr(_shared(), name)


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _docs(res) -> set[str]:
    return {h.chunk.meta.key for h in res.hits}


# ---------- 9.2 chunking and embeddings ----------

def test_l92_structure_chunks_keep_tables_and_lists_whole():
    naive_failures = sum(find_failures(d, naive_chunks(d, 500)).total for d in DOCS)
    assert naive_failures > 0                                  # the baseline really fails
    for d in DOCS:
        chunks = structure_chunks(d)
        assert find_failures(d, chunks).total == 0, d.meta.key
        assert [c.chunk_id for c in chunks] == [f"{d.meta.key}#{n}" for n in range(len(chunks))]
        for c in chunks:
            assert c.text.startswith(d.meta.title) and len(c.sections) == 1 and c.strategy == "structure"


def test_l92_a_table_chunk_keeps_its_header_row():
    c = next(c for c in structure_chunks(BY["SOP-PAX-021@1.4"]) if c.sections == ("assistance-codes",))
    assert c.text.startswith("Assistance for passengers with reduced mobility > Assistance codes\n\n")
    assert "| Code | What the passenger needs | Equipment |" in c.text and "| WCHC | Cannot walk" in c.text


def test_l92_index_is_per_tenant_and_refuses_another_embedder():
    idx = build()
    assert set(idx.stores) == {"zz", "yy"} and idx.embedder.id == "local-hashing-v1/512"
    assert all(c.meta.tenant == "yy" for c in idx.store("yy").chunks.values())
    s = ChunkStore("zz", HashingEmbedder())
    s.embedder = HashingEmbedder(256)
    with pytest.raises(ValueError, match="refusing vectors"):
        s.add(structure_chunks(BY["SOP-GND-022@1.1"]))


# ---------- 9.3 hybrid search, filters, reranking ----------

def test_l93_rrf_fuse():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60)
    assert [i for i, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_l93_matches_dates_owner_and_prefix():
    m = BY["SOP-GND-009@2.0"].meta                               # withdrawn 2026-10-01
    assert matches(m, Filters(as_of=dt.date(2026, 9, 30))) and not matches(m, Filters(as_of=TODAY))
    future = BY["SOP-GND-014@3.3"].meta                          # effective 2026-11-01
    assert not matches(future, Filters()) and matches(future, Filters(as_of=dt.date(2026, 11, 1)))
    cur = BY["SOP-GND-018@4.0"].meta
    assert matches(cur, Filters(owners=frozenset({"Baggage Services"}))) and not matches(cur, Filters(doc_prefix="SOP-PAX"))


def test_l93_hybrid_with_rerank_finds_the_hotel_rule_and_costs_it(retriever):
    res = retriever.search("hotel voucher limit without approval", AGENT, k=3, mode="hybrid", rerank=True)
    assert any("above 150 EUR" in h.chunk.text for h in res.hits)
    assert res.cost.rerank_calls == 1 and res.cost.query_tokens > 0 and res.cost.context_tokens > 0
    assert res.cost.usd > retriever.search("hotel voucher limit without approval", AGENT, k=3).cost.usd


# ---------- 9.4 access control and tenant isolation ----------

def test_l94_missing_filter_leaks_a_restricted_procedure():
    idx = build()
    q = "goodwill authority limit after a delay"
    leaky = Retriever(idx, enforce_access=False).search(q, AGENT, k=3)
    safe = Retriever(idx).search(q, AGENT, k=3)
    assert "MGR-007@1.3" in _docs(leaky)                      # the leak a missing filter causes
    assert "MGR-007@1.3" not in _docs(safe)
    assert "MGR-007@1.3" in _docs(Retriever(idx).search(q, DM, k=3))


def test_l94_post_filtering_starves_narrow_roles(retriever):
    sec = DIRECTORY["sec-sam"]
    pre = retriever.search("delay care passengers", sec, k=5, prefilter=True)
    post = retriever.search("delay care passengers", sec, k=5, prefilter=False)
    assert len(pre.hits) > 0 and len(post.hits) < len(pre.hits)
    assert all(h.chunk.meta.doc_id == "SEC-003" for h in pre.hits)


def test_l94_tenants_never_see_each_other(retriever):
    q = "rebooking rate on partner airlines"
    assert all(h.chunk.meta.tenant == "zz" for h in retriever.search(q, DM, k=8).hits)
    assert "YY-COM-005@1.0" in _docs(retriever.search(q, YY_DM, k=3))
    assert not can_read(Principal("x", "zz", ("duty_manager",)), BY["YY-COM-005@1.0"].meta)


# ---------- 9.5 freshness ----------

def test_l95_cdc_revokes_within_the_window_and_naive_sync_does_not():
    from labs.s09 import freshness_demo as F
    assert F.naive()["still_there"] is True                    # the zombie policy
    c = F.cdc()
    assert c["revocation_s"] is not None and c["revocation_s"] <= F.FRESHNESS_WINDOW_S
    assert c["replay_same_size"] is True
    stuck = F.cdc(fail=True)
    assert stuck["gone_at"] is None and stuck["tool"]["error"] == "index_stale"


# ---------- 9.6 retrieval evals ----------

def test_l96_recall_and_reciprocal_rank():
    ranked = [set(), {"A#x"}, set(), {"A#y"}]
    assert recall_at_k(ranked, {"A#x", "A#y"}, 2) == 0.5 and recall_at_k(ranked, {"A#x", "A#y"}, 4) == 1.0
    assert reciprocal_rank(ranked, {"A#x", "A#y"}) == 0.5 and reciprocal_rank(ranked, {"B#z"}) == 0.0


def test_l96_structure_chunking_beats_naive_on_the_labelled_set():
    structure = evaluate(Retriever(build()), k=5, mode="hybrid", rerank=True)
    naive = evaluate(Retriever(build(strategy="naive")), k=5, mode="hybrid", rerank=True)
    assert structure.recall >= naive.recall and structure.mrr > naive.mrr
    assert structure.recall >= 0.9


# ---------- 9.7 agentic retrieval ----------

def test_l97_the_budget_stops_a_runaway_search():
    from labs.s09 import agentic_demo
    res, bs = agentic_demo.run("runaway")
    assert bs.searches == agentic_demo.BUDGET.max_searches and bs.refused >= 1 and res.stop == "answered"
    assert all("MGR-007" not in str(e) for e in bs.log)
    planned, bp = agentic_demo.run("planned")
    assert bp.searches <= agentic_demo.BUDGET.max_searches and "SOP-GND-014" in planned.answer


# ---------- 9.9 lab: access-controlled retrieval with a revoked-policy test ----------

def _tool(retriever, who, **kw):
    return as_tool(make_search_contract(retriever, lambda: DIRECTORY[who], **kw), None, "read")


def test_l99_search_procedures_is_a_clean_read_contract(retriever):
    c = make_search_contract(retriever, lambda: AGENT)
    assert lint_contract(c) == []
    props = c.args_model.model_json_schema()["properties"]
    assert set(props) == {"question", "k"}                      # no tenant, no role: they come from the caller


def test_l99_roles_decide_what_comes_back(retriever):
    q = SearchProceduresArgs(question="goodwill authority limit for a duty manager", k=4)
    agent = _tool(retriever, "agent-anna").fn(q)
    dm = _tool(retriever, "dm-deepa").fn(q)
    assert all(i["doc_id"] != "MGR-007" for i in agent["items"])
    assert dm["items"][0]["doc_id"] == "MGR-007" and dm["as_of"] == "2026-10-09"
    assert {"doc_id", "version", "section", "effective", "text"} <= set(dm["items"][0])


@pytest.mark.parametrize("question", ["hotel voucher limit without approval", "hotel vouchers for overnight delays",
                                      "can agents issue 250 EUR hotel vouchers", "SOP-GND-009 voucher limits"])
@pytest.mark.parametrize("who", ["agent-anna", "dm-deepa", "ramp-raj"])
def test_l99_revoked_policy_is_never_returned(retriever, question, who):
    """Regression: SOP-GND-009 (hotel vouchers up to 250 EUR without approval) was withdrawn on 2026-10-01.
    It is still IN this index, the worst case; it must never come back, for any role or wording."""
    out = _tool(retriever, who).fn(SearchProceduresArgs(question=question, k=8))
    assert all(i["doc_id"] != "SOP-GND-009" for i in out["items"])
    assert all("250 EUR" not in i["text"] for i in out["items"])


def test_l99_current_rule_replaces_it_and_future_versions_wait(retriever):
    out = _tool(retriever, "agent-anna").fn(SearchProceduresArgs(question="hotel voucher limit without approval", k=4))
    assert any(i["doc_id"] == "SOP-GND-014" and i["version"] == "3.2" and "150 EUR" in i["text"] for i in out["items"])
    later = _tool(retriever, "agent-anna", as_of=lambda: dt.date(2026, 11, 2)).fn(
        SearchProceduresArgs(question="meal voucher value per passenger", k=3))
    assert later["items"][0]["version"] == "3.3"


def test_l99_stale_index_fails_closed(retriever):
    out = _tool(retriever, "agent-anna", fresh=lambda: False).fn(SearchProceduresArgs(question="hotel voucher"))
    assert out["error"] == "index_stale" and out["category"] == "retryable"


def test_l99_over_mcp_the_token_decides():
    from labs.s09 import procedures_demo
    out = procedures_demo.run()
    assert all(i["doc_id"] != "MGR-007" for i in out["agent-anna"]["items"])
    assert out["dm-deepa"]["items"][0]["doc_id"] == "MGR-007"
    assert all(i["doc_id"].startswith("YY-") for i in out["yy-dm-yara"]["items"])
