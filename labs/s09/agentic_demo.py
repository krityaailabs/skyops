"""Lecture 9.7: agentic retrieval, bounded by code. Two runs of the Section 4 loop with one tool, search_procedures,
wrapped in a BudgetedSearch (skyops/retrieval/agentic.py).

    planned    a duty manager's two-part question: the model splits it and searches per part; the third
               search finds nothing new, and the model answers with citations
    runaway    an agent asks for something no procedure THEY can read contains (goodwill limits are for duty
               managers). The model keeps rephrasing; the budget stops the hunt after four searches

    python -m labs.s09.agentic_demo
    python -m labs.s09.agentic_demo --budget 10,5     break it: a looser budget (max searches, max no-progress)
The model's turns are a scripted replay (fixtures/s09/agentic_*.jsonl); the searches, the access filter and
the budget are real code, every run. With the default budget the replay is strict (every request must match
the recording); with --budget it is matched by sequence, because the tool results, and so the requests, change.
The scripted model then still proposes only the searches it was scripted with: a live model has no such limit.
"""
from __future__ import annotations

from skyops.agent.loop import RunResult, ToolBox, run_agent
from skyops.model.client import get_client
from skyops.retrieval.access import DIRECTORY
from skyops.retrieval.agentic import BudgetedSearch, RetrievalBudget
from skyops.retrieval.indexing import build
from skyops.retrieval.search import Retriever
from skyops.retrieval.tool import make_search_contract

SCENARIOS = {
    "planned": ("dm-deepa", "ZZ214 is now estimated after 22:00, and one passenger travels with a powered wheelchair. "
                            "What care do we owe the passengers, and what must we check for the wheelchair?"),
    "runaway": ("agent-anna", "A passenger on the delayed ZZ214 is angry. What is the maximum goodwill I can offer?"),
}
BUDGET = RetrievalBudget(max_searches=4, max_no_progress=1, max_result_tokens=2400)


def run(name: str, client=None, retriever: Retriever | None = None,
        budget: RetrievalBudget = BUDGET) -> tuple[RunResult, BudgetedSearch]:
    who, question = SCENARIOS[name]
    r = retriever or Retriever(build())
    bs = BudgetedSearch(make_search_contract(r, lambda: DIRECTORY[who]), budget)
    box = ToolBox().add(bs.tool())
    match = "strict" if budget == BUDGET else "sequence"
    res = run_agent(question, client or get_client(fixture=f"fixtures/s09/agentic_{name}.jsonl", match=match), box)
    return res, bs


def parse_budget(argv: list[str]) -> RetrievalBudget:
    if "--budget" not in argv:
        return BUDGET
    searches, no_progress = (int(x) for x in argv[argv.index("--budget") + 1].split(","))
    return RetrievalBudget(max_searches=searches, max_no_progress=no_progress,
                           max_result_tokens=BUDGET.max_result_tokens)


def main() -> None:
    import sys
    budget = parse_budget(sys.argv)
    r = Retriever(build())
    if budget != BUDGET:
        print(f"budget: {budget.max_searches} searches, {budget.max_no_progress} without progress "
              f"(replay matched by sequence)")
    for name in SCENARIOS:
        res, bs = run(name, retriever=r, budget=budget)
        print(f"== {name} ({SCENARIOS[name][0]})")
        for e in bs.log:
            extra = f" new={e['new']} repeated={e['repeated']}" if e["outcome"] == "ok" else ""
            print(f"  search {e['question']!r}: {e['outcome']}{extra}")
        s = res.trace.summary()
        print(f"  searches run: {bs.searches}, refused: {bs.refused}, evidence tokens: {bs.result_tokens}, "
              f"model calls: {s['model_calls']}, stop: {res.stop}")
        print(f"  answer: {res.answer}")


if __name__ == "__main__":
    import logging
    import warnings
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")          # sequence replay warns on every changed request
    main()
