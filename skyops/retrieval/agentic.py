"""Lecture 9.7: agentic retrieval with a budget the code enforces.

The model plans: it splits a question into sub-questions and proposes one search per sub-question. Your code
bounds it. BudgetedSearch wraps the search_procedures contract for one run and applies three stopping rules,
whatever the model proposes:

    max_searches       searches per question (default 4). After that: search_budget_exhausted.
    no progress        a search whose passages were ALL returned before counts as no progress (counted over
                       the whole question, not consecutively); after max_no_progress of them, further
                       searches return no_new_results.
    max_result_tokens  evidence handed to the model per question. Once a passage would take the total past it,
                       that result is cut there and says so ("evidence_capped": true), and every later search
                       returns evidence_budget_exhausted. A cut is not "no progress": the passages were new.

Passages already returned are not sent again (deduplicated by document, version, section and passage text,
so two chunks of one long section are both kept), so rephrasing the same search costs a search but no
context tokens. All three refusals are non_retryable taxonomy results whose message tells the model what to
do next: answer from what it has, and say what is missing. The Section 4 loop's own budgets (steps, tokens,
seconds) still bound the run around it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agent.loop import Tool
from ..tools.contracts import ToolContract, ToolError, as_tool, estimate_tokens


@dataclass(frozen=True)
class RetrievalBudget:
    max_searches: int = 4
    max_no_progress: int = 1
    max_result_tokens: int = 2400


@dataclass
class BudgetedSearch:
    contract: ToolContract
    budget: RetrievalBudget = field(default_factory=RetrievalBudget)
    searches: int = 0
    no_progress: int = 0
    result_tokens: int = 0
    refused: int = 0
    capped: bool = False
    seen: set[tuple[str, str, str, str]] = field(default_factory=set)
    log: list[dict[str, Any]] = field(default_factory=list)

    def _refuse(self, code: str, message: str, question: str) -> dict[str, Any]:
        self.refused += 1
        self.log.append({"question": question, "outcome": code})
        return ToolError("non_retryable", code, message).as_result()

    def run(self, args) -> dict[str, Any]:
        b = self.budget
        if self.searches >= b.max_searches:
            return self._refuse("search_budget_exhausted",
                                f"All {b.max_searches} searches for this question are used. Answer from the passages "
                                "you have, and say what is still missing.", args.question)
        if self.capped:
            return self._refuse("evidence_budget_exhausted",
                                f"The evidence budget for this question ({b.max_result_tokens} tokens) is used. Answer "
                                "from the passages you have, and say what is still missing.", args.question)
        if self.no_progress >= b.max_no_progress:
            return self._refuse("no_new_results",
                                "Your last search found nothing new. Stop searching: answer from the passages you have, "
                                "and say what is still missing.", args.question)
        out = as_tool(self.contract, None, scope="read").fn(args)
        self.searches += 1
        if "error" in out:
            self.log.append({"question": args.question, "outcome": out["error"]})
            return out
        new, repeated, cut = [], 0, False
        for it in out["items"]:
            key = (it["doc_id"], it["version"], it["section"], it["text"])
            if key in self.seen:
                repeated += 1
                continue
            if self.result_tokens + estimate_tokens(it) > b.max_result_tokens:
                cut = self.capped = True
                break
            self.seen.add(key)
            self.result_tokens += estimate_tokens(it)
            new.append(it)
        if not new and not cut:
            self.no_progress += 1
        self.log.append({"question": args.question, "outcome": "ok", "new": len(new), "repeated": repeated})
        if cut:
            note = "Evidence budget reached: later passages were not returned. Answer from what you have."
        elif not out["items"]:
            note = out["note"]
        else:
            note = "Nothing new: these passages were already returned." if not new else None
        res = {"items": new, "already_seen": repeated, "searches_left": b.max_searches - self.searches,
               "as_of": out["as_of"], "note": note}
        if cut:
            res["evidence_capped"] = True
        return res

    def tool(self) -> Tool:
        return Tool(self.contract.name, self.contract.args_model, self.run, self.contract.description)
