"""Lecture 8.4: typed working memory instead of a free-text scratchpad.

The model keeps its working state through one tool, update_scratchpad, whose arguments are typed:

    plan            ordered steps, each done or not
    facts           short statements, each with a SOURCE: the id of an earlier tool call whose result
                    mentions the fact's subject as a whole word. No source, no fact. A subject is a flight
                    number or booking reference (2 to 12 capitals and digits); a fact's text that reads like
                    an instruction is rejected (the memory screen's pattern).
    open_questions  things to find out; speculation goes here, never into facts

Your code validates every update, keeps the state, and renders it into each request as a <scratchpad> block.
The free-text alternative (write_notes) is kept for comparison: it accepts anything and grows every step.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ..agent.loop import Tool
from .memory import INSTRUCTION

MAX_FACTS = 12
MAX_QUESTIONS = 6


class PlanStep(BaseModel):
    step: str = Field(max_length=80, description="One step of the plan, a few words.")
    done: bool = Field(default=False, description="True once the step is finished.")


class FactIn(BaseModel):
    subject: str = Field(pattern=r"^[A-Z0-9]{2,12}$",
                         description="What the fact is about: a flight number or booking reference.")
    text: str = Field(max_length=120, description="The fact, in a few words, as the tool result states it.")
    source: str = Field(default="", description="The id of the earlier tool call whose result states this fact.")


class ScratchpadUpdate(BaseModel):
    plan: list[PlanStep] | None = Field(default=None, description="The whole plan; replaces the previous one.")
    add_facts: list[FactIn] = Field(default_factory=list, description="Facts to add, each with its source call id.")
    open_questions: list[str] = Field(default_factory=list, description="Questions to add: unconfirmed ideas go here.")
    resolved: list[str] = Field(default_factory=list, description="Ids of open questions now answered, e.g. Q1.")


class Fact(BaseModel):
    id: str
    subject: str
    text: str
    source: str
    tool: str


class Scratchpad(BaseModel):
    plan: list[PlanStep] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    questions: dict[str, str] = Field(default_factory=dict)
    rejected: int = 0
    next_q: int = 1

    def render(self) -> str:
        if not (self.plan or self.facts or self.questions):
            return ""
        lines = ["<scratchpad>"]
        if self.plan:
            lines.append("plan:")
            lines += [f"  [{'x' if s.done else ' '}] {i}. {s.step}" for i, s in enumerate(self.plan, 1)]
        if self.facts:
            lines.append("facts:")
            lines += [f"  {f.id} {f.subject}: {f.text} (source {f.source}, {f.tool})" for f in self.facts]
        if self.questions:
            lines.append("open questions:")
            lines += [f"  {q}: {t}" for q, t in self.questions.items()]
        lines.append("</scratchpad>")
        return "\n".join(lines)

    def status(self) -> str:
        done = sum(s.done for s in self.plan)
        sourced = sum(1 for f in self.facts if f.source)
        facts = (f"facts {len(self.facts)}" if not self.facts else
                 f"facts {len(self.facts)}, all sourced" if sourced == len(self.facts) else
                 f"facts {len(self.facts)}, {sourced} sourced")
        return f"plan {done}/{len(self.plan)} done · {facts} · open questions {len(self.questions)} · rejected {self.rejected}"


class WorkingMemory:
    """The typed scratchpad, plus what it needs to check sources: the successful tool results seen so far."""

    def __init__(self) -> None:
        self.pad = Scratchpad()
        self.results: dict[str, tuple[str, str]] = {}     # tool call id -> (tool name, result text)

    def render(self) -> str:
        return self.pad.render()

    def status(self) -> str:
        return self.pad.status()

    def observe(self, call_id: str, name: str, content: str, ok: bool) -> None:
        if ok and name != "update_scratchpad":
            self.results[call_id] = (name, content)

    def update(self, u: ScratchpadUpdate) -> dict[str, Any]:
        """Apply an update. Facts are accepted only with a source that exists and mentions the subject as a
        whole word (ZZ23 is not mentioned by ZZ230), and only if their text doesn't read like an instruction."""
        pad, accepted, rejected = self.pad, [], []
        if u.plan is not None:
            pad.plan = list(u.plan)
        for f in u.add_facts:
            src = self.results.get(f.source)
            if not f.source or src is None:
                reason = ("no source: give the id of the tool call whose result states this; "
                          "put unconfirmed ideas in open_questions")
            elif not re.search(rf"(?<![A-Z0-9]){re.escape(f.subject)}(?![A-Z0-9])", src[1].upper()):
                reason = f"source {f.source} ({src[0]}) does not mention {f.subject}"
            elif INSTRUCTION.search(f.text):
                reason = "instruction-like text: a fact states what a result says, never what to do"
            elif any(x.subject == f.subject and x.text == f.text for x in pad.facts):
                continue                                       # already known: not an error, not a duplicate
            elif len(pad.facts) >= MAX_FACTS:
                reason = f"scratchpad is full ({MAX_FACTS} facts): finish the task or drop the plan"
            else:
                fact = Fact(id=f"F{len(pad.facts) + 1}", subject=f.subject, text=f.text, source=f.source, tool=src[0])
                pad.facts.append(fact)
                accepted.append(fact.id)
                continue
            pad.rejected += 1
            rejected.append({"subject": f.subject, "text": f.text, "reason": reason})
        for q in u.resolved:
            pad.questions.pop(q, None)
        for q in u.open_questions[:max(0, MAX_QUESTIONS - len(pad.questions))]:
            pad.questions[f"Q{pad.next_q}"] = q[:120]
            pad.next_q += 1
        return {"accepted_facts": accepted, "rejected_facts": rejected,
                "open_questions": dict(pad.questions), "plan_steps": len(pad.plan)}

    def tool(self) -> Tool:
        return Tool("update_scratchpad", ScratchpadUpdate, self.update,
                    "Update your working memory for this task: the plan, facts and open questions. It is shown "
                    "back to you on every step. Each fact needs a source: the id of an earlier tool call whose "
                    "result states it. Put anything unconfirmed in open_questions. Changes nothing outside this run.")


class NotesArgs(BaseModel):
    notes: str = Field(max_length=4000, description="Your notes for this task. Replaces the previous notes.")


class FreeTextNotes:
    """The free-text alternative, for comparison: anything goes in, nothing is checked."""

    def __init__(self) -> None:
        self.notes = ""

    def observe(self, call_id: str, name: str, content: str, ok: bool) -> None:
        pass

    def render(self) -> str:
        return f"<scratchpad>\n{self.notes}\n</scratchpad>" if self.notes else ""

    def status(self) -> str:
        return f"notes {len(self.notes)} characters, unchecked"

    def write(self, a: NotesArgs) -> dict[str, Any]:
        self.notes = a.notes
        return {"saved": True, "characters": len(a.notes)}

    def tool(self) -> Tool:
        return Tool("write_notes", NotesArgs, self.write,
                    "Write notes for this task. They replace the previous notes and are shown back to you on every "
                    "step. Changes nothing outside this run.")
