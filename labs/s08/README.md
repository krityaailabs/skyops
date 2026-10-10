# Section 8 labs: context, state and memory

Start at tag `s08-start`. Nine of the fourteen Section 8 checks fail until you write the four functions below (five pass from the start); `make fitness` names the next one. Time: about 3 hours for the section (8.7 and 8.8's four functions, plus reading and running the other labs); the 8.8 lab itself is about an hour. Live tokens: none (every model conversation is a scripted replay, disclosed in the lectures). Network: none.

Token counts in this section are **estimates** (`skyops/context/tokens.py`): no provider tokenizer runs offline. The authoritative count is the provider's `usage` in each response. The `usage` numbers inside the replayed fixtures (and the trace's token totals built from them) are **synthetic** too: `tools/build_fixtures.py` writes about four characters per token of the request JSON, plus 346 for the tool-use system prompt. `make record` with a real key replaces them with the provider's own counts. The labs use deliberately small context windows, so overflow and compaction happen in a few turns instead of a few hours.

| Lecture | File | Implement | Check |
|---|---|---|---|
| 8.1 | `skyops/context/budget.py`, `tokens.py` | nothing: run `python -m labs.s08.budget` and decide what to cut first | `test_l81_*` |
| 8.2 | `skyops/context/state.py` | nothing: sort the twelve pieces yourself, then compare with `python -m labs.s08.state_map` | `test_l82_*` |
| 8.3 | `skyops/context/compaction.py`, `manager.py` | nothing yet: watch `python -m labs.s08.compaction_demo` (you write `compact` in 8.8) | `test_l83_*` |
| 8.4 | `skyops/context/scratchpad.py` | nothing: run `python -m labs.s08.scratchpad` and compare the two traces | `test_l84_*` |
| 8.5 | `skyops/context/memory.py`, `docs/memory_policy.md` | write your own one-page policy for a memory you'd build at work | `test_l85_*` |
| 8.6 | `skyops/context/memory.py` | nothing: run `python -m labs.s08.memory_demo`, then break it (below) | `test_l86_*` |
| 8.7 | `skyops/context/window.py` | `pinned_missing`, `fit_to_window` | `test_l87_*` |
| 8.8 | `skyops/context/compaction.py`, `retention.py` | `compact`, `retention_problems` | `test_l88_*`, `test_l83_*` |

Run things as you go: `python -m labs.s08.incident`, `python -m labs.s08.shift`, or all of them with `make labs-s08`.

## Expected output

`python -m labs.s08.incident` at `s08` prints two timelines. On the v1 builder, a late request in turn six shows `rules lines 3/5 · pinned MISSING skyops-rules`, and the outbox holds the canary. On the fixed builder, every request shows `pinned ok`, and the outbox holds only the notice the duty manager asked for.

## Hint ladder

**8.7 `pinned_missing`**
1. A block is intact only if its whole `render()` (marker, text, closing tag) is in `req.system`.
2. Look only in `req.system`: copies in the messages don't count, and `None` means every block is missing.
3. Return ids in the order of `blocks`, so two missing blocks always report the same way.

**8.7 `fit_to_window`**
1. `turn_starts(req.messages)` gives the indices where you may cut. Keeping `messages[start:]` for a later start drops one more turn.
2. Never touch `system` or `tools`: `req.model_copy(update={"messages": kept})`.
3. Out of turns and still too big: `ContextOverflow(account(last_turn_only), window.input_limit)`. Then, for every result (even an unchanged one), `pinned_missing` must be empty, or raise `PinnedRulesMissing`.

**8.8 `compact`**
1. Cut at `starts[-keep_turns]`, only when there are more than `keep_turns` turns.
2. Rolling: previous summary lines first, then `summarise(messages[:cut])`. Strip an existing `- (N earlier lines omitted)` line before adding, and carry its N forward.
3. With an omission line, the summary is still at most `max_lines` lines in total.

**8.8 `retention_problems`**
1. Number requests from 1. For each one, the four checks in the docstring's order.
2. `str.count(block.marker())` finds duplicates; `orphaned_results` and `account` do the rest.
3. The over-limit message uses plain integers (`f"{total}"`, not `f"{total:,}"`), and an empty `sent` list returns `[]`.

Still stuck after three hints? The reference solution is the tag `s08` (`git diff s08-start s08 -- skyops/context`).

## Break it

- 8.6: change `MemoryStore.write` to accept `source="tool_result"`, and run `python -m labs.s08.memory_demo`. Which candidate gets through, and what would it do in every future conversation? Now remove `INSTRUCTION` from `screen()` too. What stops it now? (Nothing: that's why provenance comes first.)
- 8.4: in `WorkingMemory.update`, accept facts with an empty source and run `pytest fitness/s08 -k l84`. The rejection check fails, and the trace check fails with a replay miss: the model now sees a different scratchpad.
- 8.8: in `compact`, ignore `previous_summary` and run `pytest fitness/s08 -k "l83 or l88"`. Three checks fail: two with a replay miss at `shift.jsonl call 8`, one because the summary is no longer rolling.
- 8.7: in `labs/s08/incident.py`, set `enforce_pinned=True` in `v1_context()` and rerun. The old builder still cuts the rules, but now the detection signal is a guard: which request is never sent, what does the duty manager see, and where did the canary go? Then ask why this is a mitigation, not the fix.

## Troubleshooting

1. `LAB NOT DONE: Lecture 8.7: implement fit_to_window()` from an 8.1 or 8.4 check: every request goes through the context layer, which calls your two 8.7 functions. Write them first.
2. `ReplayMiss ... your request differs from the recording`: your context layer built a different request from the one recorded (for example, a different system-prompt order, or a summary with different lines). Compare with the docstring, step by step.
3. `test_l88` reports an orphaned tool result: you cut between a tool call and its result. Cut only at `turn_starts`.
4. Retention fails with `appears 2 times`: the summary or the scratchpad contains the rules. Compaction must never see the system prompt.
5. A memory check fails on a harmless sentence: your pattern is too broad. Patterns are the second line of defence; don't make them the first. (The booking-reference pattern reads any six-character mix of capitals and digits, like `SHA256`, as a reference: a known false positive.)

## Self-check

- Can you say, for the last request of the ten-step run, which component you'd cut first and why the rules are never on the list?
- Does every request your code sends pass `retention_problems`, on the shift and on the v1 run's failure?
- Could your memory policy be read by someone in legal, and does each rule in it have a test?

Hidden checks (instructor repository) test behaviours the docstrings already state.
