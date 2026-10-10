# Section 4 labs: the agent loop from scratch (Project 1)

Start at tag `s04-start`. Every function below raises `LabNotDone` until you write it; `make fitness` names the next one. Work in lecture order; each helper has its own check. `run_agent` (4.2) calls the four helpers from 4.3–4.6, so `test_l42` passes only once they all exist. Until then, its failure names whichever helper the loop reaches next: `check_budgets`, then `detect_runaway`, then `execute`, then `validate_call`. Time: about 3 hours, plus 3–5 hours for Project 1. Live tokens: none; every lab replays recordings. Optional live run of the eval: about 20,000 tokens per pass.

| Lecture | File | Implement | Check |
|---|---|---|---|
| 4.2 | `skyops/agent/loop.py` | `run_agent` | `test_l42_*` |
| 4.3 | `skyops/agent/loop.py` | `validate_call` | `test_l43_*` |
| 4.4 | `skyops/agent/loop.py` | `check_budgets` | `test_l44_*` |
| 4.5 | `skyops/agent/loop.py` | `execute` | `test_l45_*` |
| 4.6 | `skyops/agent/loop.py` | `detect_runaway` | `test_l46_*` |
| 4.7 | `labs/s04/eval.py` | `check_case`, `run_eval` | `test_l47_*` |
| 4.8 | `projects/p1_flight_status/assistant.py` | `chosen_flight`, `answer` | `test_l48_*` (see `BRIEF.md`) |

Run things as you go:

- `python -m labs.s04.demo` and `python -m labs.s04.demo c05_bad_args_recover` print a trace.
- `python -m labs.s04.runaway` shows the three runaway modes, uncapped and capped.
- `python -m labs.s04.eval` prints the ten-case score.
- `python -m projects.p1_flight_status.assistant` prints Project 1's grounded answers.

## Hint ladder

**4.2 `run_agent`**
1. The order inside the loop: check budgets and the runaway guard, call the model, append its message, stop if it didn't ask for tools, otherwise run every proposed call and append all the results as one user message.
2. Count a step after each model call. Use `trace.usage.total` for tokens and `clock() - start` for time.
3. On a stop, call `trace.answer(f"[{stop}] {STOP_MESSAGES[stop]}")` and return the message as the answer.

**4.3 `validate_call`**
1. Look the tool up in `box.tools`. Missing means `unknown_tool`.
2. `tool.args_model.model_validate(call.input)` raises `ValidationError`. Take `e.errors(include_url=False)[0]` and build `"<field>: <msg>"` from its `loc` and `msg`.

**4.4 `check_budgets`**
1. Use `>=`: a budget of 8 steps means the ninth call never happens.
2. Check time, then tokens, then steps.

**4.5 `execute`**
1. Validate first and return the validation error unchanged.
2. Catch `TimeoutError` and `ConnectionError` separately from everything else. Put the exception's *type name* in the message, never its text: exception text can carry secrets.
3. A result with an `"error"` key is a business answer, so mark it `is_error` and `retryable: false`.

**4.6 `detect_runaway`**
1. Count `(name, args)` pairs. Serialise args with `json.dumps(..., sort_keys=True)` so key order doesn't hide a repeat.

**4.7 `check_case`, `run_eval`**
1. Collect every failure; don't return at the first.
2. Use the exact failure strings in the docstring: the check compares them. Use `run.steps` for the step count.

**4.8 `chosen_flight`, `answer`** (see `projects/p1_flight_status/BRIEF.md`)
1. `chosen_flight`: walk `run.tool_calls` backwards and stop at the first lookup. A status call names the flight; a departures lookup names it only if exactly one flight matches.
2. `answer`: never put the model's wording in a grounded answer. Read the record with `sim.find` and use `describe`.
3. Ambiguous: use `ambiguous(run)` to list the flights found and ask which one, with `grounded=False`.

## Break it

1. Set `max_repeats` to 10 in `CAPPED` and rerun `python -m labs.s04.runaway`. Which budget stops the repeat mode now, and what does it cost per 1,000 runs?
2. In `execute`, put `str(e)` in the error message and run the hidden-check idea yourself: make a tool raise `ValueError("password=…")` and look at what the model would see.
3. Delete `forbid` from case c10 and rerun the eval. The score goes up. Is the agent better?
