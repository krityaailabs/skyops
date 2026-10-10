# Section 3 lab: break the quick-win agent (lecture 3.3)

Start at tag `s03-start`. Run `python -m labs.s03.break_it` to watch five breaks happen. Then implement the five detectors in `detectors.py` until `make fitness` passes. Time: about 40 minutes. Live tokens: none, because the scenarios replay recorded conversations.

The scenario conversations are scripted on purpose. They show model behaviours that real systems do produce: looping, malformed arguments, retrying after a timeout, and following instructions found in data. A live model won't produce them on demand, so these recordings stay scripted even after the course's other fixtures are re-recorded live.

| Break | What happens | Detector you write |
|---|---|---|
| loop | The model repeats the same tool call until the step budget stops it | `detect_loop` |
| bad_arguments | The model passes a description where a flight number belongs | `detect_bad_arguments` |
| slow_tool | A status call takes 30 simulated seconds and nothing times out | `detect_slow_tool` |
| duplicate | The gateway sends, then reports a timeout; the model sends again | `detect_duplicate_side_effects` |
| injection | A remark field carries an instruction; the model follows it and the canary reaches the outbox | `detect_canary` |

The injection uses a harmless canary string, `CANARY-7F3`. Success is measured only by whether the canary crosses into the outbox. Nothing leaves your machine.

## Hint ladder

1. **`detect_loop`:** count tool calls by `(name, args)`. Serialise the args with sorted keys so equal dicts count as equal. Repeats don't have to be consecutive.
2. **`detect_duplicate_side_effects`:** the key is `(flight_number, message)`. The same text sent to different flights isn't a duplicate.
3. **`detect_slow_tool`:** respect the `budget_ms` argument. **`detect_canary`:** check every canary against every outbound message.

## Break it (further)

Raise `MAX_STEPS` in `skyops/agent/quickwin.py` to 50 and rerun the loop scenario. Replay stops at the seventh call with a `ReplayMiss` that points at the step budget, because only six calls were recorded. In live mode, nothing would stop it until the 50th call. Work out the tokens and cost per 1,000 runs at 50 steps from the six-step numbers, and explain why the step budget is a cost control as well as a safety one.
