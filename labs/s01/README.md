# Section 1 labs: LLM foundations for engineers

Start at tag `s01-start`. Run `make fitness` and every Section 1 check fails with a message naming the lecture to finish. When all of them pass, you've reached the `s01` state.

Every lab runs in replay mode with no key and no cost.

| Lecture | File | You implement | Estimated time | Live-mode tokens (estimate) |
|---|---|---|---|---|
| 1.2 | `first_call.py` | `ask`, `stream_answer`, `call_with_retry` | 30 min | about 1,000 |
| 1.3 | `extract.py` | `extract_notice` (native schema, validation, one repair) | 40 min | about 3,000 |
| 1.4 | `costing.py` | `report` (cost per call and per 1,000 calls) | 20 min | none |
| 1.7 | `skyops/model/providers/openai_responses.py` | `_input_items`, `parse_response` | 40 min | about 2,000 |
| 1.8 | `checkpoint.py` | `extract_batch` | 30 min | about 3,000 |

## Expected output at `s01`

```
make labs-s01
answer: An AIDX message carries operational flight data, ...
stop: end_turn (provider said end_turn)
after errors: 3 calls, waited [2.0, 2.0] seconds, ...
ollama    calls=2  {'flight_number': 'ZZ173', 'status': 'delayed', ...}
5 calls, ... USD per 1,000 calls (example prices), input share ..., failures: none
```

## Hint ladder (try each hint before reading the next)

**Lecture 1.2, `call_with_retry`**
1. Catch `ModelError` and use `e.retryable`. Don't read the error message text.
2. When `e.retry_after` is set, sleep exactly that long. Otherwise back off exponentially: `base_delay_s * 2 ** (attempt - 1)`.
3. Re-raise on the last attempt and on any non-retryable error. The test expects one call for a 400 error.

**Lecture 1.3, `extract_notice`**
1. Pass `output_schema=DisruptionNotice.model_json_schema()` in the `ChatRequest`. Check `resp.stop_reason` before parsing: a cut-off or refused answer raises `ExtractionFailed` straight away.
2. Validate with `DisruptionNotice.model_validate(json.loads(resp.text))`, even when the provider constrained the output. Constrained decoding guarantees the shape, not the truth.
3. On failure, append the model's reply and a user message describing the validation error, then call again. Allow one repair, then raise `ExtractionFailed`.

**Lecture 1.4, `report`**
1. Sum the `Usage` objects; `Usage` supports `+`.
2. Price the total with `cost_usd`, and divide by the number of calls for the per-call figure.
3. `input_share` is the cost of input tokens (including cache tokens) divided by the total cost.

**Lecture 1.7, the OpenAI adapter**
1. Assistant tool calls become `function_call` items. `arguments` must be a JSON string, so use `json.dumps`.
2. Tool results become `function_call_output` items with the same `call_id`.
3. In `parse_response`:
   - skip `reasoning` items;
   - `json.loads` the arguments;
   - split `cached_tokens` out of `input_tokens`;
   - map `status: incomplete` with reason `max_output_tokens` to `max_tokens`.

**Lecture 1.8, `extract_batch`**
1. Wrap `client.complete` so every call, including repair calls, is collected for the cost report.
2. Catch `ExtractionFailed` for each notice. Append `None`, record the error, and carry on.
3. Restore `client.complete` in a `finally` block.

## Troubleshooting (the five most likely failures)

| Symptom | Cause | Fix |
|---|---|---|
| `LAB NOT DONE: Lecture 1.x …` | The function is still the stub | Implement it; read the docstring first |
| `ReplayMiss: … made more model calls than were recorded` | Your code called the model more times than the recording | Check the repair loop allows exactly one repair |
| `UserWarning: your request differs from the recording` | Your prompt wording differs | Fine in labs ("sequence" mode). CI uses strict mode |
| `ValidationError … can't have a new departure time` | A business rule failed (on_time or cancelled with a new time) | That's the point of lecture 1.3: validate, then repair once |
| `ModelError [skyops:budget_exceeded]` in live mode | The spend guard blocked a call | Raise `SKYOPS_MAX_USD_PER_RUN` deliberately, or lower `max_tokens` |

## Break it

- In `extract.py`, set `max_repairs=0` and run `python -m labs.s01.extract --provider ollama`. Explain the failure in one sentence.
- In `first_call.py`, make `call_with_retry` retry 400 errors too. Which test catches it, and what would the change cost in production?
