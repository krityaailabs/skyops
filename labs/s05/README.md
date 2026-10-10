# Section 5 labs: tool design and contracts

Start at tag `s05-start`. Five checks fail until you write the functions below; `make fitness` names the next one. Time: about 2.5 hours. Live tokens: none (replay). Optional: re-record the tool-choice demo live (about 12,000 tokens for all sixteen recordings) and compare your numbers with the course's; run it a few times, because eight questions is a small sample.

| Lecture | File | Implement | Check |
|---|---|---|---|
| 5.2 | `labs/s05/descriptions.py` | nothing: run it and read it | `test_l52_*` (passes from the start) |
| 5.5 | `skyops/tools/contracts.py` | `idempotency_key`, `execute_write` | `test_l55_*` |
| 5.6 | `skyops/tools/contracts.py` | `classify_exception` | `test_l56_*` |
| 5.7 | `skyops/tools/contracts.py` | `shape_result` | `test_l57_*` |
| 5.8 | `skyops/tools/contracts.py` | `lint_contract` | `test_l58_*` |
| 5.9 | `skyops/tools/catalog.py` | `list_departures`, `check_voucher` | `test_l59_*` |

Run things as you go: `python -m labs.s05.descriptions`, `python -m labs.s05.review_tools`, `python -m labs.s05.disruption` and `python -m labs.s05.disruption --dry-run`.

## Hint ladder

**5.5 `idempotency_key`, `execute_write`**
1. The key covers only `contract.key_fields`, plus the tool name and the scope. Canonical JSON (`sort_keys=True`), then sha256, first 24 hex characters.
2. In `execute_write`, run the precheck first, then handle dry run (remember previewed keys), then the store. Store only successful results, with the arguments they ran with: a failed write must be allowed to happen later, and a repeat with different details is an `idempotency_conflict`.
3. Use `args.model_dump(mode="json")` so dates become text, and `normalise_text` on string key fields.

**5.6 `classify_exception`**
1. Check `ToolError` first and return it unchanged. Then timeouts, connection errors and anything with `retryable = True`.
2. The message is the exception's type name only.

**5.7 `shape_result`**
1. Select fields, then slice the page, then drop items from the end while the estimate is over the cap, keeping at least one.
2. If that one item is still too big, cut its longest string values (ending with "…") until it fits, and set `oversized`.
3. `next_cursor` is `cursor + len(items)`, or `None` when nothing is left.

**5.8 `lint_contract`**
1. Read the field names and descriptions from `args_model.model_json_schema()["properties"]`.
2. The test compares your codes with the "lint" findings in `review_tools.py`, exactly: no extra codes.

**5.9 `list_departures`, `check_voucher`**
1. `sim.departures_on(args.date.isoformat())` gives the requested date's flights in timetable order. Decode the cursor token with `decode_cursor`, and encode `next_cursor` with `encode_cursor`.
2. In `check_voucher`, the order of the three rules is the order of the docstring.

Hidden checks (instructor repository) test behaviours that every docstring above already states; if one surprises you, reread the docstring.

## Break it

1. Remove `"message"` from `notify_passengers`' `key_fields`. Send two *different* messages to the same flight in one scope. What happens to the second, and why is that worse than a duplicate?
2. Make `execute_write` skip the precheck in dry runs, then run `python -m labs.s05.disruption --dry-run`. What does the dry run now promise about the 180 EUR voucher?
