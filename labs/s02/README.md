# Section 2 labs: Python for agent engineering (skippable refresher)

Start at tag `s02-start`. `make fitness` fails with a message naming the lecture to finish. Everything runs offline: the Flight Operations API is simulated through `httpx.MockTransport`, and latency is scaled down so tests run in under a second.

| Lecture | File | You implement | Time |
|---|---|---|---|
| 2.1 (demo) | `contracts.py` | nothing: read and run it | 10 min |
| 2.2 | `concurrent.py` | `fetch_all` (bounded concurrency, per-call timeouts, clean cancellation) | 30 min |
| 2.3 (demo) | `http_client.py` | nothing: read and run it | 10 min |
| 2.4 | `fakes.py` | `tool_reply`, `ScriptedModel.send` | 25 min |
| 2.5 | `skyops/tools/tape.py` | `ToolTape.wrap` (record and replay tool calls) | 30 min |
| 2.6 (demo) | `skyops/logs.py` | nothing: read and run the tests | 10 min |
| 2.7 | `assignment.py` | `AsyncFlightOpsClient.get_status` (graded, with hidden checks) | 45 min |

No live tokens are needed for any Section 2 lab.

## Hint ladder

**Lecture 2.2, `fetch_all`**
1. Create a `Semaphore(max_concurrency)`, and acquire it inside each per-flight coroutine.
2. Wrap each call in `async with asyncio.timeout(timeout_s)`. Turn `TimeoutError` into `Result(..., error="timeout")`.
3. Start the calls in an `asyncio.TaskGroup`, so cancelling the batch cancels every child. Return results in input order.

**Lecture 2.4, the scripted model**
1. `tool_reply` mirrors `text_reply`: one content block of type `tool_use` with an `id`, `name` and `input`, and the stop reason `tool_use`.
2. `send` records `req.json`, then returns `HttpResponse(200, {}, body=next reply)`.
3. When the replies run out, raise `AssertionError` saying the code made more model calls than the script has replies.

**Lecture 2.5, `ToolTape.wrap`**
1. Return a function with the same signature as `run_tool(name, args)`.
2. In record mode, call the real runner and append `{name, args, ok, output}`.
3. In replay mode:
   - serve entries in order;
   - raise `ToolReplayMiss` if the name or args differ, or if the tape runs out;
   - never call the real tools.

**Lecture 2.7, the assignment**
1. Branch on `status_code` in this order: 200, 404, 429/503, other 5xx, other 4xx.
2. Wait with the injected `sleep`: `Retry-After` if present, otherwise `2 ** (attempt - 1)`. Don't sleep after the last attempt.
3. Validate the 200 body with `FlightStatus`, and turn a `ValidationError` into `OpsDataError`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: no running event loop` | Calling a coroutine without `asyncio.run` | Run the batch with `asyncio.run(...)` |
| Peak concurrency above the limit | Semaphore acquired outside the per-call coroutine | Acquire inside `one()` |
| Cancellation test leaves calls running | Used `gather` without cancelling children | Use `TaskGroup` |
| Retry test waits `[1, 1]` instead of `[1, 3.0]` | `Retry-After` ignored | Read `r.headers.get("Retry-After")` |
| Test hangs | Real `asyncio.sleep` used in retries | Use the injected `self.sleep` |

## Break it

- In `fetch_all`, remove the semaphore and run the 2.2 checks. Which assertion fails, and what does it protect in production?
- In the tool tape, replay without checking the arguments. Which test catches it?
