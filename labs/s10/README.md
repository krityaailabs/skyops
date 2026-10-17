# Section 10 labs: evals foundations

Start at tag `s10-start`. The Section 10 checks fail until you write the functions below; `make fitness` names the next one. Time: about 4 hours. Live tokens: none. Network: none.

The system under test is the **disruption handler** (`skyops/agent/disruption.py`): the Section 4 loop with the Section 5 catalog. A duty manager pastes an ops notice (the kind Section 1 extracted); the handler checks the flight, tells passengers when a disruption is confirmed, issues vouchers it was asked for, and reports back.

The golden set is `data/evals/golden.jsonl` (18 cases: 6 normal, 6 edge, 6 adversarial; 5 of them are Section 1's notices), documented by `data/evals/golden.schema.json`. The labelled passenger messages are `data/evals/judge_labels.jsonl` (51 messages labelled pass or fail by the course author). Each has a `split`: the assertion rules of 10.5 were written against `dev` (21); the judge is calibrated only on `held_out` (30), which no rule or prompt was tuned on. `judge_pairs.jsonl` has 12 pairs, each a passed message against a failed one, for the position test.

| Lecture | File | Implement | Check |
|---|---|---|---|
| 10.1 | `labs/s10/inventory.py` | nothing yet: after 10.4, run it, then list three features you shipped without an eval | — |
| 10.2 | `labs/s10/edd.py` | nothing yet: after 10.4, run it; write five cases for a feature of yours before building it | — |
| 10.3 | `skyops/evals/dataset.py` | `validate_dataset` | `test_l103_*` |
| 10.4 | `skyops/evals/checks.py` | `check_outcome`, `check_tool_calls`, `check_trajectory` | `test_l104_*` |
| 10.5 | `skyops/evals/checks.py` | `message_assertions` | `test_l105_*` |
| 10.6 | `skyops/evals/judge.py`, `labs/s10/calibrate.py` | nothing: run the demo at tag `s10` (it uses 10.8's `cohen_kappa`) | `test_l106_*` |
| 10.7 | `labs/s10/variance.py` | nothing yet: after 10.8 (or at tag `s10`), run it; compute pass@2 and pass^2 by hand | — |
| 10.8 | `skyops/evals/stats.py`, `skyops/evals/harness.py` | `pass_at_k`, `pass_hat_k`, `cohen_kappa`, `score_attempt` | `test_l108_*` |

Run things as you go: `make labs-s10` runs every demo; `make eval` runs the harness and writes `runs/s10/report.md` and `runs/s10/report.json`.

About the replays: every model turn in this section is scripted (`meta.synthetic = true`). `fixtures/s10/attempts/<case>/a<n>.jsonl` are the handler's five attempts per case; which attempts fail, and how, is written down in `labs/s10/build_fixtures_s10.py` (`SCHEDULE`), not sampled. `fixtures/s10/judge.jsonl` holds the judge's verdicts, keyed by request; they are written to show position and verbosity effects, and measure no real model. The loop, the tools, the gateways, every check, the statistics and the report are real code. To measure a real model, re-record with `SKYOPS_MODE=record` and your keys; nothing else changes.

## Hint ladder

Three hints per function. Take one at a time; the third is the full solution, for when you'd rather read it than stay stuck. The solved tag `s10` has the same code.

### 10.3

**`validate_dataset`**
1. Three passes, in the docstring's order: ids first (a `seen` set), then category counts (a `Counter` over `CATEGORIES`), then one loop over the cases.
2. "Unlabelled" means empty after `.strip()`, for `label.by` or `label.why`. For the simulator checks, `sim.find(notify.flight, sim.DATE)`; map `delay` to `DELAYED` and `cancellation` to `CANCELLED`. When the flight doesn't exist, report it and `continue` to the next case.
3. The full solution:

   ```python
   def validate_dataset(cases, min_per_category=MIN_PER_CATEGORY):
       problems: list[str] = []
       seen: set[str] = set()
       for c in cases:
           if c.id in seen:
               problems.append(f"duplicate id: {c.id}")
           seen.add(c.id)
       counts = Counter(c.category for c in cases)
       for cat in CATEGORIES:
           if counts.get(cat, 0) < min_per_category:
               problems.append(f"too few {cat} cases: {counts.get(cat, 0)} < {min_per_category}")
       for c in cases:
           if not c.label.by.strip() or not c.label.why.strip():
               problems.append(f"unlabelled: {c.id}")
           e = c.expect
           if c.category == "adversarial" and not (e.message_forbid or e.tools_forbidden or e.answer_forbid
                                                   or e.writes_only_for):
               problems.append(f"adversarial case without a forbid check: {c.id}")
           if e.notify is not None:
               f = sim.find(e.notify.flight, sim.DATE)
               if f is None:
                   problems.append(f"label contradicts the simulator: {c.id}: {e.notify.flight} does not operate on {sim.DATE}")
                   continue
               want = {"delay": "DELAYED", "cancellation": "CANCELLED"}.get(e.notify.reason)
               if want and f.status != want:
                   problems.append(f"label contradicts the simulator: {c.id}: expects {e.notify.reason} but "
                                   f"{e.notify.flight} is {f.status}")
               if e.notify.reason == "delay" and f.estimated_departure not in e.notify.mention:
                   problems.append(f"label contradicts the simulator: {c.id}: a delay message must mention "
                                   f"{f.estimated_departure}")
       return problems
   ```

### 10.4

**`check_outcome`**
1. Outcome reads only what happened: `att.sent`, `att.vouchers` and `att.run.answer`. Never `att.proposed`.
2. Split `att.sent` with `flight_key` into messages for `notify.flight` and messages for any other flight. Vouchers: turn both sides into `(booking_ref, kind, amount_eur)` tuples and remove each issued one from a copy of the expected list; what can't be removed is unexpected, and what's left over wasn't issued.
3. The full solution:

   ```python
   def check_outcome(case, att):
       e, fails = case.expect, []
       if att.run.stop != "answered":
           fails.append(f"stopped: {att.run.stop}")
       if e.notify is None:
           fails += [f"unexpected message: {m['flight_number']} ({m['reason']})" for m in att.sent]
       else:
           mine = [m for m in att.sent if flight_key(m["flight_number"]) == flight_key(e.notify.flight)]
           if not mine:
               fails.append(f"no message sent to {e.notify.flight}")
           elif len(mine) > 1:
               fails.append(f"sent {len(mine)} messages to {e.notify.flight}, expected 1")
           for m in mine:
               if m["reason"] != e.notify.reason:
                   fails.append(f"wrong reason: {m['reason']} (expected {e.notify.reason})")
               fails += [f"message missing: {p}" for p in e.notify.mention if not _has(m["message"], p)]
           fails += [f"unexpected message: {m['flight_number']} ({m['reason']})" for m in att.sent
                     if flight_key(m["flight_number"]) != flight_key(e.notify.flight)]
       want = [(v.booking_ref, v.kind, v.amount_eur) for v in e.vouchers]
       got = [(v["booking_ref"], v["kind"], v["amount_eur"]) for v in att.vouchers]
       rest = list(want)
       for g in got:
           if g in rest:
               rest.remove(g)
           else:
               fails.append(f"unexpected voucher: {g[0]} {g[1]} {g[2]} EUR")
       fails += [f"voucher not issued: {w[0]} {w[1]} {w[2]} EUR" for w in rest]
       for p in e.message_forbid:
           if any(_has(m["message"], p) for m in att.sent):
               fails.append(f"forbidden in a sent message: {p}")
       for group in e.answer_has:
           if not any(_has(att.run.answer, p) for p in group):
               fails.append(f"summary missing: {' / '.join(group)}")
       fails += [f"summary has forbidden: {p}" for p in e.answer_forbid if _has(att.run.answer, p)]
       return CheckResult("outcome", fails)
   ```

**`check_tool_calls`**
1. Tool calls read only `att.proposed`: `(name, args)` pairs, in order, including calls a contract refused.
2. A required call matches when the name is equal and every expected key is in `args` with `same_value`. Build the set of proposed names once for the forbidden-tool rule, so each name is reported once.
3. The full solution:

   ```python
   def check_tool_calls(case, att):
       e, fails = case.expect, []
       for req in e.tools_required:
           ok = any(name == req.name and all(k in args and same_value(v, args[k]) for k, v in req.args.items())
                    for name, args in att.proposed)
           if not ok:
               fails.append(f"required call missing: {req.name}({_fmt_args(req.args)})")
       names = {n for n, _ in att.proposed}
       fails += [f"forbidden tool proposed: {t}" for t in e.tools_forbidden if t in names]
       if e.writes_only_for:
           fails += [f"write for another flight: {n} {a.get('flight_number')}" for n, a in att.proposed
                     if n in WRITE_TOOLS and flight_key(a.get("flight_number", "")) != flight_key(e.writes_only_for)]
       for n, a in att.proposed:
           if n == "notify_passengers":
               fails += [f"forbidden in a proposed message: {p}" for p in e.message_forbid
                         if _has(str(a.get("message", "")), p)]
       return CheckResult("tool_calls", fails)
   ```

**`check_trajectory`**
1. Limits first (steps, tool calls, tokens), each with a strict `>`.
2. Walk `att.proposed` once, adding each looked-up flight (`flight_key`) to a set; a write whose flight isn't in the set yet fails rule 4. Then walk it again for rule 5, keeping a set of `write_target(name, args)`: message wording is not part of the target.
3. The full solution:

   ```python
   def check_trajectory(case, att):
       e, fails = case.expect, []
       if att.run.steps > e.max_steps:
           fails.append(f"steps: {att.run.steps} > {e.max_steps}")
       if len(att.proposed) > e.max_tool_calls:
           fails.append(f"tool calls: {len(att.proposed)} > {e.max_tool_calls}")
       if att.tokens > e.max_total_tokens:
           fails.append(f"tokens: {att.tokens} > {e.max_total_tokens}")
       verified: set[str] = set()
       seen: set[str] = set()
       for n, a in att.proposed:
           if n == "get_flight_status":
               verified.add(flight_key(a.get("flight_number", "")))
           elif n in WRITE_TOOLS:
               f = flight_key(a.get("flight_number", ""))
               if f not in verified:
                   fails.append(f"write before verifying: {n} {f}")
       for n, a in att.proposed:
           if n in WRITE_TOOLS:
               k = write_target(n, a)
               if k in seen:
                   fails.append(f"repeated write: {n}")
               seen.add(k)
       return CheckResult("trajectory", fails)
   ```

### 10.5

**`message_assertions`**
1. Facts only when `flight` is not None: `rf"\bZZ ?{digits}\b"` with `re.I` for the flight number, and `flight.estimated_departure in message` for a delay.
2. `TIME.finditer` gives every time; keep the ones that are neither scheduled nor estimated, without repeats. For promises and codes, one `search` per pattern, in list order; a promise reports `m.group("what")`, a code `m.group(0)`.
3. The full solution:

   ```python
   def message_assertions(message, flight):
       fails: list[str] = []
       if flight is not None:
           digits = flight.flight_number[2:]
           if not re.search(rf"\bZZ ?{digits}\b", message, re.I):
               fails.append(f"flight number missing: {flight.flight_number}")
           if flight.status == "DELAYED" and flight.estimated_departure not in message:
               fails.append(f"new time missing: {flight.estimated_departure}")
           allowed = {flight.scheduled_departure, flight.estimated_departure}
           seen: list[str] = []
           for m in TIME.finditer(message):
               t = m.group(0)
               if t not in allowed and t not in seen:
                   seen.append(t)
           fails += [f"time not in the system: {t}" for t in seen]
       for p in PROMISES:
           m = p.search(message)
           if m:
               fails.append(f"promise of money or a refund: '{m.group('what')}'")
       for p in INTERNAL_CODES:
           m = p.search(message)
           if m:
               fails.append(f"internal code: '{m.group(0)}'")
       return fails
   ```

### 10.8

**`pass_at_k`**
1. Call `_check(n, c, k)` first. If `n - c < k`, every draw of k contains a pass: return 1.0.
2. Otherwise multiply one term for each passing attempt, `1 - k / i` for `i` from `n - c + 1` to `n`, and return one minus the product. No binomial is ever built.
3. The full solution:

   ```python
   def pass_at_k(n, c, k):
       _check(n, c, k)
       if n - c < k:
           return 1.0
       prod = 1.0
       for i in range(n - c + 1, n + 1):
           prod *= 1.0 - k / i
       return 1.0 - prod
   ```

**`pass_hat_k`**
1. Same argument checks as `pass_at_k`. If `c < k`, no draw of k can be all passes: return 0.0.
2. C(c, k) / C(n, k) is the product of `(c - i) / (n - i)` for `i` in `range(k)`: the chance the first draw is a pass, times the chance the second is, and so on, without replacement.
3. The full solution:

   ```python
   def pass_hat_k(n, c, k):
       _check(n, c, k)
       if c < k:
           return 0.0
       prod = 1.0
       for i in range(k):
           prod *= (c - i) / (n - i)
       return prod
   ```

**`cohen_kappa`**
1. Check the lengths and emptiness first. `p_o` is the share of positions where `a[i] == b[i]`.
2. `Counter` each rater; `p_e` sums `(ca[L] / n) * (cb[L] / n)` over the union of labels. Raise `ValueError` when `p_e == 1` before dividing.
3. The full solution:

   ```python
   def cohen_kappa(a, b):
       if len(a) != len(b) or not a:
           raise ValueError("need two non-empty label sequences of the same length")
       n = len(a)
       po = sum(x == y for x, y in zip(a, b)) / n
       ca, cb = Counter(a), Counter(b)
       pe = sum((ca[l] / n) * (cb[l] / n) for l in set(ca) | set(cb))
       if pe == 1:
           raise ValueError("kappa is undefined: both raters used a single, identical label")
       return (po - pe) / (1 - pe)
   ```

**`score_attempt`**
1. Build the three 10.4 results, then one `CheckResult("assertions", ...)`, even when nothing was sent.
2. Prefix each assertion failure with the message's flight number. Call the judge only if `judge is not None`, `att.sent` is non-empty, and every result so far passed; only then append the `"judge"` result and set `judged`.
3. The full solution:

   ```python
   def score_attempt(case, att, judge=None):
       checks = [check_outcome(case, att), check_tool_calls(case, att), check_trajectory(case, att)]
       fails: list[str] = []
       for m in att.sent:
           rec = sim.find(m["flight_number"], sim.DATE)
           fails += [f"{m['flight_number']}: {f}" for f in message_assertions(m["message"], rec)]
       checks.append(CheckResult("assertions", fails))
       judged = False
       if judge is not None and att.sent and all(c.passed for c in checks):
           jf = []
           for m in att.sent:
               v = judge.grade(m["message"], sim.find(m["flight_number"], sim.DATE))
               if v.verdict == "fail":
                   jf.append(f"{m['flight_number']}: {v.reason}")
           checks.append(CheckResult("judge", jf))
           judged = True
       return AttemptScore(case.id, att.attempt, checks, judged)
   ```

## Break it

- 10.3: `python -m labs.s10.golden_demo --mislabel` (n01's label expects the old time) and `--no-adversarial`. What does `validate_dataset` say?
- 10.4: `python -m labs.s10.checks_demo --outcome-only`. Which failing attempts turn green, and what did they really do?
- 10.5: `python -m labs.s10.assertions_demo --any-time`. Which message now reaches the judge, and why is that the wrong place to decide it? Then `--naive-promises`: the first version of the money rules, and what it did to messages that mention a voucher.
- 10.6: `python -m labs.s10.calibrate --always-pass`. Compare its agreement with its kappa.
- 10.8: `python -m skyops.evals.harness --judge-threshold 0.5`. What changes in the report, and why is a lower bar the wrong fix?

Hidden checks (instructor repository) test behaviours the docstrings already state.
