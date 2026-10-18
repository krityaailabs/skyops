# Section 11 labs: evals in CI and in production

Start at tag `s11-start`. The Section 11 checks fail until you write the functions below; `make fitness` names the next one. Time: about three hours. Live tokens: none. Network: none.

Section 10's harness runs unchanged. Section 11 adds:

- **the gate** (`skyops/evals/gate.py`, `config/gate.toml`): each eval suite compared with the baseline committed on the target branch (`data/evals/baseline.json`), per category: a floor on pass^k, a maximum pass-rate drop that must also be significant, and no case that now fails every attempt. Plus quarantine (`data/evals/quarantine.json`), the dataset hash, the judge rule and contamination.
- **two candidate handlers** (`skyops/agent/disruption.py`): `v3` (shorter, and the disagreement rule lost) and `v4` (v3 plus worked examples copied from the golden set). `config/handler.toml` says which version is deployed (`v2`).
- **a held-out suite** (`data/evals/heldout.jsonl`, 12 cases from the review queue, rebuilt 2026-09-26, on flights the golden set never mentions; policy in `heldout.meta.json`).
- **online evals** (`skyops/evals/online.py`, `config/online.toml`) over a synthetic production log (`data/prod/traces.jsonl`, 1,680 traces).
- **contamination checks** (`skyops/evals/contamination.py`), and a proposed fine-tuning set with a planted leak (`labs/s11/data/sft_proposed.jsonl`).
- **the CI workflow** (`.github/workflows/evals.yml`) and `.github/CODEOWNERS`, checked offline by `tools/check_workflow.py` against GitHub's workflow schema (vendored from SchemaStore). `tools/set_handler.py v3` changes the deployed version portably.

| Lecture | File | Implement | Check |
|---|---|---|---|
| 11.1 | `skyops/evals/gate.py`, `skyops/evals/stats.py` | `compare_category`, `case_diff`, `two_proportion_test` | `test_l111_*` |
| 11.2 | `skyops/evals/stats.py`, `skyops/evals/gate.py` | `wilson_interval`, `quarantine_problems` | `test_l112_*` |
| 11.3 | `labs/s11/online_demo.py` | nothing: run it after 11.5 (it uses 11.5's `psi`), then design your own online eval | — |
| 11.4 | `labs/s11/contamination_demo.py` | nothing: run it after 11.5 (it uses `find_contamination`); find the leak by hand first | — |
| 11.5 | `skyops/evals/contamination.py`, `skyops/evals/online.py`, `labs/s11/incident.py` | `find_contamination`, `psi` | `test_l115_*` |
| 11.6 | `skyops/evals/gate.py`, `.github/workflows/evals.yml` | `evaluate_gate` | `test_l116_*` |

The demos `labs/s11/gate_demo.py` (11.1) and `labs/s11/flaky_demo.py` (11.2) run on the solved tag, or on yours once 11.6 and 11.2 are done.

Run things as you go: `make labs-s11` runs every demo; `make ci-gate` runs exactly what CI runs; `make workflow-check` validates the workflow; `make baseline` writes a new baseline (commit it in its own reviewed change).

About the replays: every model turn is scripted (`meta.synthetic = true`). `fixtures/s11/<suite>/<version>/<case>/a<n>.jsonl` hold the attempts for the v3 and v4 candidates and for the held-out suite; which attempts fail is written in `labs/s11/build_fixtures_s11.py`. The production log is synthetic too (the fault scenario is `skyops/sim/incident18.py`, written by `labs/s11/build_prod_log.py`, seeded): its failure rates are parameters, not measurements. The gate, the checks, the statistics and the scans are real code.

## Hint ladder

Three hints per function. Take one at a time; the third is the full solution. The solved tag `s11` has the same code.

### 11.1

**`compare_category`**
1. Two independent checks, floor first. pass^k is the MEAN over cases of `pass_hat_k(n, passes, k)`, never pass_hat_k of the pooled counts.
2. Pool each side for the drop (`sum(passes) / sum(n)`), round the drop to 4 places before comparing with `max_drop`, and only then call `drop_p(base passes, base n, cand passes, cand n)`, which picks the z-test or Fisher's exact test. Format the p-value with `fmt_p`, and add `, exact` for Fisher.
3. The full solution:

   ```python
   def compare_category(category: str, base: list[tuple[int, int]], cand: list[tuple[int, int]], rule: Rule,
                        k: int = 3, alpha: float = 0.05) -> tuple[list[str], list[str]]:
       if not base or not cand:
           raise ValueError(f"{category}: no cases to compare")
       fails, warns = [], []
       phk = sum(pass_hat_k(n, c, k) for n, c in cand) / len(cand)
       if round(phk, 4) < rule.min_pass_hat_k:
           fails.append(f"{category}: pass^{k} {phk:.2f} is below the floor {rule.min_pass_hat_k:.2f}")
       bn, bc = sum(n for n, _ in base), sum(c for _, c in base)
       cn, cc = sum(n for n, _ in cand), sum(c for _, c in cand)
       drop = round(bc / bn - cc / cn, 4)
       if drop > rule.max_drop:
           p, method = drop_p(bc, bn, cc, cn)
           msg = (f"{category}: pass rate fell from {bc / bn:.0%} to {cc / cn:.0%} ({drop * 100:.0f} points, more than "
                  f"{rule.max_drop * 100:.0f}; {fmt_p(p)}{', exact' if method == 'exact' else ''})")
           if p < alpha:
               fails.append(msg)
           else:
               warns.append(msg + ", not significant: add cases before trusting it")
       return fails, warns
   ```

**`case_diff`**
1. Two loops: the candidate's ids in their order, then `sorted(set(base) - set(cand))`.
2. Compare rates without floats: `c['passes'] * b['n']` against `b['passes'] * c['n']`. Equal products mean the same rate (2/4 and 1/2), so skip the case.
3. The full solution:

   ```python
   def case_diff(base: dict[str, dict[str, Any]], cand: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
       rows = []
       fmt = lambda v: f"{v['passes']}/{v['n']}"  # noqa: E731
       for cid, c in cand.items():
           b = base.get(cid)
           if b is None:
               rows.append({"id": cid, "category": c["category"], "base": "-", "cand": fmt(c), "change": "new"})
               continue
           lhs, rhs = c["passes"] * b["n"], b["passes"] * c["n"]
           if lhs != rhs:
               rows.append({"id": cid, "category": c["category"], "base": fmt(b), "cand": fmt(c),
                            "change": "worse" if lhs < rhs else "better"})
       for cid in sorted(set(base) - set(cand)):
           rows.append({"id": cid, "category": base[cid]["category"], "base": fmt(base[cid]), "cand": "-",
                        "change": "removed"})
       return rows
   ```

**`two_proportion_test`**
1. The pooled rate is `(c1 + c2) / (n1 + n2)`; if it is exactly 0 or 1, return `(0.0, 1.0)` before dividing by a zero standard error.
2. `z = (p1 - p2) / se` is positive when the second rate is lower; the one-sided p-value is `0.5 * math.erfc(z / math.sqrt(2))`.
3. The full solution:

   ```python
   def two_proportion_test(c1: int, n1: int, c2: int, n2: int) -> tuple[float, float]:
       if n1 < 1 or n2 < 1 or not (0 <= c1 <= n1 and 0 <= c2 <= n2):
           raise ValueError(f"need n >= 1 and 0 <= c <= n (got {c1}/{n1}, {c2}/{n2})")
       pooled = (c1 + c2) / (n1 + n2)
       if pooled in (0.0, 1.0):
           return 0.0, 1.0
       se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
       z = (c1 / n1 - c2 / n2) / se
       return z, 0.5 * math.erfc(z / math.sqrt(2))
   ```

### 11.2

**`quarantine_problems`**
1. One problem per entry: check in the docstring's order (held-out suite, unknown case, adversarial, span, renewals, expiry, then the per-category cap) and `continue` after the first one you report.
2. `date.fromisoformat` for every date; `(expires - added).days` for the span, negative meaning "expires before it was added". Only `today > expires` is expired; an entry added after today isn't in force yet. The cap is `max(1, cases in that category // 10)`, counted per `(suite, category)`. In-force ids are `"<suite>/<case>"`.
3. The full solution:

   ```python
   def quarantine_problems(entries: list[dict[str, Any]], today: str, known: dict[str, dict[str, str]],
                           max_days: int = 30, max_renewals: int = 2) -> tuple[set[str], list[str]]:
       active, problems = set(), []
       t = date.fromisoformat(today)
       per_cat: dict[tuple[str, str], int] = {}
       for e in entries:
           cid, suite = e.get("case", "?"), e.get("suite", "golden")
           missing = next((f for f in QUARANTINE_FIELDS if not str(e.get(f, "")).strip()), None)
           if missing:
               problems.append(f"quarantine {cid}: missing {missing}")
               continue
           if suite in PRIVATE_SUITES:
               problems.append("quarantine: a held-out case can't be quarantined (id in the restricted report)")
               continue
           if cid not in known.get(suite, {}):
               problems.append(f"quarantine {cid}: no such case in {suite}")
               continue
           cat = known[suite][cid]
           if cat == "adversarial":
               problems.append(f"quarantine {cid}: adversarial cases can't be quarantined")
               continue
           added, expires = date.fromisoformat(e["added"]), date.fromisoformat(e["expires"])
           days = (expires - added).days
           if days < 0:
               problems.append(f"quarantine {cid}: expires before it was added")
               continue
           if days > max_days:
               problems.append(f"quarantine {cid}: {days} days is longer than {max_days}")
               continue
           renewals = int(e.get("renewals", 0))
           if renewals > max_renewals:
               problems.append(f"quarantine {cid}: renewed {renewals} times, more than {max_renewals}")
               continue
           if t > expires:
               problems.append(f"quarantine {cid}: expired on {e['expires']} (owner {e['owner']}): fix the case or "
                               "renew it in a reviewed change")
               continue
           if added > t:
               continue
           limit = max(1, sum(1 for c in known[suite].values() if c == cat) // 10)
           if per_cat.get((suite, cat), 0) >= limit:
               problems.append(f"quarantine {cid}: {cat} already has {limit} quarantined case(s), the limit")
               continue
           per_cat[(suite, cat)] = per_cat.get((suite, cat), 0) + 1
           active.add(f"{suite}/{cid}")
       return active, problems
   ```

**`wilson_interval`**
1. Validate first: `n >= 1` and `0 <= c <= n`, else `ValueError`.
2. Compute `denom`, `centre` and `half` exactly as the docstring writes them, then clamp, and return exactly 0.0 or 1.0 at the ends when `c == 0` or `c == n`.
3. The full solution:

   ```python
   def wilson_interval(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
       if n < 1 or not 0 <= c <= n:
           raise ValueError(f"need n >= 1 and 0 <= c <= n (got c={c}, n={n})")
       p = c / n
       denom = 1 + z * z / n
       centre = (p + z * z / (2 * n)) / denom
       half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
       lo = 0.0 if c == 0 else max(0.0, centre - half)
       hi = 1.0 if c == n else min(1.0, centre + half)
       return lo, hi
   ```

### 11.5

**`find_contamination`**
1. Build each source's n-gram set once, as the union of `ngrams(tokens(text), n)` over its texts, so no n-gram spans two texts.
2. Short items (fewer than n tokens) can't have n-grams: look for their whole token list contiguously inside each text's tokens (`_contiguous`). Skip items with no tokens; round the share to 4 places.
3. The full solution:

   ```python
   def find_contamination(items: list[Item], sources: dict[str, list[str]], n: int = N,
                          threshold: float = THRESHOLD) -> list[Hit]:
       grams = {name: set().union(*(ngrams(tokens(t), n) for t in texts)) if texts else set()
                for name, texts in sources.items()}
       toks_by_source = {name: [tokens(t) for t in texts] for name, texts in sources.items()}
       hits = []
       for it in items:
           toks = tokens(it.text)
           if not toks:
               continue
           mine = ngrams(toks, n)
           for name in sources:
               if len(toks) >= n:
                   share = len(mine & grams[name]) / len(mine)
               else:
                   share = 1.0 if any(_contiguous(toks, t) for t in toks_by_source[name]) else 0.0
               if share >= threshold:
                   hits.append(Hit(it.id, it.kind, name, round(share, 4)))
       return hits
   ```

**`psi`**
1. Bins are `sorted(set(expected) | set(actual))`; a missing key is a count of 0.
2. Share = count / that side's total, and `share or eps` replaces an exact zero without renormalising. Sum `(a - e) * math.log(a / e)`.
3. The full solution:

   ```python
   def psi(expected: dict[str, int], actual: dict[str, int], eps: float = 1e-4) -> float:
       te, ta = sum(expected.values()), sum(actual.values())
       if te == 0 or ta == 0:
           raise ValueError("both distributions need at least one count")
       total = 0.0
       for b in sorted(set(expected) | set(actual)):
           e = expected.get(b, 0) / te or eps
           a = actual.get(b, 0) / ta or eps
           total += (a - e) * math.log(a / e)
       return total
   ```

### 11.6

**`evaluate_gate`**
1. Follow the docstring's numbered steps in order, and `continue` to the next suite where it says so (missing report, missing baseline, changed dataset).
2. Quarantined ids are removed from BOTH sides before the categories and the broken-case rule, but the diff uses all cases and marks them. A category left with no cases is a failure. For a private suite, contamination hits and broken cases are each one line with a count, never ids.
3. The full solution:

   ```python
   def evaluate_gate(reports: dict[str, dict[str, Any]], baseline: dict[str, Any], config: dict[str, Any],
                     quarantine: list[dict[str, Any]], today: str, contamination: list = ()) -> GateResult:
       res = GateResult(passed=False)
       private = [h for h in contamination if h.kind == "held-out notice"]
       for h in contamination:
           if h.kind != "held-out notice":
               res.failures.append(f"contamination: {h.item} ({h.kind}) found in {h.source} ({h.share:.0%} of its "
                                   "n-grams)")
       if private:
           res.failures.append(f"heldout: {len(private)} contamination hit(s) (ids in the restricted report)")
       known = {s: {c["id"]: c["category"] for c in r["cases"]} for s, r in reports.items()}
       active, problems = quarantine_problems(quarantine, today, known, config["max_quarantine_days"])
       res.failures += problems
       res.quarantined = sorted(active)
       k = config["k"]
       for s in config["suites"]:
           if s not in reports:
               res.failures.append(f"no report for suite {s}")
               continue
           if s not in baseline.get("suites", {}):
               res.failures.append(f"no baseline for suite {s}: write one in a reviewed change")
               continue
           rep, b = reports[s], baseline["suites"][s]
           j = rep.get("judge", {})
           if j.get("used") and j.get("kappa_judged", 0) < j.get("threshold", 1):
               res.failures.append(f"{s}: the judge counted without calibration")
           if rep["meta"]["dataset_sha256"] != b["dataset_sha256"]:
               res.failures.append(f"{s}: dataset changed (baseline {b['dataset_sha256']}, now "
                                   f"{rep['meta']['dataset_sha256']}): re-baseline in its own reviewed change")
               continue
           base_all, cand_all = b["cases"], case_counts(rep)
           keep = lambda d: {cid: v for cid, v in d.items() if f"{s}/{cid}" not in active}  # noqa: E731
           base, cand = keep(base_all), keep(cand_all)
           for cat, rule in config["rules"].items():
               if not any(v["category"] == cat for v in base_all.values()):
                   continue
               bl = [(v["n"], v["passes"]) for v in base.values() if v["category"] == cat]
               cl = [(v["n"], v["passes"]) for v in cand.values() if v["category"] == cat]
               if not bl or not cl:
                   res.failures.append(f"{s} {cat}: every case is quarantined")
                   continue
               f, w = compare_category(cat, bl, cl, rule, k, config["alpha"])
               res.failures += [f"{s} {x}" for x in f]
               res.warnings += [f"{s} {x}" for x in w]
           broken = [f"{s} {cid}: now fails every attempt ({base[cid]['passes']}/{base[cid]['n']} -> 0/{v['n']})"
                     for cid, v in cand.items() if cid in base and base[cid]["passes"] > 0 and v["passes"] == 0]
           if broken and s in PRIVATE_SUITES:
               broken = [f"{s}: {len(broken)} case(s) now fail every attempt (ids in the restricted report)"]
           res.failures += broken
           res.diff[s] = [{**row, "quarantined": f"{s}/{row['id']}" in active} for row in case_diff(base_all, cand_all)]
           res.numbers[s] = {"baseline": category_numbers(base, k), "candidate": category_numbers(cand, k)}
       res.passed = not res.failures
       return res
   ```
