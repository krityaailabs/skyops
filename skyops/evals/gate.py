"""Lectures 11.1, 11.2 and 11.6: the release gate. Run the eval suites, compare them with the stored baseline, and
fail the build on a regression, with a diff that says which category and which cases.

    python -m skyops.evals.gate                        # the deployed handler (config/handler.toml), baseline at HEAD
    python -m skyops.evals.gate --handler v3           # judge a candidate version
    python -m skyops.evals.gate --baseline-ref origin/main   # what CI does: baseline, thresholds and quarantine
                                                             # list from the TARGET branch
    python -m skyops.evals.gate --write-baseline       # record a new baseline (commit it in its own reviewed change)
Exit status: 0 green, 1 red (a regression), 2 the gate couldn't run (for example, no recording for a new prompt).

The rules, in config/gate.toml, per suite and per category:
- floor:  pass^k must not fall below the category's min_pass_hat_k (pass^k, because an agent that acts on
          passengers alone is only as good as its every-attempt reliability);
- drop:   the pass rate must not fall more than max_drop below the baseline's, unless the fall isn't
          statistically significant (a one-sided two-proportion test at alpha), which is then a warning;
- broken: a case that passed at least once in the baseline must not now fail every attempt.
Plus: the dataset must be the one the baseline was recorded on (same sha256), a judge may only count if its own
kappa cleared its threshold, quarantine entries must be complete, unexpired, capped per category and never for an
adversarial or held-out case, no category may be quarantined away entirely, and no eval item may appear in the
handler's prompt, the retrieval corpus or training data (contamination).
The baseline, the thresholds and the quarantine list are all read from the base ref, never the working tree. The
pull request still controls the gate's code, recordings and workflow file, so those need code owners
(.github/CODEOWNERS) and the job must be a required check.
Quarantined cases still run and appear in the diff; they just don't count. The held-out suite's per-case diff
never goes to the CI log: in a real repository its cases and recordings live where prompt authors can't read them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..model.transport import ReplayMiss
from ..settings import ROOT
from .stats import drop_p, pass_hat_k, wilson_interval
from skyops.model.errors import LabNotDone

GATE_CONFIG = ROOT / "config/gate.toml"
BASELINE = ROOT / "data/evals/baseline.json"
QUARANTINE = ROOT / "data/evals/quarantine.json"
OUT = ROOT / "runs/s11"
PRIVATE_SUITES = ("heldout",)
CATEGORIES = ("normal", "edge", "adversarial")
QUARANTINE_FIELDS = ("owner", "reason", "ticket", "added", "expires")


@dataclass(frozen=True)
class Rule:
    min_pass_hat_k: float
    max_drop: float


def load_config(path: Path = GATE_CONFIG, text: str | None = None) -> dict[str, Any]:
    raw = tomllib.loads(text if text is not None else Path(path).read_text())
    g = raw["gate"]
    return {"suites": list(g["suites"]), "k": int(g["k"]), "alpha": float(g["alpha"]),
            "max_quarantine_days": int(g["max_quarantine_days"]),
            "contamination_threshold": float(g["contamination_threshold"]),
            "rules": {c: Rule(float(r["min_pass_hat_k"]), float(r["max_drop"])) for c, r in g["category"].items()}}


# ---------- baselines ----------

def case_counts(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per case: its category, attempts and passes. A baseline stores only these, so category numbers can be
    recomputed for any subset (for example, without the quarantined cases) on both sides."""
    return {r["id"]: {"category": r["category"], "n": r["n"], "passes": r["passes"]} for r in report["cases"]}


def make_baseline(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"note": "Written by `python -m skyops.evals.gate --write-baseline`; change it only in its own reviewed "
                    "pull request.",
            "suites": {s: {"dataset_sha256": r["meta"]["dataset_sha256"], "handler": r["meta"]["handler"],
                           "attempts_per_case": r["meta"]["attempts_per_case"], "k": r["meta"]["k"],
                           "model_turns": r["meta"]["model_turns"], "cases": case_counts(r)}
                       for s, r in reports.items()}}


def file_at_ref(ref: str, path: Path) -> str:
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    out = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"gate: can't read {rel} at {ref}: {out.stderr.strip()}")
    return out.stdout


def baseline_from_ref(ref: str, path: Path = BASELINE) -> dict[str, Any]:
    """The baseline as committed on `ref` (CI passes the pull request's target branch). Reading it from git, not
    from the working tree, means a pull request can't lower the bar it is judged against by editing the file.
    The thresholds (config/gate.toml) are read from the same ref, for the same reason."""
    return json.loads(file_at_ref(ref, path))


def category_numbers(cases: dict[str, dict[str, Any]], k: int) -> dict[str, dict[str, Any]]:
    out = {}
    for cat in CATEGORIES:
        rs = [v for v in cases.values() if v["category"] == cat]
        if not rs:
            continue
        n, c = sum(r["n"] for r in rs), sum(r["passes"] for r in rs)
        lo, hi = wilson_interval(c, n)
        out[cat] = {"cases": len(rs), "attempts": n, "passes": c, "pass_rate": round(c / n, 4),
                    "pass_hat_k": round(sum(pass_hat_k(r["n"], r["passes"], k) for r in rs) / len(rs), 4),
                    "wilson_95": [round(lo, 4), round(hi, 4)]}
    return out


# ---------- 11.1: one category, and the case diff ----------

def fmt_p(p: float) -> str:
    """A p-value for people: "p = 0.009", or "p < 0.001" when it rounds to zero."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def compare_category(category: str, base: list[tuple[int, int]], cand: list[tuple[int, int]], rule: Rule,
                     k: int = 3, alpha: float = 0.05) -> tuple[list[str], list[str]]:
    """Lecture 11.1: judge one category. `base` and `cand` are (n, passes) per case. Return (failures, warnings).

    1. Floor: the candidate's pass^k is the mean over its cases of pass_hat_k(n, passes, k). If
       round(pass^k, 4) < rule.min_pass_hat_k, fail with
       "<category>: pass^<k> <pass^k:.2f> is below the floor <min_pass_hat_k:.2f>".
    2. Drop: pool each side (rate = total passes / total attempts) and let drop = round(base rate - cand rate, 4).
       If drop > rule.max_drop, get (p, method) = drop_p(base passes, base attempts, cand passes, cand attempts)
       (the z-test when every expected cell is at least ten, else Fisher's exact test) and build the message
       "<category>: pass rate fell from <base:.0%> to <cand:.0%> (<drop*100:.0f> points, more than
       <max_drop*100:.0f>; <fmt_p(p)>)", with ", exact" after the p-value when method is "exact". If p < alpha it
       is a failure; otherwise it is a warning, with ", not significant: add cases before trusting it" appended.
    Both checks always run, floor first. An empty `cand` or `base` raises ValueError: a category with no cases
    can't be judged, and silently passing it would hide a deleted category.
    Attempts of one case are not independent, so either p-value is optimistic; with a few cases per category the
    floor does most of the work.
    """
    raise LabNotDone("Lecture 11.1: implement compare_category(). Hints: labs/s11/README.md, lecture 11.1.")


def case_diff(base: dict[str, dict[str, Any]], cand: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Lecture 11.1: the cases whose result changed, for the red build's diff.

    `base` and `cand` map case id -> {"category", "n", "passes"}. Walk the candidate's ids in their order, then
    the ids only in the baseline, sorted. Emit a row {"id", "category", "base", "cand", "change"} where base and
    cand are "<passes>/<n>" (or "-" when the case is missing on that side) and change is:
    "new" (only in the candidate), "removed" (only in the baseline), "worse" or "better" (comparing
    passes / n exactly, by cross-multiplying, never with floats). Cases with an equal rate are left out.
    The category comes from the candidate, or from the baseline for a removed case.
    """
    raise LabNotDone("Lecture 11.1: implement case_diff(). Hints: labs/s11/README.md, lecture 11.1.")


# ---------- 11.2: quarantine ----------

def load_quarantine(path: Path = QUARANTINE) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text()) if Path(path).exists() else []


def quarantine_problems(entries: list[dict[str, Any]], today: str, known: dict[str, dict[str, str]],
                        max_days: int = 30, max_renewals: int = 2) -> tuple[set[str], list[str]]:
    """Lecture 11.2: which quarantine entries are in force, and what is wrong with the rest.

    Each entry has "case", "suite", "owner", "reason", "ticket", "added" and "expires" (dates as YYYY-MM-DD), and
    optionally "renewals" (an int, default 0). `known` maps suite -> {case id: category}. For each entry, in
    order, report only the FIRST problem that applies, checking in this order:
    1. a field in QUARANTINE_FIELDS order that is missing or blank: "quarantine <case>: missing <field>";
    2. the suite is in PRIVATE_SUITES: "quarantine: a held-out case can't be quarantined (id in the restricted
       report)" (never name a held-out case, and never exempt one);
    3. the case isn't in known[suite]: "quarantine <case>: no such case in <suite>";
    4. the case is adversarial: "quarantine <case>: adversarial cases can't be quarantined" (an attack only has
       to work once, so its case is never exempt);
    5. expires before added: "quarantine <case>: expires before it was added"; more than max_days after it:
       "quarantine <case>: <days> days is longer than <max_days>";
    6. renewals > max_renewals: "quarantine <case>: renewed <renewals> times, more than <max_renewals>";
    7. today is after expires: "quarantine <case>: expired on <expires> (owner <owner>): fix the case or renew it
       in a reviewed change". On the expiry date itself the entry is still in force;
    8. too many in one category: in each (suite, category), at most max(1, cases in that category // 10) entries
       may be in force; each entry beyond that (in entry order) adds "quarantine <case>: <category> already has
       <limit> quarantined case(s), the limit".
    An entry dated after today (added > today) is not in force yet and reports nothing. Every other entry with no
    problem is in force: add "<suite>/<case>" to the returned set. Return (in_force, problems).
    """
    raise LabNotDone("Lecture 11.2: implement quarantine_problems(). Hints: labs/s11/README.md, lecture 11.2.")


# ---------- 11.6: the whole gate ----------

@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diff: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    numbers: dict[str, dict[str, Any]] = field(default_factory=dict)
    quarantined: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)       # filled by main(): what was compared with what


def evaluate_gate(reports: dict[str, dict[str, Any]], baseline: dict[str, Any], config: dict[str, Any],
                  quarantine: list[dict[str, Any]], today: str, contamination: list = ()) -> GateResult:
    """Lecture 11.6: the gate's decision. Collect every failure and warning, in this order:

    1. Each contamination hit (objects with item, kind, source, share) whose kind is not "held-out notice", in
       order: "contamination: <item> (<kind>) found in <source> (<share:.0%> of its n-grams)". Then, if any hit is
       a held-out notice, ONE failure that names no case:
       "heldout: <count> contamination hit(s) (ids in the restricted report)".
    2. Quarantine: known = {suite: {case id: category}} from every report's "cases"; call
       quarantine_problems(quarantine, today, known, config["max_quarantine_days"]); its problems are failures.
       Keep the in-force set, sorted, as result.quarantined.
    3. For each suite in config["suites"], in order:
       a. no report: "no report for suite <s>", next suite; no baseline entry (baseline["suites"]):
          "no baseline for suite <s>: write one in a reviewed change", next suite;
       b. a judge counted without clearing its bar (report["judge"]["used"] true and kappa_judged below its
          threshold): "<s>: the judge counted without calibration";
       c. report meta dataset_sha256 differs from the baseline's:
          "<s>: dataset changed (baseline <old>, now <new>): re-baseline in its own reviewed change", next suite;
       d. base = the baseline's cases, cand = case_counts(report); drop the in-force quarantined ids from both;
          for each category in config["rules"] order that the FULL baseline has: if no case of it is left on
          either side, fail with "<s> <category>: every case is quarantined" (a category must never vanish);
          otherwise compare_category(category, [(n, passes)...] from base, the same from cand, the rule,
          config["k"], config["alpha"]); prefix every failure and warning with "<s> ";
       e. broken cases, in cand order, among the cases kept in d: base passes > 0 and cand passes == 0:
          "<s> <id>: now fails every attempt (<bp>/<bn> -> 0/<cn>)". For a suite in PRIVATE_SUITES, never name
          the cases: one failure, "<s>: <count> case(s) now fail every attempt (ids in the restricted report)";
       f. result.diff[s] = case_diff(all base cases, all cand cases), each row with "quarantined": True or False;
          result.numbers[s] = {"baseline": category_numbers(kept base, k), "candidate": category_numbers(kept
          cand, k)}.
    passed is True only when there are no failures; warnings never fail the gate.
    """
    raise LabNotDone("Lecture 11.6: implement evaluate_gate(). Hints: labs/s11/README.md, lecture 11.6.")


# ---------- running it ----------

ZERO_SHA = "0" * 40


def resolve_ref(ref: str) -> str:
    """A push that creates a branch has no previous commit (GitHub sends forty zeros): fall back to HEAD."""
    return "HEAD" if not ref or ref == ZERO_SHA else ref


def gate_today(explicit: str | None = None, ref: str = "HEAD") -> str:
    """The date quarantine expiry is judged on. A CI decision must be reproducible: rerunning the same commit must
    give the same answer, so the default is a commit's date, not the wall clock. It is the BASE commit's date (the
    target branch, already reviewed), because the change under test can set its own commit's date."""
    if explicit:
        return explicit
    if os.environ.get("SKYOPS_TODAY"):
        return os.environ["SKYOPS_TODAY"]
    out = subprocess.run(["git", "log", "-1", "--format=%cs", resolve_ref(ref)], cwd=ROOT, capture_output=True,
                         text=True)
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else date.today().isoformat()


def run_reports(version: str, suites: list[str], k: int) -> dict[str, dict[str, Any]]:
    from .harness import run_suite
    from .judge import Judge
    return {s: run_suite(suite=s, version=version, k=k, judge=Judge()) for s in suites}


def redact(text: str, suite: str) -> str:
    """Anything about a private suite that would name its cases (a fixture path, a case id) never reaches the log."""
    return f"{suite}: details in the restricted report" if suite in PRIVATE_SUITES else text


def render(res: GateResult, version: str, base_handler: str, ref: str, today: str) -> list[str]:
    lines = [f"eval gate: handler {version} against the baseline at {ref} ({base_handler}), judged on {today}"]
    for q in res.quarantined:
        suite = q.split("/", 1)[0]
        lines.append(f"quarantined (run and reported, not counted): {redact(q, suite)}")
    for s, nums in res.numbers.items():
        for cat, c in nums["candidate"].items():
            b = nums["baseline"].get(cat)
            if b:
                lines.append(f"  {s:8} {cat:12} pass rate {b['pass_rate']:.0%} -> {c['pass_rate']:.0%} "
                             f"(95% {c['wilson_95'][0]:.0%}-{c['wilson_95'][1]:.0%})   pass^k {b['pass_hat_k']:.2f} -> "
                             f"{c['pass_hat_k']:.2f}")
    lines += [f"FAIL {f}" for f in res.failures] + [f"WARN {w}" for w in res.warnings]
    for s, rows in res.diff.items():
        if not rows:
            continue
        if s in PRIVATE_SUITES:
            worse = sum(r["change"] == "worse" for r in rows)
            lines.append(f"{s}: {len(rows)} case(s) changed, {worse} worse; per-case results stay in the restricted "
                         "report")
            continue
        lines.append(f"changed cases ({s}):")
        lines += [f"  {r['id']:26} {r['category']:12} {r['base']:>4} -> {r['cand']:<4} {r['change']}"
                  + ("  (quarantined)" if r["quarantined"] else "") for r in rows]
    return lines


def write_outputs(res: GateResult, lines: list[str], out: Path = OUT) -> tuple[Path, Path]:
    """gate.json (for machines; never a private suite's per-case rows or ids) and gate.md (the job summary)."""
    out.mkdir(parents=True, exist_ok=True)
    pj, pm = out / "gate.json", out / "gate.md"
    public = {**res.__dict__, "diff": {s: rows for s, rows in res.diff.items() if s not in PRIVATE_SUITES},
              "quarantined": [q for q in res.quarantined if q.split("/", 1)[0] not in PRIVATE_SUITES]}
    pj.write_text(json.dumps(public, indent=1, sort_keys=True) + "\n")
    head = "## Eval gate: " + ("green" if res.passed else "RED")
    pm.write_text(head + "\n\n```\n" + "\n".join(lines) + "\n```\n")
    return pj, pm


def shown(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    import logging
    logging.disable(logging.CRITICAL)
    from .contamination import eval_items, find_contamination, heldout_vs_golden, model_inputs, shared_flights
    from .runner import deployed_version
    ap = argparse.ArgumentParser()
    ap.add_argument("--handler", default=None, help="candidate version (default: config/handler.toml)")
    ap.add_argument("--baseline-ref", default="HEAD",
                    help="git ref to read the baseline, thresholds and quarantine list from (CI: the target branch)")
    ap.add_argument("--baseline", default=None, help="read the baseline from this file instead (experiments only)")
    ap.add_argument("--suite", action="append", default=None, help="override the suites in config/gate.toml")
    ap.add_argument("--today", default=None,
                    help="date for quarantine expiry (default: SKYOPS_TODAY, else the base commit's date)")
    ap.add_argument("--no-contamination", action="store_true", help="skip the contamination step")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="where gate.json and gate.md go")
    a = ap.parse_args(argv)
    ref = resolve_ref(a.baseline_ref)
    if a.baseline:
        cfg, quarantine = load_config(), load_quarantine()
    else:
        cfg = load_config(text=file_at_ref(ref, GATE_CONFIG))
        quarantine = json.loads(file_at_ref(ref, QUARANTINE)) if not a.write_baseline else []
    if a.suite:
        cfg["suites"] = a.suite
    version = a.handler or deployed_version()
    today = gate_today(a.today, ref)
    try:
        reports = {}
        for s in cfg["suites"]:
            reports.update(run_reports(version, [s], cfg["k"]))
    except ReplayMiss as e:
        why = redact(str(e).splitlines()[0], s)
        res = GateResult(passed=False, failures=[f"can't run handler {version} on suite {s}: {why}"],
                         meta={"candidate_handler": version, "baseline_ref": ref, "today": today})
        lines = [f"gate: can't run handler {version}: {why}",
                 "a prompt change needs its own recordings: re-record (SKYOPS_MODE=record) and commit them with it"]
        write_outputs(res, lines, Path(a.out))
        for l in lines:
            print(l)
        return 2
    if a.write_baseline:
        BASELINE.write_text(json.dumps(make_baseline(reports), indent=1, sort_keys=True) + "\n")
        print(f"wrote {BASELINE.relative_to(ROOT)} for handler {version}: commit it in its own reviewed change")
        return 0
    if a.baseline:
        baseline, ref = json.loads(Path(a.baseline).read_text()), a.baseline
    else:
        baseline = baseline_from_ref(ref)
    hits = [] if a.no_contamination else (
        find_contamination(eval_items(), model_inputs(version), threshold=cfg["contamination_threshold"])
        + heldout_vs_golden() + shared_flights())
    res = evaluate_gate(reports, baseline, cfg, quarantine, today, hits)
    base_handler = ", ".join(sorted({v["handler"].split()[-1] for v in baseline["suites"].values()}))
    res.meta = {"candidate_handler": version, "baseline_ref": ref, "baseline_handler": base_handler, "today": today,
                "suites": {s: {"dataset_sha256": r["meta"]["dataset_sha256"], "model_turns": r["meta"]["model_turns"],
                               "attempts_per_case": r["meta"]["attempts_per_case"], "k": r["meta"]["k"],
                               "judge_used": r["judge"].get("used", False),
                               "judge_kappa_judged": r["judge"].get("kappa_judged")}
                           for s, r in reports.items()}}
    lines = render(res, version, f"handler {base_handler}", ref, today)
    pj, pm = write_outputs(res, lines, Path(a.out))
    for l in lines:
        print(l)
    verdict = "GREEN" if res.passed else "RED"
    print(f"RESULT: {verdict} ({len(res.failures)} failure(s), {len(res.warnings)} warning(s)); wrote "
          f"{shown(pj)} and {shown(pm)}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
