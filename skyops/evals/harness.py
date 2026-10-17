"""Lecture 10.8: the eval harness. Dataset in, report out.

    python -m skyops.evals.harness                 # every golden case, N_ATTEMPTS attempts each, judge if calibrated
    python -m skyops.evals.harness --no-judge      # deterministic checks only
writes runs/s10/report.json (for machines, and Section 11's CI gate) and runs/s10/report.md (for people).

The order inside one attempt is the point of Section 10: deterministic checks first (outcome, tool calls,
trajectory, message assertions); the judge only for an attempt that passed all of them, and only if its
calibration against human labels cleared the threshold. Every number in the report comes from code that
runs offline: the model turns (the handler's and the judge's) are replays, and the report says so.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cost import per_thousand
from ..settings import ROOT, Settings
from ..sim import flights as sim
from .checks import CheckResult, check_outcome, check_tool_calls, check_trajectory, message_assertions
from .dataset import GOLDEN, GoldenCase, dataset_sha256, load_golden
from .judge import FIXTURE as JUDGE_FIXTURE, THRESHOLD, Calibration, Judge, calibrate
from .runner import N_ATTEMPTS, Attempt, attempt_files, run_case, turns_label
from .stats import pass_at_k, pass_hat_k

K = 3
OUT = ROOT / "runs/s10"
CHECK_ORDER = ("outcome", "tool_calls", "trajectory", "assertions", "judge")


@dataclass
class AttemptScore:
    case_id: str
    attempt: int
    checks: list[CheckResult] = field(default_factory=list)
    judged: bool = False

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[str]:
        return [f"{c.name}: {f}" for c in self.checks for f in c.failures]


def score_attempt(case: GoldenCase, att: Attempt, judge: Judge | None = None) -> AttemptScore:
    """Lecture 10.8: score one attempt, deterministic checks first, the judge last and only if needed.

    1. Run check_outcome, check_tool_calls and check_trajectory, in that order, and keep all three results.
    2. Message assertions: for each message in att.sent, in order, run
       message_assertions(m["message"], sim.find(m["flight_number"], sim.DATE)) and prefix each failure with
       "<flight_number>: ". Keep one CheckResult("assertions", failures), even when att.sent is empty.
    3. The judge runs ONLY when `judge` is not None, att.sent is not empty, and every result so far passed.
       Then grade each sent message with judge.grade(m["message"], <the same flight record>), and keep
       CheckResult("judge", ["<flight_number>: <verdict.reason>" for each "fail" verdict]); set judged=True.
       Otherwise there is no "judge" result and judged stays False: never call the judge to confirm a failure
       code already found, and never on an attempt that sent nothing.
    Return AttemptScore(case.id, att.attempt, checks, judged). Passing `judge=None` is how the harness runs an
    uncalibrated judge: not at all.
    """
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


def summarise(cases: list[GoldenCase], scores: dict[str, list[AttemptScore]], attempts: dict[str, list[Attempt]],
              k: int = K) -> dict[str, Any]:
    """Aggregate attempt scores into the report: per case, per category, per check, and overall."""
    rows = []
    for c in cases:
        sc = scores[c.id]
        n, passes = len(sc), sum(s.passed for s in sc)
        rows.append({"id": c.id, "category": c.category, "n": n, "passes": passes,
                     "pass_rate": round(passes / n, 4),
                     "pass_at_k": round(pass_at_k(n, passes, k), 4), "pass_hat_k": round(pass_hat_k(n, passes, k), 4),
                     "first_attempt_passed": sc[0].passed,
                     "failures": [{"attempt": s.attempt, "checks": s.failures} for s in sc if not s.passed]})

    def agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
        return {"cases": len(rs), "attempts": sum(r["n"] for r in rs),
                "pass_rate": round(sum(r["passes"] for r in rs) / sum(r["n"] for r in rs), 4),
                "pass_at_k": round(sum(r["pass_at_k"] for r in rs) / len(rs), 4),
                "pass_hat_k": round(sum(r["pass_hat_k"] for r in rs) / len(rs), 4),
                "first_attempt_green": sum(r["first_attempt_passed"] for r in rs),
                "every_attempt_green": sum(r["passes"] == r["n"] for r in rs)}

    failing = Counter()
    for sc in scores.values():
        for s in sc:
            for ch in s.checks:
                failing[ch.name] += not ch.passed
    all_atts = [a for v in attempts.values() for a in v]
    return {"overall": agg(rows),
            "by_category": {cat: agg([r for r in rows if r["category"] == cat])
                            for cat in ("normal", "edge", "adversarial") if any(r["category"] == cat for r in rows)},
            "attempts_failing_check": {name: failing.get(name, 0) for name in CHECK_ORDER},
            "judged_attempts": sum(s.judged for v in scores.values() for s in v),
            "agent_per_1000_attempts_usd": round(per_thousand(sum(a.cost_usd for a in all_atts) / len(all_atts)), 2),
            "agent_tokens": sum(a.tokens for a in all_atts),
            "cases": rows}


def run_suite(cases: list[GoldenCase] | None = None, n: int = N_ATTEMPTS, k: int = K, judge: Judge | None = None,
              calibration: Calibration | None = None, threshold: float = THRESHOLD) -> dict[str, Any]:
    """Run every case n times, score every attempt, and build the report. If a judge is given, it is calibrated
    first (or `calibration` is used), and its verdicts count only if ITS OWN kappa, on the held-out messages it
    actually grades, reaches `threshold`. The pipeline's kappa is reported beside it, never gated on: it includes
    verdicts code decided.
    Replayed, calibrating on every run is free. Live, it costs one judge call per labelled item, so in production
    store the calibration per judge model and prompt version, and pass it in."""
    cases = cases or load_golden()
    use = None
    cal_out: dict[str, Any] = {"used": False, "reason": "no judge"}
    if judge is not None:
        calibration = calibration or calibrate(judge)
        usable = calibration.kappa_judged >= threshold
        use = judge if usable else None
        own = (f"the judge's own kappa on the {calibration.judged_items} held-out messages it grades, "
               f"{calibration.kappa_judged:.2f},")
        cal_out = {"used": usable, "criterion": "passenger message: says clearly and calmly what happened "
                   "and what to do next", "model": judge.client.model, "threshold": threshold,
                   "kappa_judged": calibration.kappa_judged, "judged_items": calibration.judged_items,
                   "kappa_judge_alone": calibration.kappa_judge, "kappa_pipeline": calibration.kappa_pipeline,
                   "kappa_judged_ci": list(calibration.kappa_judged_ci), "labelled_items": calibration.items,
                   "reason": f"{own} meets the threshold {threshold}" if usable else
                   f"{own} is below the threshold {threshold}"}
        calls_before = judge.calls
    attempts = {c.id: run_case(c, n) for c in cases}
    scores = {c.id: [score_attempt(c, a, use) for a in attempts[c.id]] for c in cases}
    rep = summarise(cases, scores, attempts, k)
    rep["judge"] = cal_out
    if judge is not None:
        rep["judge"]["calls_in_suite"] = judge.calls - calls_before
    rep["meta"] = {"suite": "golden disruption cases", "dataset": str(GOLDEN.relative_to(ROOT)),
                   "dataset_sha256": dataset_sha256(), "simulator_date": sim.DATE, "attempts_per_case": n, "k": k,
                   "handler": "skyops.agent.disruption v2",
                   "model_turns": turns_label(attempt_files([c.id for c in cases], n)),
                   "judge_turns": (turns_label([JUDGE_FIXTURE], Settings.from_env(provider="openai").mode)
                                   if judge is not None else "none")}
    return rep


def render_markdown(rep: dict[str, Any]) -> str:
    m, o, k = rep["meta"], rep["overall"], rep["meta"]["k"]
    j = rep["judge"]
    lines = [f"# Eval report: {m['suite']}", "",
             f"Dataset `{m['dataset']}` (sha256 {m['dataset_sha256']}), {o['cases']} cases, "
             f"{m['attempts_per_case']} attempts each, k = {k}. Simulator date {m['simulator_date']}.",
             f"Model turns: {m['model_turns']}. Judge turns: {m['judge_turns']}.", "",
             "| | cases | pass rate | pass@k | pass^k | first attempt green | every attempt green |",
             "|---|---|---|---|---|---|---|"]
    for name, a in [("all", o)] + list(rep["by_category"].items()):
        lines.append(f"| {name} | {a['cases']} | {a['pass_rate']:.0%} | {a['pass_at_k']:.0%} | {a['pass_hat_k']:.0%} | "
                     f"{a['first_attempt_green']} | {a['every_attempt_green']} |")
    detail = (f"; pipeline kappa {j['kappa_pipeline']:.2f}, judge alone {j['kappa_judge_alone']:.2f}"
              if "kappa_pipeline" in j else "")
    lines += ["", "Judge: " + (f"used ({j['reason']}{detail}; {j.get('calls_in_suite', 0)} calls)" if j["used"]
                               else f"not used ({j['reason']}{detail})"),
              "", "Attempts failing each check: " + ", ".join(f"{n} {c}" for n, c in rep["attempts_failing_check"].items()),
              f"Agent cost: {rep['agent_per_1000_attempts_usd']:.2f} USD per 1,000 attempts (example prices).", "",
              "| case | category | passes | pass@k | pass^k | failing attempts |", "|---|---|---|---|---|---|"]
    for r in rep["cases"]:
        fa = "; ".join(f"a{f['attempt']}: {', '.join(f['checks'])}" for f in r["failures"]) or "-"
        lines.append(f"| {r['id']} | {r['category']} | {r['passes']}/{r['n']} | {r['pass_at_k']:.2f} | "
                     f"{r['pass_hat_k']:.2f} | {fa} |")
    return "\n".join(lines) + "\n"


def write_report(rep: dict[str, Any], out: Path = OUT) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    pj, pm = out / "report.json", out / "report.md"
    pj.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    pm.write_text(render_markdown(rep))
    return pj, pm


def main(argv: list[str] | None = None) -> None:
    import logging
    logging.disable(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_ATTEMPTS)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.k < 1 or a.n < a.k:
        ap.error(f"--n {a.n} must be at least --k {a.k}: pass@k and pass^k draw k of the n attempts")
    if a.n > N_ATTEMPTS and Settings.from_env().mode == "replay":
        ap.error(f"--n {a.n}: only {N_ATTEMPTS} attempts per case are recorded; re-record to replay more")
    rep = run_suite(n=a.n, k=a.k, judge=None if a.no_judge else Judge(), threshold=a.judge_threshold)
    pj, pm = write_report(rep, Path(a.out))
    o, j = rep["overall"], rep["judge"]
    print(f"{o['cases']} cases x {rep['meta']['attempts_per_case']} attempts = {o['attempts']} attempts; "
          f"model turns: {rep['meta']['model_turns']}")
    print(f"pass rate {o['pass_rate']:.0%}   pass@{a.k} {o['pass_at_k']:.0%}   pass^{a.k} {o['pass_hat_k']:.0%}")
    for cat, c in rep["by_category"].items():
        print(f"  {cat:12} pass rate {c['pass_rate']:.0%}   pass^{a.k} {c['pass_hat_k']:.0%}   "
              f"every attempt green {c['every_attempt_green']}/{c['cases']}")
    print("attempts failing each check: " + ", ".join(f"{n} {c}" for n, c in rep["attempts_failing_check"].items()))
    print("judge: " + (f"used ({j['reason']}), {j['calls_in_suite']} calls" if j["used"]
                       else f"not used ({j['reason']})"))
    try:
        shown = [str(p.relative_to(ROOT)) for p in (pj, pm)]
    except ValueError:
        shown = [str(pj), str(pm)]
    print(f"wrote {shown[0]} and {shown[1]}")


if __name__ == "__main__":
    main()
