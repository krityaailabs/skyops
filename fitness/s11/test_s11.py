"""Section 11 fitness: evals in CI and in production. A per-category regression gate against a stored baseline,
flake handling (replay, intervals, a significance test, quarantine with an owner and an expiry), contamination
checks, PSI for drift, and a CI workflow that runs the same command as you do. Fail at s11-start, pass at s11.
Offline and deterministic: the handler's turns are scripted replays and the production log is synthetic; every
check, test and gate decision is real code."""
import json
import logging
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

from skyops.evals.contamination import Item, find_contamination
from skyops.evals.gate import (BASELINE, Rule, case_diff, compare_category, evaluate_gate, load_config,
                               load_quarantine, quarantine_problems)
from skyops.evals.harness import run_suite
from skyops.evals.online import psi
from skyops.evals.stats import two_proportion_test, wilson_interval
from skyops.settings import ROOT

R = Rule(min_pass_hat_k=0.4, max_drop=0.10)


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def gate(version, suites=("golden",), today="2026-10-17", contamination=()):
    cfg = {**load_config(), "suites": list(suites)}
    reports = {s: run_suite(suite=s, version=version, k=cfg["k"]) for s in suites}
    return evaluate_gate(reports, json.loads(BASELINE.read_text()), cfg, load_quarantine(), today, contamination), \
        reports


# ---------- 11.1 the regression gate ----------

def test_l111_compare_category_checks_the_floor_and_a_significant_drop():
    base = [(5, 5)] * 6                                 # 30/30
    worse = [(5, 2)] * 6                                # 12/30, pass^3 0.0
    f, w = compare_category("edge", base, worse, R)
    assert f == ["edge: pass^3 0.00 is below the floor 0.40",
                 "edge: pass rate fell from 100% to 40% (60 points, more than 10; p < 0.001, exact)"]
    assert w == []
    f, w = compare_category("edge", [(5, 4), (5, 4)], [(5, 3), (5, 3)], Rule(0.0, 0.10))   # 80% -> 60%, n = 10
    assert f == [] and w == ["edge: pass rate fell from 80% to 60% (20 points, more than 10; p = 0.314, exact), "
                             "not significant: add cases before trusting it"]      # Fisher: 58058 / 184756
    assert compare_category("edge", base, base, R) == ([], [])
    with pytest.raises(ValueError):
        compare_category("edge", base, [], R)


def test_l111_case_diff_lists_only_the_changed_cases_in_order():
    base = {"a": {"category": "edge", "n": 5, "passes": 3}, "b": {"category": "edge", "n": 4, "passes": 2},
            "z": {"category": "normal", "n": 5, "passes": 5}, "y": {"category": "normal", "n": 5, "passes": 1}}
    cand = {"b": {"category": "edge", "n": 2, "passes": 1}, "a": {"category": "edge", "n": 5, "passes": 0},
            "c": {"category": "adversarial", "n": 5, "passes": 4}, "d": {"category": "edge", "n": 5, "passes": 5}}
    base["d"] = {"category": "edge", "n": 5, "passes": 4}
    assert case_diff(base, cand) == [
        {"id": "a", "category": "edge", "base": "3/5", "cand": "0/5", "change": "worse"},
        {"id": "c", "category": "adversarial", "base": "-", "cand": "4/5", "change": "new"},
        {"id": "d", "category": "edge", "base": "4/5", "cand": "5/5", "change": "better"},
        {"id": "y", "category": "normal", "base": "1/5", "cand": "-", "change": "removed"},
        {"id": "z", "category": "normal", "base": "5/5", "cand": "-", "change": "removed"},
    ]                                                   # b: 2/4 and 1/2 are the same rate, so it isn't listed


def test_l111_two_proportion_test_is_one_sided_with_a_pooled_error():
    z, p = two_proportion_test(24, 30, 16, 30)
    assert round(z, 4) == 2.1909 and round(p, 5) == 0.01423
    assert math.isclose(p, 0.5 * math.erfc(z / math.sqrt(2)))
    assert two_proportion_test(5, 5, 7, 7) == (0.0, 1.0)
    with pytest.raises(ValueError):
        two_proportion_test(6, 5, 1, 5)


def test_l111_small_cells_use_fishers_exact_test():
    from skyops.evals.stats import drop_p, fisher_drop_p
    assert round(fisher_drop_p(17, 20, 12, 20), 4) == 0.0776                   # 5.5 expected failures: too few
    assert drop_p(17, 20, 12, 20) == (fisher_drop_p(17, 20, 12, 20), "exact")
    assert drop_p(24, 30, 16, 30) == (two_proportion_test(24, 30, 16, 30)[1], "z")   # 10 expected in every cell
    assert fisher_drop_p(5, 5, 5, 5) == 1.0


def _pairs(cases, cat):
    return [(v["n"], v["passes"]) for v in cases.values() if v["category"] == cat]


def test_l111_per_category_comparison_blocks_v3_where_one_overall_number_would_not():
    base = json.loads(BASELINE.read_text())["suites"]["golden"]["cases"]
    rep = run_suite(suite="golden", version="v3")
    cand = {r["id"]: {"category": r["category"], "n": r["n"], "passes": r["passes"]} for r in rep["cases"]}
    assert rep["overall"]["pass_rate"] >= 0.70             # a single overall threshold of 70% passes it
    f, _ = compare_category("edge", _pairs(base, "edge"), _pairs(cand, "edge"), R)
    assert f == ["edge: pass rate fell from 80% to 53% (27 points, more than 10; p = 0.014)"]
    assert compare_category("normal", _pairs(base, "normal"), _pairs(cand, "normal"), R) == ([], [])
    worse = [r["id"] for r in case_diff(base, cand) if r["change"] == "worse"]
    assert worse == ["e01_gate_mismatch", "e04_departed", "e05_time_mismatch", "a03_fake_cancel"]


# ---------- 11.2 flaky evals ----------

def test_l112_wilson_interval_matches_the_formula_and_never_collapses():
    lo, hi = wilson_interval(24, 30)
    assert (round(lo, 4), round(hi, 4)) == (0.6269, 0.905)
    assert wilson_interval(0, 5)[0] == 0.0 and round(wilson_interval(0, 5)[1], 4) == 0.4345
    assert wilson_interval(5, 5)[1] == 1.0 and wilson_interval(5, 5)[0] < 1.0
    assert wilson_interval(6, 6)[1] == 1.0 and wilson_interval(0, 7)[0] == 0.0          # exactly, not 0.999...
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_l112_quarantine_needs_an_owner_an_expiry_and_never_takes_an_attack():
    known = {"golden": {"e03": "edge", "e04": "edge", "a01": "adversarial"}, "heldout": {"e21": "edge"}}
    ok = {"case": "e03", "suite": "golden", "owner": "on-call", "reason": "r", "ticket": "SKY-1",
          "added": "2026-10-17", "expires": "2026-10-31"}
    entries = [ok, {**ok, "owner": " "}, {**ok, "suite": "heldout", "case": "e21"}, {**ok, "case": "x9"},
               {**ok, "case": "a01"}, {**ok, "expires": "2026-12-01"}, {**ok, "expires": "2026-10-10"},
               {**ok, "renewals": 3}, {**ok, "case": "e04"}]
    active, problems = quarantine_problems(entries, "2026-10-31", known)
    assert active == {"golden/e03"}                     # the expiry date itself is still in force
    assert problems == ["quarantine e03: missing owner",
                        "quarantine: a held-out case can't be quarantined (id in the restricted report)",
                        "quarantine x9: no such case in golden",
                        "quarantine a01: adversarial cases can't be quarantined",
                        "quarantine e03: 45 days is longer than 30",
                        "quarantine e03: expires before it was added",
                        "quarantine e03: renewed 3 times, more than 2",
                        "quarantine e04: edge already has 1 quarantined case(s), the limit"]
    active, problems = quarantine_problems([ok], "2026-11-01", known)
    assert active == set() and problems == [
        "quarantine e03: expired on 2026-10-31 (owner on-call): fix the case or renew it in a reviewed change"]
    assert quarantine_problems([ok], "2026-10-16", known) == (set(), [])         # not added yet: not in force


def test_l112_replay_makes_runs_repeatable_and_the_real_quarantine_expires():
    a, b = run_suite(suite="golden", version="v2"), run_suite(suite="golden", version="v2")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    known = {"golden": {r["id"]: r["category"] for r in a["cases"]}}
    assert quarantine_problems(load_quarantine(), "2026-10-17", known) == ({"golden/e03_shorthand_spacing"}, [])
    active, problems = quarantine_problems(load_quarantine(), "2026-11-01", known)
    assert active == set() and problems[0].startswith("quarantine e03_shorthand_spacing: expired on 2026-10-31")


# ---------- 11.5 Incident 18: contamination ----------

def test_l115_find_contamination_follows_its_rules():
    items = [Item("long", "golden notice", "ZZ214 SIN now ETD 21:30 (was 19:40), late inbound."),
             Item("short", "golden notice", "ZZ142 JFK boarding on time, gate C2."),
             Item("none", "judge label", "...")]
    sources = {"ft": ["zz214 sin now etd 21:30 was 19:40, late inbound aircraft"],
               "split": ["ZZ142 JFK boarding", "on time, gate C2."],                  # never across two texts
               "prompt": ["Note: ZZ142 JFK boarding on time, gate C2. Then check."]}
    hits = find_contamination(items, sources)
    assert [(h.item, h.source, h.share) for h in hits] == [("long", "ft", 1.0), ("short", "prompt", 1.0)]
    half = find_contamination(items[:1], {"half": ["zz214 sin now etd 21 30 was 19"]}, threshold=0.25)
    assert [(h.item, h.share) for h in half] == [("long", 0.25)]                     # 1 of the item's 4 8-grams


def test_l115_contamination_kept_the_old_gate_green_and_the_held_out_suite_sees_v4():
    from skyops.evals.contamination import eval_items, model_inputs
    base = json.loads(BASELINE.read_text())["suites"]
    gold = {r["id"]: r for r in run_suite(suite="golden", version="v4")["cases"]}
    held = {r["id"]: r for r in run_suite(suite="heldout", version="v4")["cases"]}
    for cat in ("normal", "edge", "adversarial"):                 # the old gate: golden only, by category
        assert compare_category(cat, _pairs(base["golden"]["cases"], cat), _pairs(gold, cat), R)[0] == []
    f, _ = compare_category("edge", _pairs(base["heldout"]["cases"], "edge"), _pairs(held, "edge"), R)
    assert f[1].startswith("edge: pass rate fell from 85% to 25%")
    hits = find_contamination(eval_items(), model_inputs("v4"))
    assert sorted(h.item for h in hits) == ["a03_fake_cancel", "e01_gate_mismatch", "e04_departed",
                                            "e05_time_mismatch"]
    assert find_contamination(eval_items(), model_inputs("v2")) == []


def test_l115_no_held_out_case_shares_a_flight_with_the_golden_set():
    from skyops.evals.contamination import case_flights, shared_flights
    from skyops.evals.dataset import GOLDEN, HELDOUT, load_golden
    golden = set().union(*(case_flights(c) for c in load_golden(GOLDEN)))
    assert all(not (case_flights(c) & golden) for c in load_golden(HELDOUT)) and shared_flights() == []
    assert {"ZZ214", "ZZ230", "XY455"} <= golden                        # notices' flights count, not just case.flight


def test_l115_psi_is_zero_for_identical_symmetric_and_survives_an_empty_bin():
    assert psi({"a": 5, "b": 5}, {"a": 50, "b": 50}) == 0.0
    x = psi({"a": 80, "b": 20}, {"a": 50, "b": 50})
    assert round(x, 4) == round(0.3 * math.log(0.5 / 0.2) - 0.3 * math.log(0.5 / 0.8), 4) == 0.4159
    assert math.isclose(x, psi({"a": 50, "b": 50}, {"a": 80, "b": 20}))
    e = psi({"a": 10}, {"a": 5, "b": 5})                                # "b" is empty in the first: eps
    assert math.isclose(e, (0.5 - 1.0) * math.log(0.5) + (0.5 - 1e-4) * math.log(0.5 / 1e-4))
    with pytest.raises(ValueError):
        psi({}, {"a": 1})


def test_l115_the_detection_signal_fires_on_the_day_of_the_release():
    from labs.s11 import incident
    f = incident.figures()
    first = next(d for d, *_, alert in f["daily_all"] if alert)
    assert first == incident.RELEASE[:10] and f["v4_bad"] > 5 * f["before_bad"] * f["v4_runs"] / f["before_runs"]


# ---------- 11.6 the CI gate ----------

def _rep(cases, sha="s1", judge=None):
    return {"meta": {"dataset_sha256": sha}, "judge": judge or {"used": False},
            "cases": [{"id": i, "category": c, "n": n, "passes": p} for i, c, n, p in cases]}


def test_l116_evaluate_gate_collects_every_failure_in_order():
    cfg = {"suites": ["golden", "heldout", "extra"], "k": 3, "alpha": 0.05, "max_quarantine_days": 30,
           "rules": {"normal": Rule(0.5, 0.1), "edge": Rule(0.3, 0.1)}}
    base = {"suites": {"golden": {"dataset_sha256": "s1", "cases": {
        "n1": {"category": "normal", "n": 5, "passes": 5}, "n2": {"category": "normal", "n": 5, "passes": 5},
        "e1": {"category": "edge", "n": 5, "passes": 3}, "e2": {"category": "edge", "n": 5, "passes": 4}}},
        "heldout": {"dataset_sha256": "h1", "cases": {}}}}
    golden = _rep([("n1", "normal", 5, 0), ("n2", "normal", 5, 0), ("e1", "edge", 5, 0), ("e2", "edge", 5, 4)],
                  judge={"used": True, "kappa_judged": 0.5, "threshold": 0.6})
    reports = {"golden": golden, "heldout": _rep([], sha="h2")}
    quarantine = [{"case": "e1", "suite": "golden", "owner": "o", "reason": "r", "ticket": "t",
                   "added": "2026-10-17", "expires": "2026-10-31"}, {"case": "n9", "suite": "golden"}]
    leak = SimpleNamespace(item="e2", kind="golden notice", source="prompt v9", share=1.0)
    res = evaluate_gate(reports, base, cfg, quarantine, "2026-10-20", [leak])
    assert res.failures == [
        "contamination: e2 (golden notice) found in prompt v9 (100% of its n-grams)",
        "quarantine n9: missing owner",
        "golden: the judge counted without calibration",
        "golden normal: pass^3 0.00 is below the floor 0.50",
        "golden normal: pass rate fell from 100% to 0% (100 points, more than 10; p < 0.001, exact)",
        "golden n1: now fails every attempt (5/5 -> 0/5)",
        "golden n2: now fails every attempt (5/5 -> 0/5)",
        "heldout: dataset changed (baseline h1, now h2): re-baseline in its own reviewed change",
        "no report for suite extra",
    ]
    assert not res.passed and res.quarantined == ["golden/e1"]       # e1 is 3/5 -> 0/5 but quarantined
    assert [(r["id"], r["quarantined"]) for r in res.diff["golden"]] == [("n1", False), ("n2", False), ("e1", True)]
    assert res.numbers["golden"]["candidate"]["edge"]["cases"] == 1
    held = SimpleNamespace(item="e21", kind="held-out notice", source="golden set (same flight)", share=1.0)
    only_edge = {"golden": _rep([("e1", "edge", 5, 3)])}
    base1 = {"suites": {"golden": {"dataset_sha256": "s1", "cases": {"e1": {"category": "edge", "n": 5, "passes": 3}}}}}
    res = evaluate_gate(only_edge, base1, {**cfg, "suites": ["golden"]}, quarantine[:1], "2026-10-20", [held])
    assert res.failures == ["heldout: 1 contamination hit(s) (ids in the restricted report)",
                            "golden edge: every case is quarantined"]       # a category never silently vanishes


def test_l116_the_full_gate_blocks_v3_and_v4_and_passes_main():
    from skyops.evals.contamination import eval_items, model_inputs
    green, _ = gate("v2", suites=("golden", "heldout"))
    assert green.passed and green.quarantined == ["golden/e03_shorthand_spacing"]
    red, _ = gate("v3")
    assert red.failures[:3] == ["golden edge: pass rate fell from 80% to 48% (32 points, more than 10; p = 0.019, "
                                "exact)",
                                "golden e01_gate_mismatch: now fails every attempt (3/5 -> 0/5)",
                                "golden e05_time_mismatch: now fails every attempt (3/5 -> 0/5)"]
    then, _ = gate("v4", today="2026-09-28")                       # Incident 18: the gate as it was
    assert then.passed
    hits = find_contamination(eval_items(), model_inputs("v4"))
    now, _ = gate("v4", suites=("golden", "heldout"), today="2026-09-28", contamination=hits)
    assert not now.passed and sum(f.startswith("contamination: ") for f in now.failures) == 4
    assert any(f.startswith("heldout edge: pass rate fell from 85% to 25%") for f in now.failures)


def test_l116_the_workflow_is_valid_and_runs_the_same_command_as_make():
    sys.path.insert(0, str(ROOT / "tools"))
    import check_workflow
    assert check_workflow.problems() == []
    doc = check_workflow.load(check_workflow.WORKFLOW)
    runs = [s.get("run", "") for s in doc["jobs"]["eval-gate"]["steps"]]
    assert any(r.startswith("make ci-gate") for r in runs)
    assert "ci-gate:" in (ROOT / "Makefile").read_text()
    assert (ROOT / ".github/CODEOWNERS").exists()


def test_l116_the_workflow_checker_refuses_a_gate_that_can_be_turned_off(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import check_workflow
    text = check_workflow.WORKFLOW.read_text().replace("  pull_request:", "  pull_request_target:")
    text = text.replace('run: make ci-gate BASE_REF="$BASE_REF"',
                        'run: make ci-gate BASE_REF="$BASE_REF"\n        continue-on-error: true')
    bad = tmp_path / "bad.yml"
    bad.write_text(text)
    found = check_workflow.problems(bad)
    assert any("pull_request_target" in f for f in found)
    assert any("must not have `continue-on-error`" in f for f in found)


def _cli(tmp, *args):
    return subprocess.run([sys.executable, "-m", "skyops.evals.gate", "--baseline", str(BASELINE), "--out", str(tmp),
                           *args],
                          cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin",
                                                                         "SKYOPS_TODAY": "2026-10-17"})


def test_l116_the_ci_gate_is_green_for_main_red_for_the_injected_regression(tmp_path):
    ok, red = _cli(tmp_path), _cli(tmp_path, "--handler", "v3")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert red.returncode == 1 and "RESULT: RED" in red.stdout
    assert "FAIL golden edge: pass rate fell from 80% to 48%" in red.stdout
    assert "per-case results stay in the restricted report" in red.stdout     # held-out cases never in the log
    assert "n21_" not in red.stdout and "e21_" not in red.stdout
    report = json.loads((tmp_path / "gate.json").read_text())
    assert report["meta"]["candidate_handler"] == "v3" and report["meta"]["baseline_handler"] == "v2"
    assert report["meta"]["suites"]["golden"]["judge_used"] is False and "heldout" not in report["diff"]


def test_l116_a_prompt_without_recordings_cant_pass_on_old_ones(tmp_path):
    out = _cli(tmp_path, "--handler", "v1")
    assert out.returncode == 2 and "re-record" in out.stdout
    assert "RED" in (tmp_path / "gate.md").read_text()                # the job summary is written on exit 2 too
