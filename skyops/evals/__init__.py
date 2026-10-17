"""Section 10: the eval harness. Golden data, deterministic checks, a calibrated judge, variance across runs.

    dataset.py   the golden-set format (JSONL + JSON Schema), loading and validation (10.3)
    runner.py    run the disruption handler on one case, one attempt, against its replay
    checks.py    outcome, tool-call and trajectory checks (10.4); message assertions (10.5)
    judge.py     an LLM judge for the one criterion code can't check, and its calibration (10.6)
    stats.py     pass@k, pass^k and Cohen's kappa (10.7, 10.8)
    harness.py   score every attempt, aggregate, write the report (10.8)
Section 11 puts this harness in CI and runs it on sampled production traffic.
"""
