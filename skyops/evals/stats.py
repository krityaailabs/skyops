"""Lectures 10.7 and 10.8: the statistics the harness reports. Plain Python, exact, no dependencies.

pass@k  (Chen et al. 2021, "Evaluating Large Language Models Trained on Code", the HumanEval paper): the
        chance that at least one of k attempts passes. Unbiased from n attempts with c passes:
        1 - C(n-c, k) / C(n, k). Computed as a product, because the binomials overflow for large n.
pass^k  (Yao et al. 2024, tau-bench): the chance that ALL k attempts pass: C(c, k) / C(n, k). For an agent
        that acts on passengers, this is the reliability number; pass@k is the "can it ever" number.
Cohen's kappa (Cohen 1960): agreement between two raters, corrected for the agreement chance alone would give:
        (p_o - p_e) / (1 - p_e). Used to calibrate the judge against human labels. With a few dozen labels it is
        rough, so bootstrap_kappa_ci reports how rough.
Section 11 adds two tools for the gate and for drift: the Wilson score interval for a pass rate (wilson_interval),
and a one-sided two-proportion z-test for "did this rate fall?" (two_proportion_test).
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Hashable, Sequence
from skyops.model.errors import LabNotDone


def _check(n: int, c: int, k: int) -> None:
    if not (0 <= c <= n) or k < 1 or k > n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n (got n={n}, c={c}, k={k})")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Lecture 10.8: the unbiased pass@k estimator for one case, from n attempts of which c passed.

    pass@k = 1 - C(n - c, k) / C(n, k): one minus the chance that k attempts drawn without replacement are
    all failures. Raise ValueError unless 0 <= c <= n and 1 <= k <= n. If n - c < k, every draw of k
    contains a pass: return 1.0. Otherwise compute the ratio as a running product,
    prod over i in (n - c + 1 .. n) of (1 - k / i), as the paper's numerically stable version does, so large n
    (say n = 200) never builds huge integers or overflows a float. Return a float in [0, 1].
    Note: 1 - (1 - c/n) ** k is NOT this estimator; it is biased low.
    """
    _check(n, c, k)
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Lecture 10.8: the unbiased pass^k estimator (all k attempts pass), from n attempts with c passes.

    pass^k = C(c, k) / C(n, k). Same argument checks as pass_at_k (ValueError). If c < k, return 0.0.
    Compute it as the product over i in 0 .. k-1 of (c - i) / (n - i). Return a float in [0, 1].
    """
    _check(n, c, k)
    if c < k:
        return 0.0
    prod = 1.0
    for i in range(k):
        prod *= (c - i) / (n - i)
    return prod


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Lecture 10.8: Cohen's kappa between two raters' labels for the same items, in the same order.

    p_o = the share of items where a[i] == b[i]; p_e = the sum over every label L used by either rater of
    (share of a that is L) * (share of b that is L); kappa = (p_o - p_e) / (1 - p_e). Works for any number
    of labels, not just two. Raise ValueError if the sequences differ in length or are empty, and if p_e == 1
    (both raters used one and the same label for everything: kappa is undefined, and the set can't calibrate
    anything). Return a float; it can be negative (worse than chance).
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label sequences of the same length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(ca) | set(cb))
    if pe == 1:
        raise ValueError("kappa is undefined: both raters used a single, identical label")
    return (po - pe) / (1 - pe)


def bootstrap_kappa_ci(a: Sequence[Hashable], b: Sequence[Hashable], n_boot: int = 2000, seed: int = 10,
                       level: float = 0.95) -> tuple[float, float]:
    """A percentile bootstrap interval for Cohen's kappa: resample the items with replacement n_boot times and
    take the middle `level` of the kappas. Seeded, so the report is reproducible. Resamples where kappa is
    undefined (one label everywhere) are skipped. On forty labels the interval is wide, which is the point:
    a threshold needs hundreds of labels before a kappa near it means much."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label sequences of the same length")
    rng, n, ks = random.Random(seed), len(a), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            ks.append(cohen_kappa([a[i] for i in idx], [b[i] for i in idx]))
        except ValueError:
            continue
    ks.sort()
    lo = ks[int((1 - level) / 2 * len(ks))]
    hi = ks[min(len(ks) - 1, int((1 + level) / 2 * len(ks)))]
    return lo, hi


def agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Raw agreement, the number kappa corrects. Shown next to kappa, never instead of it."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label sequences of the same length")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def interpret_kappa(k: float) -> str:
    """Landis and Koch's (1977) bands. Widely quoted, and a convention, not a law: set your own threshold
    from what a wrong verdict costs, and report the number."""
    if k <= 0:
        return "no better than chance"
    for top, word in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")):
        if k <= top:
            return word
    return "almost perfect"


# ---------- Section 11: intervals and tests for the gate and for drift ----------

def wilson_interval(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Lecture 11.2: the Wilson score interval for a pass rate of c passes in n attempts (Wilson 1927).

    p = c / n; denom = 1 + z**2 / n; centre = (p + z**2 / (2 * n)) / denom;
    half = (z / denom) * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)). Return (centre - half, centre + half), each
    clamped to [0, 1], and exactly 0.0 as the lower end when c == 0 and exactly 1.0 as the upper end when c == n
    (floating point would otherwise give 0.9999999999999999). Raise ValueError unless n >= 1 and 0 <= c <= n.
    With z = 1.96 it is a 95% interval.
    Unlike the textbook p +/- z * sqrt(p(1-p)/n), it never collapses to zero width at 0/n or n/n, which is
    exactly where small eval suites live. It assumes the n attempts are independent; attempts of the same case
    are not (some cases are simply harder), so for a suite it is a lower bound on the real uncertainty.
    """
    raise LabNotDone("Lecture 11.2: implement wilson_interval(). Hints: labs/s11/README.md, lecture 11.2.")


def two_proportion_test(c1: int, n1: int, c2: int, n2: int) -> tuple[float, float]:
    """Lecture 11.1: is the second rate (c2 of n2) LOWER than the first (c1 of n1)? A one-sided two-proportion
    z-test with the pooled standard error.

    p1 = c1 / n1, p2 = c2 / n2, pooled = (c1 + c2) / (n1 + n2), se = sqrt(pooled * (1 - pooled) * (1/n1 + 1/n2)),
    z = (p1 - p2) / se, and the one-sided p-value is P(Z >= z) = 0.5 * erfc(z / sqrt(2)). Return (z, p).
    If pooled is 0 or 1 (both rates identical at the extreme), there is no evidence of a drop: return (0.0, 1.0).
    Raise ValueError unless n1, n2 >= 1 and each c is within 0..n.
    Assumptions, stated because they matter: every attempt is an independent trial with the same chance of
    passing within its group, and both groups are large enough for the normal approximation (a rough guide:
    at least five passes and five failures expected in each). Eval attempts are clustered by case, so the
    p-value is optimistic; treat it as a noise filter, never as proof.
    """
    raise LabNotDone("Lecture 11.1: implement two_proportion_test(). Hints: labs/s11/README.md, lecture 11.1.")


def fisher_drop_p(c1: int, n1: int, c2: int, n2: int) -> float:
    """Fisher's exact test, one-sided: the chance of the second group having c2 passes OR FEWER, given both groups'
    sizes and the total number of passes, if both groups shared one pass rate (the hypergeometric tail). Exact at
    any size, so it is the right test when cells are small; it still treats attempts as independent."""
    if n1 < 1 or n2 < 1 or not (0 <= c1 <= n1 and 0 <= c2 <= n2):
        raise ValueError(f"need n >= 1 and 0 <= c <= n (got {c1}/{n1}, {c2}/{n2})")
    k, n = c1 + c2, n1 + n2
    total = math.comb(n, n2)
    return min(1.0, sum(math.comb(k, x) * math.comb(n - k, n2 - x) for x in range(max(0, k - n1), c2 + 1)) / total)


MIN_EXPECTED = 10        # below this many expected passes or failures in any cell, the normal approximation is unsafe


def drop_p(c1: int, n1: int, c2: int, n2: int) -> tuple[float, str]:
    """The one-sided p-value for "the second rate is lower than the first", and the method used. Use
    two_proportion_test when every cell's expected count (each group's n times the pooled pass and fail shares) is
    at least MIN_EXPECTED; otherwise Fisher's exact test. Return (p, "z") or (p, "exact")."""
    pooled = (c1 + c2) / (n1 + n2)
    expected = [n * share for n in (n1, n2) for share in (pooled, 1 - pooled)]
    if min(expected) >= MIN_EXPECTED:
        return two_proportion_test(c1, n1, c2, n2)[1], "z"
    return fisher_drop_p(c1, n1, c2, n2), "exact"
