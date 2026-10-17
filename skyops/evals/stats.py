"""Lectures 10.7 and 10.8: the statistics the harness reports. Plain Python, exact, no dependencies.

pass@k  (Chen et al. 2021, "Evaluating Large Language Models Trained on Code", the HumanEval paper): the
        chance that at least one of k attempts passes. Unbiased from n attempts with c passes:
        1 - C(n-c, k) / C(n, k). Computed as a product, because the binomials overflow for large n.
pass^k  (Yao et al. 2024, tau-bench): the chance that ALL k attempts pass: C(c, k) / C(n, k). For an agent
        that acts on passengers, this is the reliability number; pass@k is the "can it ever" number.
Cohen's kappa (Cohen 1960): agreement between two raters, corrected for the agreement chance alone would give:
        (p_o - p_e) / (1 - p_e). Used to calibrate the judge against human labels. With a few dozen labels it is
        rough, so bootstrap_kappa_ci reports how rough.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Hashable, Sequence


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
