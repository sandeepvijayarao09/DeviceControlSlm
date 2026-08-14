"""Statistical rigor: bootstrap CIs and paired model comparisons (eval-protocol §6).

Pure-stdlib (deterministic given seed) so results are reproducible without
SciPy/NumPy pinning issues.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple


def bootstrap_ci(values: List[float], iters: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) for the (1-alpha) percentile bootstrap CI of the mean."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return sum(values) / n, lo, hi


def paired_bootstrap_diff(a: List[float], b: List[float], iters: int = 10000,
                          seed: int = 0) -> Tuple[float, float, float, float]:
    """Paired bootstrap of mean(a)-mean(b). Returns (diff, lo, hi, p_two_sided).
    a and b are aligned per-task 0/1 outcomes for two systems."""
    assert len(a) == len(b) and a, "aligned non-empty vectors required"
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        da = db = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            da += a[i]
            db += b[i]
        diffs.append((da - db) / n)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters) - 1]
    obs = sum(a) / n - sum(b) / n
    # two-sided p: fraction of bootstrap diffs on the opposite side of 0 from obs
    if obs >= 0:
        p = 2 * (sum(1 for d in diffs if d <= 0) / iters)
    else:
        p = 2 * (sum(1 for d in diffs if d >= 0) / iters)
    return obs, lo, hi, min(1.0, p)


def mcnemar(a: List[int], b: List[int]) -> Tuple[int, int, float]:
    """McNemar's test on paired binary outcomes. Returns (b01, b10, p).
    b01 = a wrong & b right; b10 = a right & b wrong. Uses the exact binomial
    two-sided p-value on the discordant pairs."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    # two-sided exact binomial with p=0.5
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * cdf)
