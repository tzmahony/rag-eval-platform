"""Statistics for honest comparisons: bootstrap CIs, paired tests, multiple-comparison control.

Small eval sets are noisy. A configuration that scores 0.82 vs 0.79 on 30
questions is usually *not* demonstrably better. These helpers make that explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    mean: float | None
    lo: float | None
    hi: float | None
    n: int

    def to_dict(self) -> dict:
        return {"mean": self.mean, "lo": self.lo, "hi": self.hi, "n": self.n}


def _clean(values) -> np.ndarray:
    return np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)


def bootstrap_ci(values, *, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> Estimate:
    """Percentile bootstrap confidence interval for the mean."""
    x = _clean(values)
    if x.size == 0:
        return Estimate(None, None, None, 0)
    mean = float(x.mean())
    if x.size == 1:
        return Estimate(mean, mean, mean, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(mean, float(lo), float(hi), int(x.size))


def paired_values(a: dict[str, float | None], b: dict[str, float | None]) -> tuple[np.ndarray, np.ndarray]:
    """Align two {example_id: value} maps on the examples where both are defined."""
    keys = sorted(k for k in a.keys() & b.keys() if a[k] is not None and b[k] is not None)
    return np.array([a[k] for k in keys], dtype=float), np.array([b[k] for k in keys], dtype=float)


def paired_permutation_test(a: np.ndarray, b: np.ndarray, *, n_resamples: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test for mean(a - b) == 0.

    Exact for n <= 16, Monte Carlo otherwise. Makes no normality assumption,
    which matters for bounded and often bimodal metrics such as faithfulness.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n == 0 or np.allclose(d, 0):
        return 1.0
    observed = abs(d.mean())
    if n <= 16:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    null = np.abs((signs * d).mean(axis=1))
    # +1 smoothing keeps Monte Carlo p-values valid (never exactly 0)
    return float((np.sum(null >= observed - 1e-12) + 1) / (null.size + 1))


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, *, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Estimate:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d.tolist(), n_resamples=n_resamples, alpha=alpha, seed=seed)


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (controls family-wise error rate)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def cohens_kappa(a: list[int], b: list[int]) -> float | None:
    if not a or len(a) != len(b):
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None

    def ranks(x):
        x = np.asarray(x, dtype=float)
        order = x.argsort(kind="mergesort")
        r = np.empty_like(x)
        r[order] = np.arange(len(x))
        for v in np.unique(x):  # average ties
            mask = x == v
            r[mask] = r[mask].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])
