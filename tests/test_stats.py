import numpy as np
import pytest

from ragbench import stats


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    vals = list(np.linspace(0, 1, 50))
    a = stats.bootstrap_ci(vals, seed=1)
    b = stats.bootstrap_ci(vals, seed=1)
    assert a == b
    assert a.lo < a.mean < a.hi
    assert a.n == 50


def test_bootstrap_ignores_none():
    e = stats.bootstrap_ci([1.0, None, 0.0])
    assert e.n == 2 and e.mean == 0.5
    assert stats.bootstrap_ci([None]).mean is None


def test_permutation_test_detects_real_difference():
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 1, 60)
    a = b + 0.2
    assert stats.paired_permutation_test(a, b) < 0.001


def test_permutation_test_null():
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 1, 60)
    a = b + rng.normal(0, 0.1, 60)
    assert stats.paired_permutation_test(a, b) > 0.05
    assert stats.paired_permutation_test(b, b) == 1.0


def test_exact_permutation_small_n():
    # all 4 differences positive: exact two-sided p = 2/16, smoothed -> 3/17
    p = stats.paired_permutation_test(np.ones(4), np.zeros(4))
    assert p == pytest.approx(3 / 17)


def test_holm():
    adj = stats.holm_bonferroni([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_paired_values_alignment():
    a, b = stats.paired_values({"x": 1.0, "y": None, "z": 3.0}, {"x": 0.0, "y": 1.0, "w": 2.0, "z": 1.0})
    assert a.tolist() == [1.0, 3.0] and b.tolist() == [0.0, 1.0]


def test_kappa_and_spearman():
    assert stats.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
