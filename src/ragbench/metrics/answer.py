"""Deterministic, judge-independent answer metrics."""

from __future__ import annotations

from collections import Counter

from ragbench.text import normalize_answer


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred), same / len(ref)
    return 2 * p * r / (p + r)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))
