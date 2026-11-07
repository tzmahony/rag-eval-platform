"""Metric definitions.

Every per-example metric is a float in a known range (or ``None`` when it does
not apply, e.g. retrieval recall for an unanswerable question). Aggregation
skips ``None`` values and reports ``n`` so you can see how many examples a
number is based on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    group: str
    higher_is_better: bool = True
    fmt: str = "{:.3f}"


METRICS: list[MetricSpec] = [
    MetricSpec("retrieval_precision", "Precision@k", "retrieval"),
    MetricSpec("retrieval_recall", "Recall@k", "retrieval"),
    MetricSpec("retrieval_mrr", "MRR", "retrieval"),
    MetricSpec("retrieval_ndcg", "nDCG@k", "retrieval"),
    MetricSpec("retrieval_hit", "Hit@k", "retrieval"),
    MetricSpec("context_relevance", "Context relevance", "context"),
    MetricSpec("faithfulness", "Faithfulness", "generation"),
    MetricSpec("answer_correctness", "Answer correctness", "generation"),
    MetricSpec("token_f1", "Token F1", "generation"),
    MetricSpec("abstained", "Abstention rate", "generation", fmt="{:.2f}"),
    MetricSpec("latency_retrieval_ms", "Retrieval latency (ms)", "performance", False, "{:.1f}"),
    MetricSpec("latency_generation_ms", "Generation latency (ms)", "performance", False, "{:.0f}"),
    MetricSpec("latency_total_ms", "Total latency (ms)", "performance", False, "{:.0f}"),
    MetricSpec("input_tokens", "Input tokens", "cost", False, "{:.0f}"),
    MetricSpec("output_tokens", "Output tokens", "cost", False, "{:.0f}"),
    MetricSpec("cost_usd", "Cost per query (USD)", "cost", False, "${:.5f}"),
]

METRIC_BY_KEY = {m.key: m for m in METRICS}

# Metrics that are tested for significance between configurations.
QUALITY_METRICS = [
    "retrieval_precision",
    "retrieval_recall",
    "retrieval_mrr",
    "context_relevance",
    "faithfulness",
    "answer_correctness",
]
