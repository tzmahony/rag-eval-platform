import math

import pytest

from ragbench.judges import HeuristicJudge
from ragbench.metrics.answer import exact_match, token_f1
from ragbench.metrics.retrieval import retrieval_metrics
from ragbench.types import Chunk, EvalExample, ScoredChunk


def _hits(*doc_ids):
    return [
        ScoredChunk(Chunk(f"{d}#{i}", d, f"text about {d}", i), 1.0 / (i + 1), i + 1) for i, d in enumerate(doc_ids)
    ]


def test_retrieval_metrics_doc_level():
    ex = EvalExample("q", "?", "!", relevant_doc_ids=("a", "b"))
    m = retrieval_metrics(_hits("x", "a", "y", "z"), ex, k=4, n_relevant_in_index=2)
    assert m["retrieval_precision"] == pytest.approx(0.25)
    assert m["retrieval_recall"] == pytest.approx(0.5)
    assert m["retrieval_mrr"] == pytest.approx(0.5)
    assert m["retrieval_hit"] == 1.0
    assert m["retrieval_ndcg"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_retrieval_metrics_evidence_level():
    ex = EvalExample("q", "?", "!", relevant_doc_ids=("a",), evidence=("tank capacity is 30 liters",))
    hits = [
        ScoredChunk(Chunk("a#0", "a", "The weight is 26 kilograms.", 0), 2.0, 1),
        ScoredChunk(Chunk("a#1", "a", "Its tank capacity is 30 liters.", 1), 1.0, 2),
    ]
    m = retrieval_metrics(hits, ex, k=2, n_relevant_in_index=1)
    # same document, but only the second chunk holds the evidence
    assert m["retrieval_precision"] == pytest.approx(0.5)
    assert m["retrieval_recall"] == 1.0
    assert m["retrieval_mrr"] == pytest.approx(0.5)


def test_retrieval_metrics_skip_unanswerable():
    ex = EvalExample("q", "?", "!", answerable=False)
    assert all(v is None for v in retrieval_metrics(_hits("a"), ex, k=1).values())


def test_token_f1_and_em():
    assert token_f1("21 days per year", "21 days per year") == 1.0
    assert token_f1("", "x") == 0.0
    assert 0 < token_f1("It is 21 days", "21 days per year") < 1
    assert exact_match("The 21 days.", "21 days") == 1.0


judge = HeuristicJudge()
CTX = ["The K9 tank capacity is 30 liters. The K9 weighs 26 kilograms empty."]


def test_faithfulness_flags_unsupported_number():
    good = judge.faithfulness("The tank capacity is 30 liters.", CTX)
    bad = judge.faithfulness("The tank capacity is 45 liters.", CTX)
    assert good.score == 1.0
    assert bad.score == 0.0


def test_faithfulness_partial_and_abstention():
    r = judge.faithfulness("The tank capacity is 30 liters. It was designed in Denver by Elon.", CTX)
    assert r.score == 0.5
    assert judge.faithfulness("I don't know.", CTX).score == 1.0


def test_context_relevance_prefers_on_topic_context():
    q = "What is the tank capacity of the K9?"
    on = judge.context_relevance(q, CTX).score
    off = judge.context_relevance(q, ["Employees accrue 21 days of PTO per year."]).score
    assert on > off


def test_answer_correctness_rules():
    assert judge.answer_correctness("q", "It holds 30 liters.", "30 liters.", True).score == 1.0
    assert judge.answer_correctness("q", "It holds 40 liters.", "30 liters.", True).score <= 0.5
    assert judge.answer_correctness("q", "I don't know.", "30 liters.", True).score == 0.0
    assert judge.answer_correctness("q", "I don't know.", "n/a", False).score == 1.0
    assert judge.answer_correctness("q", "It is 60 km/h.", "n/a", False).score == 0.0
