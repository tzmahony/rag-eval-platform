"""Retrieval precision/recall and ranking metrics against labelled evidence."""

from __future__ import annotations

import math

from ragbench.text import content_tokens
from ragbench.types import Chunk, EvalExample, ScoredChunk


def evidence_in_text(evidence: str, text: str, threshold: float = 0.6) -> bool:
    """True if at least ``threshold`` of the evidence's content tokens occur in ``text``.

    A threshold below 1 tolerates evidence that straddles a chunk boundary.
    """
    ev = content_tokens(evidence)
    if not ev:
        return False
    toks = set(content_tokens(text))
    return sum(t in toks for t in ev) / len(ev) >= threshold


def is_relevant(chunk: Chunk, ex: EvalExample, threshold: float = 0.6) -> bool:
    """Chunk-level relevance: matches an evidence passage when evidence is labelled,
    otherwise falls back to belonging to a relevant document."""
    if ex.evidence:
        return any(evidence_in_text(e, chunk.text, threshold) for e in ex.evidence)
    return chunk.doc_id in ex.relevant_doc_ids


def retrieval_metrics(
    retrieved: list[ScoredChunk],
    ex: EvalExample,
    *,
    k: int,
    n_relevant_in_index: int | None = None,
    threshold: float = 0.6,
) -> dict[str, float | None]:
    keys = ["retrieval_precision", "retrieval_recall", "retrieval_mrr", "retrieval_ndcg", "retrieval_hit"]
    if not ex.answerable or not (ex.relevant_doc_ids or ex.evidence):
        return {key: None for key in keys} | {"doc_recall": None}

    top = retrieved[:k]
    rels = [is_relevant(sc.chunk, ex, threshold) for sc in top]

    precision = sum(rels) / k if k else 0.0

    if ex.evidence:
        found = [any(evidence_in_text(e, sc.chunk.text, threshold) for sc in top) for e in ex.evidence]
        recall = sum(found) / len(found)
    else:
        retrieved_docs = {sc.chunk.doc_id for sc in top}
        recall = len(retrieved_docs & set(ex.relevant_doc_ids)) / len(ex.relevant_doc_ids)

    doc_recall = (
        len({sc.chunk.doc_id for sc in top} & set(ex.relevant_doc_ids)) / len(ex.relevant_doc_ids)
        if ex.relevant_doc_ids
        else None
    )

    mrr = next((1.0 / (i + 1) for i, r in enumerate(rels) if r), 0.0)

    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels) if r)
    n_ideal = min(k, n_relevant_in_index if n_relevant_in_index is not None else max(sum(rels), 1))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "retrieval_mrr": mrr,
        "retrieval_ndcg": ndcg,
        "retrieval_hit": float(any(rels)),
        "doc_recall": doc_recall,
    }
