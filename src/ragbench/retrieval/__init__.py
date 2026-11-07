from __future__ import annotations

from typing import Any

from ragbench.retrieval.base import Retriever
from ragbench.retrieval.bm25 import BM25Retriever
from ragbench.retrieval.dense import DenseRetriever, Embedder, make_embedder
from ragbench.retrieval.hybrid import HybridRetriever

__all__ = ["BM25Retriever", "DenseRetriever", "Embedder", "HybridRetriever", "Retriever", "make_retriever"]


def make_retriever(cfg: dict[str, Any] | None) -> Retriever:
    """Build a retriever from config, e.g. ``{"type": "bm25"}``,
    ``{"type": "dense", "embedder": "lsa"}`` or
    ``{"type": "hybrid", "retrievers": [...], "weights": [...]}``."""
    cfg = dict(cfg or {"type": "bm25"})
    cfg.pop("k", None)  # k is a pipeline setting
    kind = cfg.pop("type", "bm25")
    if kind == "bm25":
        return BM25Retriever(**cfg)
    if kind in ("tfidf", "lsa"):
        return DenseRetriever(make_embedder({"type": kind, **cfg}))
    if kind == "dense":
        return DenseRetriever(make_embedder(cfg.pop("embedder", "lsa")))
    if kind == "hybrid":
        subs = [make_retriever(c) for c in cfg.pop("retrievers")]
        return HybridRetriever(subs, **cfg)
    raise ValueError(f"Unknown retriever '{kind}'. Options: bm25, tfidf, lsa, dense, hybrid")
