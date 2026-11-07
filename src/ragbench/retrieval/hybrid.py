from __future__ import annotations

from ragbench.retrieval.base import Retriever
from ragbench.types import Chunk


class HybridRetriever(Retriever):
    """Weighted reciprocal-rank fusion (RRF) over several retrievers."""

    def __init__(self, retrievers: list[Retriever], weights: list[float] | None = None, rrf_k: int = 60):
        super().__init__()
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        if len(self.weights) != len(retrievers):
            raise ValueError("weights must match retrievers")
        self.rrf_k = rrf_k

    def _build(self, chunks: list[Chunk]) -> None:
        for r in self.retrievers:
            r.index(chunks)

    def scores(self, query: str) -> list[float]:
        fused = [0.0] * len(self.chunks)
        for r, w in zip(self.retrievers, self.weights):
            s = r.scores(query)
            order = sorted(range(len(s)), key=lambda i: (-s[i], i))
            for rank, i in enumerate(order, 1):
                fused[i] += w / (self.rrf_k + rank)
        return fused

    def describe(self) -> str:
        parts = ", ".join(f"{r.describe()}*{w:g}" for r, w in zip(self.retrievers, self.weights))
        return f"hybrid[{parts}]"
