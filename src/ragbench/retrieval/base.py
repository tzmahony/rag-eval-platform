from __future__ import annotations

from abc import ABC, abstractmethod

from ragbench.types import Chunk, ScoredChunk


class Retriever(ABC):
    """Index a set of chunks once, then answer top-k queries."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty chunk list")
        self.chunks = list(chunks)
        self._build(self.chunks)

    @abstractmethod
    def _build(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def scores(self, query: str) -> list[float]:
        """One score per indexed chunk (higher = more relevant)."""

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        if not self.chunks:
            raise RuntimeError("Retriever has not been indexed")
        s = self.scores(query)
        order = sorted(range(len(s)), key=lambda i: (-s[i], i))[:k]
        return [ScoredChunk(self.chunks[i], float(s[i]), rank) for rank, i in enumerate(order, 1)]

    def describe(self) -> str:
        return type(self).__name__
