from __future__ import annotations

import math
from collections import Counter

from ragbench.retrieval.base import Retriever
from ragbench.text import content_tokens
from ragbench.types import Chunk


class BM25Retriever(Retriever):
    """Okapi BM25 over stemmed, stopword-filtered tokens."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__()
        self.k1, self.b = k1, b

    def _build(self, chunks: list[Chunk]) -> None:
        self._tf = [Counter(content_tokens(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len) or 1.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = [t for t in content_tokens(query) if t in self._idf]
        out = []
        for tf, dl in zip(self._tf, self._len):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * dl / self._avg)
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def describe(self) -> str:
        return f"bm25(k1={self.k1}, b={self.b})"
