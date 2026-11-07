"""Dense retrieval: TF-IDF vectors, LSA embeddings, or external embedding models."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import numpy as np

from ragbench.retrieval.base import Retriever
from ragbench.text import content_tokens
from ragbench.types import Chunk


class Embedder(ABC):
    def fit(self, texts: list[str]) -> None:  # noqa: B027 - optional hook
        """Optional: learn from the corpus (used by corpus-fitted embedders)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...

    def describe(self) -> str:
        return type(self).__name__


class TfidfEmbedder(Embedder):
    def __init__(self, sublinear_tf: bool = True):
        self.sublinear_tf = sublinear_tf

    def fit(self, texts: list[str]) -> None:
        docs = [Counter(content_tokens(t)) for t in texts]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self.vocab)))
        for r, t in enumerate(texts):
            for tok, f in Counter(content_tokens(t)).items():
                j = self.vocab.get(tok)
                if j is not None:
                    m[r, j] = (1 + math.log(f)) if self.sublinear_tf else f
        return m * self.idf

    def describe(self) -> str:
        return "tfidf"


class LSAEmbedder(Embedder):
    """Latent semantic analysis: truncated SVD of the TF-IDF matrix. Fully offline."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._tfidf = TfidfEmbedder()

    def fit(self, texts: list[str]) -> None:
        self._tfidf.fit(texts)
        x = self._tfidf.embed(texts)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        self._proj = vt[: min(self.dim, vt.shape[0])].T

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._tfidf.embed(texts) @ self._proj

    def describe(self) -> str:
        return f"lsa(dim={self.dim})"


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'ragbench[embeddings]' to use sentence-transformers") from e
        self.model_name = model
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs model download
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))

    def describe(self) -> str:
        return f"st({self.model_name})"


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 128):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'ragbench[openai]' to use OpenAI embeddings") from e
        self.model_name, self.batch_size = model, batch_size
        self._client = openai.OpenAI()

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - network
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._client.embeddings.create(model=self.model_name, input=texts[i : i + self.batch_size])
            out.extend(d.embedding for d in resp.data)
        return np.asarray(out)

    def describe(self) -> str:
        return f"openai({self.model_name})"


EMBEDDERS: dict[str, type[Embedder]] = {
    "tfidf": TfidfEmbedder,
    "lsa": LSAEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
    "openai": OpenAIEmbedder,
}


def make_embedder(cfg: dict[str, Any] | str) -> Embedder:
    cfg = {"type": cfg} if isinstance(cfg, str) else dict(cfg)
    kind = cfg.pop("type")
    if kind not in EMBEDDERS:
        raise ValueError(f"Unknown embedder '{kind}'. Options: {sorted(EMBEDDERS)}")
    return EMBEDDERS[kind](**cfg)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class DenseRetriever(Retriever):
    """Cosine similarity between query and chunk embeddings."""

    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder

    def _build(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        self._matrix = _normalize(self.embedder.embed(texts))

    def scores(self, query: str) -> list[float]:
        q = _normalize(self.embedder.embed([query]))[0]
        return (self._matrix @ q).tolist()

    def describe(self) -> str:
        return f"dense({self.embedder.describe()})"
