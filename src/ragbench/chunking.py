"""Chunking strategies. Each turns a Document into a list of Chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ragbench.text import split_sentences
from ragbench.types import Chunk, Document


class Chunker(ABC):
    @abstractmethod
    def split(self, doc: Document) -> list[str]: ...

    def chunk(self, docs: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            for i, text in enumerate(self.split(doc)):
                chunks.append(
                    Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=text, index=i, metadata={"title": doc.title})
                )
        return chunks

    def describe(self) -> str:
        return type(self).__name__


class FixedSizeChunker(Chunker):
    """Sliding window over words. ``size`` and ``overlap`` are in words."""

    def __init__(self, size: int = 120, overlap: int = 20):
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("require size > 0 and 0 <= overlap < size")
        self.size, self.overlap = size, overlap

    def split(self, doc: Document) -> list[str]:
        words = doc.text.split()
        step = self.size - self.overlap
        out = []
        for start in range(0, max(len(words), 1), step):
            window = words[start : start + self.size]
            if window:
                out.append(" ".join(window))
            if start + self.size >= len(words):
                break
        return out

    def describe(self) -> str:
        return f"fixed(size={self.size}, overlap={self.overlap})"


class SentenceChunker(Chunker):
    """Groups ``size`` consecutive sentences, overlapping by ``overlap`` sentences."""

    def __init__(self, size: int = 3, overlap: int = 1):
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("require size > 0 and 0 <= overlap < size")
        self.size, self.overlap = size, overlap

    def split(self, doc: Document) -> list[str]:
        sents = split_sentences(doc.text)
        step = self.size - self.overlap
        out = []
        for start in range(0, max(len(sents), 1), step):
            group = sents[start : start + self.size]
            if group:
                out.append(" ".join(group))
            if start + self.size >= len(sents):
                break
        return out

    def describe(self) -> str:
        return f"sentence(size={self.size}, overlap={self.overlap})"


class ParagraphChunker(Chunker):
    """Splits on blank lines; merges short paragraphs up to ``max_words``."""

    def __init__(self, max_words: int = 200):
        self.max_words = max_words

    def split(self, doc: Document) -> list[str]:
        paras = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
        out: list[str] = []
        buf: list[str] = []
        count = 0
        for p in paras:
            n = len(p.split())
            if buf and count + n > self.max_words:
                out.append("\n\n".join(buf))
                buf, count = [], 0
            buf.append(p)
            count += n
        if buf:
            out.append("\n\n".join(buf))
        return out

    def describe(self) -> str:
        return f"paragraph(max_words={self.max_words})"


class DocumentChunker(Chunker):
    """No chunking: one chunk per document (a useful baseline)."""

    def split(self, doc: Document) -> list[str]:
        return [doc.text]

    def describe(self) -> str:
        return "document"


CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedSizeChunker,
    "sentence": SentenceChunker,
    "paragraph": ParagraphChunker,
    "document": DocumentChunker,
}


def make_chunker(cfg: dict[str, Any] | None) -> Chunker:
    cfg = dict(cfg or {"type": "paragraph"})
    kind = cfg.pop("type", "paragraph")
    if kind not in CHUNKERS:
        raise ValueError(f"Unknown chunker '{kind}'. Options: {sorted(CHUNKERS)}")
    return CHUNKERS[kind](**cfg)
