"""A RAG pipeline = chunker + retriever + k + prompt + LLM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ragbench.chunking import Chunker, make_chunker
from ragbench.llm.base import LLM
from ragbench.retrieval import Retriever, make_retriever
from ragbench.types import Chunk, Document, LLMResponse, ScoredChunk

DEFAULT_SYSTEM = (
    "You answer questions using only the provided context. If the context does not contain the answer, "
    'reply exactly "I don\'t know." Do not use outside knowledge. Answer concisely in one to three sentences.'
)

DEFAULT_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

DEFAULT_PASSAGE_FORMAT = "[{rank}] (source: {doc_id})\n{text}"


@dataclass
class RAGConfig:
    name: str
    chunker: dict[str, Any] = field(default_factory=lambda: {"type": "paragraph"})
    retriever: dict[str, Any] = field(default_factory=lambda: {"type": "bm25"})
    k: int = 4
    system_prompt: str = DEFAULT_SYSTEM
    prompt_template: str = DEFAULT_TEMPLATE
    passage_format: str = DEFAULT_PASSAGE_FORMAT

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RAGConfig:
        d = dict(d)
        retr = dict(d.get("retriever", {"type": "bm25"}))
        k = d.pop("k", retr.pop("k", 4))
        d["retriever"] = retr
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown RAG config keys for '{d.get('name')}': {sorted(unknown)}")
        return cls(k=int(k), **d)


class RetrievalIndex:
    """Chunked + indexed corpus for one RAG configuration (shared across LLMs)."""

    def __init__(self, config: RAGConfig, documents: list[Document]):
        self.config = config
        self.chunker: Chunker = make_chunker(config.chunker)
        self.retriever: Retriever = make_retriever(config.retriever)
        start = time.perf_counter()
        self.chunks: list[Chunk] = self.chunker.chunk(documents)
        self.retriever.index(self.chunks)
        self.build_seconds = time.perf_counter() - start

    def retrieve(self, question: str) -> tuple[list[ScoredChunk], float]:
        start = time.perf_counter()
        hits = self.retriever.retrieve(question, self.config.k)
        return hits, time.perf_counter() - start

    def describe(self) -> str:
        return f"{self.chunker.describe()} · {self.retriever.describe()} · k={self.config.k}"


def build_prompt(config: RAGConfig, question: str, hits: list[ScoredChunk]) -> str:
    context = "\n\n".join(
        config.passage_format.format(rank=h.rank, doc_id=h.chunk.doc_id, text=h.chunk.text, score=h.score) for h in hits
    )
    return config.prompt_template.format(context=context, question=question)


def generate_answer(config: RAGConfig, llm: LLM, question: str, hits: list[ScoredChunk]) -> tuple[str, LLMResponse]:
    prompt = build_prompt(config, question, hits)
    resp = llm.generate(
        prompt, system=config.system_prompt, hints={"question": question, "contexts": [h.chunk.text for h in hits]}
    )
    return prompt, resp
