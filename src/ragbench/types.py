"""Core data types shared across the framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    rank: int  # 1-based


@dataclass(frozen=True)
class EvalExample:
    """One labelled evaluation item.

    ``relevant_doc_ids`` drives document-level retrieval labels. ``evidence``
    (optional) lists verbatim passages that must be retrieved; when present it
    gives finer, chunk-level retrieval labels. ``answerable=False`` marks
    questions whose correct behaviour is to abstain.
    """

    id: str
    question: str
    reference_answer: str
    relevant_doc_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    answerable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float = 0.0
    cached: bool = False
    stop_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": self.latency_s,
            "cost_usd": self.cost_usd,
            "cached": self.cached,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMResponse:
        return cls(**d)
