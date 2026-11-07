from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeResult:
    score: float | None
    details: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class Judge(ABC):
    """Scores the three model-graded qualities. All scores are in [0, 1]."""

    name = "judge"

    @abstractmethod
    def context_relevance(self, question: str, contexts: list[str]) -> JudgeResult:
        """Share of the retrieved context that is relevant to the question."""

    @abstractmethod
    def faithfulness(self, answer: str, contexts: list[str]) -> JudgeResult:
        """Share of the answer's claims that are supported by the retrieved context."""

    @abstractmethod
    def answer_correctness(self, question: str, answer: str, reference: str, answerable: bool) -> JudgeResult:
        """Agreement of the answer with the reference answer."""
