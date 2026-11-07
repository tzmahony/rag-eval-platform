"""Deterministic offline models so the whole pipeline runs without API keys.

They are *not* language models: they extract sentences from the retrieved
context. Their purpose is to exercise every metric with known behaviour:

* ``mock:extractive``  - returns the single best-matching context sentence.
* ``mock:verbose``     - returns the top three sentences (more tokens, higher cost).
* ``mock:hallucinating`` - like extractive, but appends an unsupported claim to a
  deterministic ~50% of answers, which faithfulness should detect.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ragbench.llm.base import LLM, Completion
from ragbench.text import content_tokens, estimate_tokens, split_sentences

PRESETS: dict[str, dict[str, Any]] = {
    "extractive": {"n_sentences": 1, "hallucination_rate": 0.0, "base_latency_ms": 250, "ms_per_token": 8},
    "verbose": {"n_sentences": 3, "hallucination_rate": 0.0, "base_latency_ms": 600, "ms_per_token": 25},
    "hallucinating": {"n_sentences": 1, "hallucination_rate": 0.5, "base_latency_ms": 180, "ms_per_token": 5},
}

FABRICATIONS = [
    "This rule was introduced in 2011 after a board review.",
    "Exceptions are granted automatically to staff in the Denver office.",
    "The limit doubles during the fourth quarter.",
]

ABSTAIN = "I don't know based on the provided context."


def _unit(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class MockLLM(LLM):
    provider = "mock"

    def __init__(self, model: str = "extractive", **kwargs: Any):
        preset = dict(PRESETS.get(model, PRESETS["extractive"]))
        for k in list(preset):
            if k in kwargs:
                preset[k] = kwargs.pop(k)
        self.min_overlap = kwargs.pop("min_overlap", 0.5)
        super().__init__(model, **kwargs)
        self.cfg = preset

    def _complete(self, prompt: str, system: str | None, hints: dict[str, Any] | None) -> Completion:
        input_tokens = estimate_tokens((system or "") + prompt)
        if not hints or "question" not in hints:
            text = ABSTAIN
        else:
            text = self._answer(hints["question"], hints.get("contexts", []))
        out_tokens = estimate_tokens(text)
        latency = (self.cfg["base_latency_ms"] + self.cfg["ms_per_token"] * out_tokens + 0.2 * input_tokens) / 1000
        # deterministic jitter (+/-15%) so latency percentiles are not degenerate
        latency *= 0.85 + 0.3 * _unit(prompt)
        return Completion(text, input_tokens, out_tokens, "end_turn", simulated_latency_s=latency)

    def _answer(self, question: str, contexts: list[str]) -> str:
        q = set(content_tokens(question))
        if not q:
            return ABSTAIN
        scored = []
        for ci, ctx in enumerate(contexts):
            for si, sent in enumerate(split_sentences(ctx)):
                overlap = len(q & set(content_tokens(sent))) / len(q)
                scored.append((overlap, -ci, -si, sent))
        scored.sort(reverse=True)
        best = [s for s in scored if s[0] >= self.min_overlap][: self.cfg["n_sentences"]]
        if not best:
            return ABSTAIN
        seen: list[str] = []
        for *_, sent in best:
            if sent not in seen:
                seen.append(sent)
        answer = " ".join(seen)
        if _unit("halluc:" + question) < self.cfg["hallucination_rate"]:
            answer += " " + FABRICATIONS[int(_unit("which:" + question) * len(FABRICATIONS)) % len(FABRICATIONS)]
        return answer
