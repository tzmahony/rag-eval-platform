from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ragbench.llm.cache import ResponseCache
from ragbench.llm.pricing import Price, price_for
from ragbench.types import LLMResponse


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None
    simulated_latency_s: float | None = None  # mock models report a modelled latency


class LLM(ABC):
    """Provider-agnostic chat model wrapper that measures latency, tokens and cost.

    Subclasses implement :meth:`_complete`. ``hints`` carries structured data
    (question, contexts) that real providers ignore but offline mocks use.
    """

    provider: str = "base"

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 2048,
        price: Price | None = None,
        cache: ResponseCache | None = None,
        **params: Any,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.params = params
        self.cache = cache
        self._price = price if price is not None else price_for(self.id)

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def price(self) -> Price | None:
        return self._price

    @abstractmethod
    def _complete(self, prompt: str, system: str | None, hints: dict[str, Any] | None) -> Completion: ...

    def generate(self, prompt: str, *, system: str | None = None, hints: dict[str, Any] | None = None) -> LLMResponse:
        key = None
        if self.cache is not None:
            key = ResponseCache.key(
                model=self.id, system=system, prompt=prompt, max_tokens=self.max_tokens, params=self.params
            )
            hit = self.cache.get(key)
            if hit is not None:
                hit.cached = True
                return hit

        start = time.perf_counter()
        try:
            c = self._complete(prompt, system, hints)
        except Exception as e:  # provider errors are recorded, not fatal
            return LLMResponse(
                text="",
                model=self.id,
                input_tokens=0,
                output_tokens=0,
                latency_s=time.perf_counter() - start,
                error=f"{type(e).__name__}: {e}",
            )
        latency = c.simulated_latency_s if c.simulated_latency_s is not None else time.perf_counter() - start
        cost = self._price.cost(c.input_tokens, c.output_tokens) if self._price else 0.0
        resp = LLMResponse(
            text=c.text,
            model=self.id,
            input_tokens=c.input_tokens,
            output_tokens=c.output_tokens,
            latency_s=latency,
            cost_usd=cost,
            stop_reason=c.stop_reason,
        )
        if self.cache is not None and key is not None and resp.error is None:
            self.cache.put(key, resp)
        return resp
