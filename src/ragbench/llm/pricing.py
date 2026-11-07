"""Per-model token pricing, in US dollars per 1M tokens.

Prices change; treat this table as a convenience default and override it per
experiment with the ``pricing:`` block in the config (see README). Claude prices
are first-party API list prices. OpenAI prices are list prices at the time of
writing and should be verified before you publish numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1_000_000


DEFAULT_PRICES: dict[str, Price] = {
    # Anthropic
    "anthropic:claude-fable-5-1": Price(10.00, 50.00),
    "anthropic:claude-opus-5-5": Price(4.00, 20.00),
    "anthropic:claude-opus-5": Price(5.00, 25.00),
    "anthropic:claude-opus-4-8": Price(5.00, 25.00),
    "anthropic:claude-sonnet-5-5": Price(2.00, 10.00),
    "anthropic:claude-sonnet-5": Price(2.00, 10.00),
    "anthropic:claude-sonnet-4-6": Price(3.00, 15.00),
    "anthropic:claude-haiku-5-5": Price(0.10, 0.50),
    "anthropic:claude-haiku-4-5": Price(1.00, 5.00),
    # OpenAI (verify before publishing)
    "openai:gpt-4.1": Price(2.00, 8.00),
    "openai:gpt-4.1-mini": Price(0.40, 1.60),
    "openai:gpt-4o": Price(2.50, 10.00),
    "openai:gpt-4o-mini": Price(0.15, 0.60),
    # Offline mock models: priced so that cost/quality trade-offs show up in reports.
    "mock:extractive": Price(0.50, 1.50),
    "mock:verbose": Price(3.00, 15.00),
    "mock:hallucinating": Price(0.20, 0.60),
}

_warned: set[str] = set()


def price_for(model_id: str, overrides: dict[str, Price] | None = None) -> Price | None:
    if overrides and model_id in overrides:
        return overrides[model_id]
    p = DEFAULT_PRICES.get(model_id)
    if p is None and model_id not in _warned:
        _warned.add(model_id)
        log.warning("No pricing for %s; cost will be reported as 0. Add it under 'pricing:'.", model_id)
    return p
