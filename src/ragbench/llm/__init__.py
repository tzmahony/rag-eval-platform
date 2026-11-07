from __future__ import annotations

from typing import Any

from ragbench.llm.base import LLM, Completion
from ragbench.llm.cache import ResponseCache
from ragbench.llm.pricing import DEFAULT_PRICES, Price

__all__ = ["DEFAULT_PRICES", "LLM", "Completion", "Price", "ResponseCache", "make_llm", "parse_model_spec"]


def parse_model_spec(spec: str | dict[str, Any]) -> tuple[str, str, dict[str, Any], str | None]:
    """Accepts ``"provider:model"`` or ``{"id": "provider:model", "name": ..., "params": {...}}``."""
    if isinstance(spec, str):
        spec = {"id": spec}
    model_id = spec["id"]
    if ":" not in model_id:
        raise ValueError(f"Model id must look like 'provider:model', got '{model_id}'")
    provider, model = model_id.split(":", 1)
    return provider, model, dict(spec.get("params", {})), spec.get("name")


def make_llm(
    spec: str | dict[str, Any],
    *,
    cache: ResponseCache | None = None,
    prices: dict[str, Price] | None = None,
) -> LLM:
    provider, model, params, _ = parse_model_spec(spec)
    model_id = f"{provider}:{model}"
    price = (prices or {}).get(model_id)
    kwargs: dict[str, Any] = {"cache": cache, **params}
    if price is not None:
        kwargs["price"] = price
    if provider == "mock":
        from ragbench.llm.mock import MockLLM

        return MockLLM(model, **kwargs)
    if provider == "anthropic":
        from ragbench.llm.anthropic import AnthropicLLM

        return AnthropicLLM(model, **kwargs)
    if provider == "openai":
        from ragbench.llm.openai import OpenAILLM

        return OpenAILLM(model, **kwargs)
    raise ValueError(f"Unknown provider '{provider}'. Options: mock, anthropic, openai")
