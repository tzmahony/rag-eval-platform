"""Claude models via the official Anthropic Python SDK (``pip install 'ragbench[anthropic]'``).

Credentials resolve the SDK's usual way (``ANTHROPIC_API_KEY`` etc.).

Two deliberate choices for evaluation:
* No server-side model fallback is configured. A fallback would silently swap
  the model under test; refusals are recorded via ``stop_reason`` instead.
* ``effort`` (``low``..``max``) is exposed because it is the main cost/quality
  lever on current Claude models. Thinking tokens are billed as output tokens
  and so are included in reported cost.
"""

from __future__ import annotations

from typing import Any

from ragbench.llm.base import LLM, Completion


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", *, effort: str | None = None, client: Any = None, **kw: Any):
        super().__init__(model, **kw)
        self.effort = effort
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError("pip install 'ragbench[anthropic]' to use Claude models") from e
            client = anthropic.Anthropic()
        self._client = client

    def _complete(self, prompt: str, system: str | None, hints: dict[str, Any] | None) -> Completion:
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            req["system"] = system
        if self.effort:
            req["output_config"] = {"effort": self.effort}
        resp = self._client.messages.create(**req)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return Completion(
            text=text.strip(),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason,
        )
