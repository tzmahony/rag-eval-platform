"""OpenAI chat models via the official SDK (``pip install 'ragbench[openai]'``).

Also works with any OpenAI-compatible server (vLLM, Ollama, ...) by passing
``base_url`` in the model params.
"""

from __future__ import annotations

from typing import Any

from ragbench.llm.base import LLM, Completion


class OpenAILLM(LLM):
    provider = "openai"

    def __init__(self, model: str = "gpt-4.1-mini", *, base_url: str | None = None, client: Any = None, **kw: Any):
        super().__init__(model, **kw)
        if client is None:
            try:
                import openai
            except ImportError as e:  # pragma: no cover
                raise ImportError("pip install 'ragbench[openai]' to use OpenAI models") from e
            client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        self._client = client

    def _complete(self, prompt: str, system: str | None, hints: dict[str, Any] | None) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        extra = {k: v for k, v in self.params.items() if k in ("temperature", "top_p", "seed")}
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens, **extra
        )
        choice = resp.choices[0]
        return Completion(
            text=(choice.message.content or "").strip(),
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            stop_reason=choice.finish_reason,
        )
