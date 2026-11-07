from __future__ import annotations

from typing import Any

from ragbench.judges.base import Judge, JudgeResult
from ragbench.judges.heuristic import HeuristicJudge
from ragbench.judges.llm import LLMJudge

__all__ = ["HeuristicJudge", "Judge", "JudgeResult", "LLMJudge", "make_judge"]


def make_judge(cfg: dict[str, Any] | None, **llm_kwargs: Any) -> Judge:
    cfg = dict(cfg or {"type": "heuristic"})
    kind = cfg.pop("type", "heuristic")
    if kind == "heuristic":
        return HeuristicJudge(**cfg)
    if kind == "llm":
        from ragbench.llm import make_llm

        model = cfg.pop("model")
        spec = dict(model) if isinstance(model, dict) else {"id": model, "params": cfg.pop("params", {})}
        # Judges on reasoning models need headroom: thinking counts toward max_tokens.
        spec["params"] = {"max_tokens": 4096, **spec.get("params", {})}
        return LLMJudge(make_llm(spec, **llm_kwargs), **cfg)
    raise ValueError(f"Unknown judge '{kind}'. Options: heuristic, llm")
