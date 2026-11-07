"""LLM-as-judge implementations of context relevance, faithfulness and correctness.

Prompts ask for strict JSON. Per-chunk / per-claim verdicts (rather than a single
holistic 1-10 score) make the judgments auditable: the verdicts are stored in the
results so you can inspect *why* an answer was marked unfaithful.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ragbench.judges.base import Judge, JudgeResult
from ragbench.llm.base import LLM
from ragbench.text import is_abstention

JUDGE_SYSTEM = (
    "You are a meticulous evaluator of retrieval-augmented generation systems. "
    "Follow the instructions exactly and reply with a single JSON object and nothing else."
)

CONTEXT_RELEVANCE_PROMPT = """Decide, for each retrieved passage, whether it contains information that helps answer the question.

Question: {question}

Passages:
{passages}

Reply with JSON: {{"verdicts": [{{"passage": <number>, "relevant": true|false, "reason": "<short>"}}, ...]}}
Include exactly one verdict per passage."""

FAITHFULNESS_PROMPT = """Break the answer into its individual factual claims. For each claim, decide whether it is fully supported by the context. A claim is supported only if the context states it or it follows directly from the context; general knowledge does not count.

Context:
{context}

Answer:
{answer}

Reply with JSON: {{"claims": [{{"claim": "<claim>", "supported": true|false, "reason": "<short>"}}, ...]}}
If the answer makes no factual claims (for example it declines to answer), reply {{"claims": []}}."""

CORRECTNESS_PROMPT = """Grade the candidate answer against the reference answer.

Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Scoring rubric:
5 - Fully correct: contains every key fact in the reference and nothing that contradicts it.
4 - Correct but missing a minor detail.
3 - Partially correct: some key facts right, some missing or wrong.
2 - Mostly incorrect, with a fragment of the right information.
1 - Incorrect, contradicts the reference, or fails to answer.
{abstention_rule}
Reply with JSON: {{"score": <1-5>, "reason": "<short>"}}"""

ABSTAIN_RULE_UNANSWERABLE = (
    "The question cannot be answered from the available documents. A candidate that says it does not know "
    "or that the information is not available scores 5; any candidate that asserts a specific answer scores 1."
)
ABSTAIN_RULE_ANSWERABLE = "A candidate that declines to answer scores 1."

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text`` (tolerates code fences and preambles)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1 :] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError(f"No JSON object in judge output: {text[:200]!r}") from None
        return json.loads(m.group(0))


def _truthy(v: Any) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"))


class LLMJudge(Judge):
    name = "llm"

    def __init__(self, llm: LLM, max_context_chars: int = 24_000):
        self.llm = llm
        self.max_context_chars = max_context_chars
        self.name = f"llm({llm.id})"

    def _ask(self, prompt: str) -> tuple[dict[str, Any] | None, JudgeResult]:
        resp = self.llm.generate(prompt, system=JUDGE_SYSTEM)
        base = JudgeResult(
            None, cost_usd=resp.cost_usd, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens
        )
        if resp.error:
            base.error = resp.error
            return None, base
        try:
            return parse_json(resp.text), base
        except (ValueError, json.JSONDecodeError) as e:
            base.error = f"judge parse error: {e}"
            return None, base

    def context_relevance(self, question: str, contexts: list[str]) -> JudgeResult:
        if not contexts:
            return JudgeResult(0.0, {"reason": "no context"})
        passages = "\n\n".join(
            f"[{i}] {c[: self.max_context_chars // len(contexts)]}" for i, c in enumerate(contexts, 1)
        )
        data, res = self._ask(CONTEXT_RELEVANCE_PROMPT.format(question=question, passages=passages))
        if data is None:
            return res
        verdicts = data.get("verdicts", [])
        by_idx = {}
        for v in verdicts:
            try:
                by_idx[int(v.get("passage"))] = _truthy(v.get("relevant"))
            except (TypeError, ValueError):
                continue
        flags = [by_idx.get(i, False) for i in range(1, len(contexts) + 1)]
        res.score = sum(flags) / len(flags)
        res.details = {"verdicts": verdicts}
        return res

    def faithfulness(self, answer: str, contexts: list[str]) -> JudgeResult:
        if not answer.strip():
            return JudgeResult(None, error="empty answer")
        if is_abstention(answer) and len(answer) < 160:
            return JudgeResult(1.0, {"abstained": True, "claims": []})
        context = "\n\n".join(contexts)[: self.max_context_chars]
        data, res = self._ask(FAITHFULNESS_PROMPT.format(context=context, answer=answer))
        if data is None:
            return res
        claims = data.get("claims", [])
        res.score = (sum(_truthy(c.get("supported")) for c in claims) / len(claims)) if claims else 1.0
        res.details = {"claims": claims}
        return res

    def answer_correctness(self, question: str, answer: str, reference: str, answerable: bool) -> JudgeResult:
        rule = ABSTAIN_RULE_ANSWERABLE if answerable else ABSTAIN_RULE_UNANSWERABLE
        data, res = self._ask(
            CORRECTNESS_PROMPT.format(
                question=question, reference=reference, answer=answer or "(empty)", abstention_rule=rule
            )
        )
        if data is None:
            return res
        try:
            raw = float(data["score"])
        except (KeyError, TypeError, ValueError):
            res.error = f"judge returned no numeric score: {data!r}"[:300]
            return res
        raw = min(5.0, max(1.0, raw))
        res.score = (raw - 1) / 4
        res.details = {"raw_score": raw, "reason": data.get("reason", "")}
        return res
