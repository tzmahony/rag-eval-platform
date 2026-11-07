"""Deterministic lexical judge.

Free, fast and reproducible, which makes it ideal for CI and for smoke-testing
configurations. It under-rewards paraphrase, so use :class:`LLMJudge` for final
numbers, and use ``ragbench calibrate`` to check how well either judge agrees
with human labels.
"""

from __future__ import annotations

from ragbench.judges.base import Judge, JudgeResult
from ragbench.text import content_tokens, is_abstention, is_number, split_sentences


class HeuristicJudge(Judge):
    name = "heuristic"

    def __init__(self, relevance_threshold: float = 0.3, support_threshold: float = 0.75):
        self.relevance_threshold = relevance_threshold
        self.support_threshold = support_threshold

    def context_relevance(self, question: str, contexts: list[str]) -> JudgeResult:
        q = set(content_tokens(question))
        if not contexts:
            return JudgeResult(0.0, {"reason": "no context"})
        per_chunk = []
        total = relevant = 0
        for ctx in contexts:
            sents = split_sentences(ctx)
            rel = 0
            for s in sents:
                overlap = len(q & set(content_tokens(s)))
                if q and overlap >= 1 and overlap / len(q) >= self.relevance_threshold:
                    rel += 1
            per_chunk.append(rel / len(sents) if sents else 0.0)
            total += len(sents)
            relevant += rel
        score = relevant / total if total else 0.0
        return JudgeResult(score, {"per_chunk": per_chunk, "relevant_sentences": relevant, "sentences": total})

    def faithfulness(self, answer: str, contexts: list[str]) -> JudgeResult:
        if is_abstention(answer):
            return JudgeResult(1.0, {"abstained": True, "claims": []})
        claims = split_sentences(answer)
        ctx_tokens = set(content_tokens(" ".join(contexts)))
        verdicts = []
        for claim in claims:
            toks = content_tokens(claim)
            if not toks:
                continue
            support = sum(t in ctx_tokens for t in toks) / len(toks)
            numbers_ok = all(t in ctx_tokens for t in toks if is_number(t))
            verdicts.append(
                {
                    "claim": claim,
                    "support": round(support, 3),
                    "supported": numbers_ok and support >= self.support_threshold,
                }
            )
        if not verdicts:
            return JudgeResult(1.0, {"claims": []})
        score = sum(v["supported"] for v in verdicts) / len(verdicts)
        return JudgeResult(score, {"claims": verdicts})

    def answer_correctness(self, question: str, answer: str, reference: str, answerable: bool) -> JudgeResult:
        abstained = is_abstention(answer)
        if not answerable:
            return JudgeResult(float(abstained), {"expected_abstention": True, "abstained": abstained})
        if abstained or not answer.strip():
            return JudgeResult(0.0, {"abstained": True})
        ref = content_tokens(reference)
        ans = set(content_tokens(answer))
        if not ref:
            return JudgeResult(None, error="empty reference")
        recall = sum(t in ans for t in ref) / len(ref)
        ref_numbers = [t for t in ref if is_number(t)]
        missing = [n for n in ref_numbers if n not in ans]
        # Tolerant scale: ~70% reference-token recall counts as fully correct
        # (references often contain extra explanation the answer needn't repeat).
        score = min(1.0, recall / 0.7)
        # Key facts (numbers) are mandatory: missing any caps the score at 0.5.
        if missing:
            score = min(score, 0.5)
        return JudgeResult(score, {"reference_recall": round(recall, 3), "missing_numbers": missing})
