from types import SimpleNamespace

import pytest

from ragbench.judges.llm import LLMJudge, parse_json
from ragbench.llm import LLM, Completion, ResponseCache, make_llm
from ragbench.llm.anthropic import AnthropicLLM
from ragbench.llm.openai import OpenAILLM
from ragbench.llm.pricing import Price


class ScriptedLLM(LLM):
    provider = "scripted"

    def __init__(self, replies, **kw):
        super().__init__("m", price=Price(1.0, 2.0), **kw)
        self.replies = list(replies)
        self.calls = 0

    def _complete(self, prompt, system, hints):
        self.calls += 1
        return Completion(self.replies.pop(0), 1000, 500)


def test_cost_and_cache(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    llm = ScriptedLLM(["hello"], cache=cache)
    r1 = llm.generate("p")
    assert r1.cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    r2 = llm.generate("p")
    assert r2.cached and r2.text == "hello" and llm.calls == 1


def test_provider_errors_are_captured():
    class Boom(LLM):
        provider = "boom"

        def _complete(self, *a):
            raise RuntimeError("rate limited")

    r = Boom("x", price=Price(0, 0)).generate("p")
    assert r.error and "rate limited" in r.error and r.text == ""


def test_mock_models_are_deterministic():
    m = make_llm("mock:extractive")
    hints = {"question": "What is the tank capacity?", "contexts": ["The tank capacity is 30 liters. Other text."]}
    a, b = m.generate("p", hints=hints), m.generate("p", hints=hints)
    assert a.text == b.text == "The tank capacity is 30 liters."
    assert a.latency_s == b.latency_s and a.cost_usd > 0
    assert "know" in m.generate("p", hints={"question": "Who won the 1998 World Cup?", "contexts": ["Rice."]}).text


def test_anthropic_adapter_with_fake_client():
    captured = {}

    def create(**req):
        captured.update(req)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=" 30 liters. ")],
            usage=SimpleNamespace(input_tokens=1200, output_tokens=80),
            stop_reason="end_turn",
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    llm = AnthropicLLM("claude-sonnet-5-5", effort="low", client=client)
    r = llm.generate("question", system="sys")
    assert r.text == "30 liters." and r.input_tokens == 1200
    assert r.cost_usd == pytest.approx((1200 * 2 + 80 * 10) / 1e6)
    assert captured["output_config"] == {"effort": "low"} and captured["system"] == "sys"
    assert "temperature" not in captured


def test_openai_adapter_with_fake_client():
    def create(**req):
        assert req["messages"][0]["role"] == "system"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    r = OpenAILLM("gpt-4.1-mini", client=client).generate("hi", system="s")
    assert r.text == "ok" and r.output_tokens == 2 and r.cost_usd > 0


def test_parse_json_tolerates_fences_and_preamble():
    assert parse_json('```json\n{"score": 4}\n```') == {"score": 4}
    assert parse_json('Sure! {"score": 2, "reason": "x"} done') == {"score": 2, "reason": "x"}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_llm_judge_scores():
    llm = ScriptedLLM(
        [
            '{"verdicts": [{"passage": 1, "relevant": true}, {"passage": 2, "relevant": false}]}',
            '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}',
            '{"score": 4, "reason": "minor omission"}',
            "garbage",
        ]
    )
    j = LLMJudge(llm)
    assert j.context_relevance("q", ["c1", "c2"]).score == 0.5
    f = j.faithfulness("A. B.", ["ctx"])
    assert f.score == 0.5 and f.cost_usd > 0
    assert j.answer_correctness("q", "a", "ref", True).score == 0.75
    bad = j.answer_correctness("q", "a", "ref", True)
    assert bad.score is None and "parse" in bad.error
