"""Experiment configuration, execution and aggregation.

An experiment is the cross product of RAG configurations and LLMs, evaluated on
the same labelled dataset so every comparison is *paired* (same questions,
same order), which is far more statistically powerful than comparing two
independent averages.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ragbench import stats
from ragbench.data import load_corpus, load_dataset
from ragbench.judges import Judge, make_judge
from ragbench.llm import LLM, Price, ResponseCache, make_llm, parse_model_spec
from ragbench.metrics import METRICS, QUALITY_METRICS
from ragbench.metrics.answer import token_f1
from ragbench.metrics.performance import percentile
from ragbench.metrics.retrieval import is_relevant, retrieval_metrics
from ragbench.pipeline import RAGConfig, RetrievalIndex, generate_answer
from ragbench.text import is_abstention
from ragbench.types import EvalExample, ScoredChunk

# ----------------------------------------------------------------------------- config


@dataclass
class ExperimentConfig:
    name: str
    corpus: Path
    dataset: Path
    rag: list[RAGConfig]
    llms: list[str | dict[str, Any]]
    judge: dict[str, Any] = field(default_factory=lambda: {"type": "heuristic"})
    description: str = ""
    output_dir: Path = Path("runs")
    limit: int | None = None
    seed: int = 0
    concurrency: int = 4
    cache: str | bool = True
    baseline: str | None = None
    relevance_threshold: float = 0.6
    alpha: float = 0.05
    bootstrap_resamples: int = 2000
    pricing: dict[str, Price] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        return cls.from_dict(data, base_dir=path.parent)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> ExperimentConfig:
        base_dir = base_dir or Path.cwd()
        for key in ("name", "corpus", "dataset", "rag", "llms"):
            if key not in data:
                raise ValueError(f"Experiment config is missing '{key}'")
        rag = [RAGConfig.from_dict(r) for r in data["rag"]]
        names = [r.name for r in rag]
        if len(names) != len(set(names)):
            raise ValueError("RAG config names must be unique")
        pricing = {mid: Price(float(p["input"]), float(p["output"])) for mid, p in (data.get("pricing") or {}).items()}
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            corpus=(base_dir / data["corpus"]).resolve(),
            dataset=(base_dir / data["dataset"]).resolve(),
            rag=rag,
            llms=list(data["llms"]),
            judge=data.get("judge") or {"type": "heuristic"},
            output_dir=Path(data.get("output_dir", "runs")),
            limit=data.get("limit"),
            seed=int(data.get("seed", 0)),
            concurrency=int(data.get("concurrency", 4)),
            cache=data.get("cache", True),
            baseline=data.get("baseline"),
            relevance_threshold=float(data.get("relevance_threshold", 0.6)),
            alpha=float(data.get("alpha", 0.05)),
            bootstrap_resamples=int(data.get("bootstrap_resamples", 2000)),
            pricing=pricing,
            raw=data,
        )


def llm_label(spec: str | dict[str, Any]) -> str:
    provider, model, _, name = parse_model_spec(spec)
    return name or f"{provider}:{model}"


def config_label(rag_name: str, llm_name: str) -> str:
    return f"{rag_name} / {llm_name}"


# ----------------------------------------------------------------------------- running


@dataclass
class RunResult:
    config: ExperimentConfig
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    output_dir: Path | None = None


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


def _hits_to_json(hits: list[ScoredChunk], ex: EvalExample, threshold: float) -> list[dict[str, Any]]:
    return [
        {
            "rank": h.rank,
            "chunk_id": h.chunk.id,
            "doc_id": h.chunk.doc_id,
            "score": round(h.score, 6),
            "relevant": is_relevant(h.chunk, ex, threshold) if ex.answerable else False,
            "text": h.chunk.text,
        }
        for h in hits
    ]


def run_experiment(cfg: ExperimentConfig, *, quiet: bool = False, write: bool = True) -> RunResult:
    documents = load_corpus(cfg.corpus)
    examples = load_dataset(cfg.dataset)
    if cfg.limit:
        examples = examples[: cfg.limit]
    _log(f"[ragbench] {cfg.name}: {len(documents)} documents, {len(examples)} examples", quiet)

    cache = None
    if cfg.cache:
        cache_path = Path(cfg.cache) if isinstance(cfg.cache, str) else Path(".ragbench_cache/responses.sqlite")
        cache = ResponseCache(cache_path)

    judge: Judge = make_judge(cfg.judge, cache=cache, prices=cfg.pricing)
    llms: list[tuple[str, LLM]] = [(llm_label(s), make_llm(s, cache=cache, prices=cfg.pricing)) for s in cfg.llms]
    _log(f"[ragbench] judge: {judge.name}; models: {', '.join(n for n, _ in llms)}", quiet)

    records: list[dict[str, Any]] = []
    index_info: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        for rag in cfg.rag:
            index = RetrievalIndex(rag, documents)
            index_info[rag.name] = {
                "description": index.describe(),
                "n_chunks": len(index.chunks),
                "avg_chunk_words": sum(len(c.text.split()) for c in index.chunks) / len(index.chunks),
                "build_seconds": index.build_seconds,
            }
            _log(f"[ragbench] indexed '{rag.name}': {index.describe()} ({len(index.chunks)} chunks)", quiet)

            # Retrieval and context relevance depend only on the RAG config, so they are
            # computed once and shared by every LLM (saves judge cost, keeps pairs aligned).
            retrieval: dict[str, dict[str, Any]] = {}
            for ex in examples:
                hits, secs = index.retrieve(ex.question)
                n_rel = sum(is_relevant(c, ex, cfg.relevance_threshold) for c in index.chunks) if ex.answerable else 0
                retrieval[ex.id] = {
                    "hits": hits,
                    "latency_ms": secs * 1000,
                    "metrics": retrieval_metrics(
                        hits, ex, k=rag.k, n_relevant_in_index=n_rel, threshold=cfg.relevance_threshold
                    ),
                }
            ctx_results = list(
                pool.map(
                    lambda ex, retrieval=retrieval: judge.context_relevance(
                        ex.question, [h.chunk.text for h in retrieval[ex.id]["hits"]]
                    ),
                    examples,
                )
            )
            for ex, cr in zip(examples, ctx_results):
                retrieval[ex.id]["context"] = cr

            for llm_name, llm in llms:
                label = config_label(rag.name, llm_name)
                _log(f"[ragbench] running {label}", quiet)

                def evaluate(
                    ex: EvalExample, rag=rag, llm=llm, llm_name=llm_name, label=label, retrieval=retrieval
                ) -> dict[str, Any]:
                    r = retrieval[ex.id]
                    hits = r["hits"]
                    contexts = [h.chunk.text for h in hits]
                    _, resp = generate_answer(rag, llm, ex.question, hits)
                    rec: dict[str, Any] = {
                        "config": label,
                        "rag": rag.name,
                        "llm": llm_name,
                        "model_id": llm.id,
                        "example_id": ex.id,
                        "category": ex.metadata.get("category", "all"),
                        "answerable": ex.answerable,
                        "question": ex.question,
                        "reference_answer": ex.reference_answer,
                        "answer": resp.text,
                        "retrieved": _hits_to_json(hits, ex, cfg.relevance_threshold),
                        **{k: v for k, v in r["metrics"].items()},
                        "context_relevance": r["context"].score,
                        "latency_retrieval_ms": r["latency_ms"],
                        "latency_generation_ms": resp.latency_s * 1000,
                        "latency_total_ms": r["latency_ms"] + resp.latency_s * 1000,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd,
                        "cached": resp.cached,
                        "stop_reason": resp.stop_reason,
                        "error": resp.error,
                    }
                    judge_cost = r["context"].cost_usd / max(1, len(llms))
                    if resp.error:
                        rec.update(faithfulness=None, answer_correctness=None, token_f1=None, abstained=None)
                        rec["judge"] = {}
                    else:
                        faith = judge.faithfulness(resp.text, contexts)
                        corr = judge.answer_correctness(ex.question, resp.text, ex.reference_answer, ex.answerable)
                        judge_cost += faith.cost_usd + corr.cost_usd
                        rec.update(
                            faithfulness=faith.score,
                            answer_correctness=corr.score,
                            token_f1=token_f1(resp.text, ex.reference_answer) if ex.answerable else None,
                            abstained=float(is_abstention(resp.text)),
                        )
                        rec["judge"] = {
                            "context_relevance": r["context"].details,
                            "faithfulness": faith.details,
                            "answer_correctness": corr.details,
                            "errors": [e for e in (r["context"].error, faith.error, corr.error) if e],
                        }
                    rec["judge_cost_usd"] = judge_cost
                    return rec

                records.extend(pool.map(evaluate, examples))

    if cache is not None:
        cache.close()

    summary = summarize(records, cfg, index_info=index_info, n_examples=len(examples), judge_name=judge.name)
    result = RunResult(cfg, records, summary)
    if write:
        from ragbench.report import write_outputs

        result.output_dir = write_outputs(result)
        _log(f"[ragbench] wrote results to {result.output_dir}", quiet)
    return result


# ----------------------------------------------------------------------------- aggregation


def summarize(
    records: list[dict[str, Any]],
    cfg: ExperimentConfig,
    *,
    index_info: dict[str, dict[str, Any]] | None = None,
    n_examples: int | None = None,
    judge_name: str = "",
) -> dict[str, Any]:
    labels = list(dict.fromkeys(r["config"] for r in records))
    by_cfg = {lab: [r for r in records if r["config"] == lab] for lab in labels}

    configs: dict[str, Any] = {}
    for lab in labels:
        rows = by_cfg[lab]
        metrics = {
            m.key: stats.bootstrap_ci(
                [r.get(m.key) for r in rows], n_resamples=cfg.bootstrap_resamples, seed=cfg.seed
            ).to_dict()
            for m in METRICS
        }
        lat = [r["latency_total_ms"] for r in rows if not r.get("error")]
        answerable = [r for r in rows if r["answerable"] and r.get("abstained") is not None]
        unanswerable = [r for r in rows if not r["answerable"] and r.get("abstained") is not None]
        categories: dict[str, Any] = {}
        for cat in sorted({r["category"] for r in rows}):
            vals = [
                r["answer_correctness"] for r in rows if r["category"] == cat and r["answer_correctness"] is not None
            ]
            categories[cat] = {"answer_correctness": sum(vals) / len(vals) if vals else None, "n": len(vals)}
        configs[lab] = {
            "rag": rows[0]["rag"],
            "llm": rows[0]["llm"],
            "model_id": rows[0]["model_id"],
            "n": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "metrics": metrics,
            "latency_p50_ms": percentile(lat, 50),
            "latency_p95_ms": percentile(lat, 95),
            "total_cost_usd": sum(r["cost_usd"] for r in rows),
            "judge_cost_usd": sum(r.get("judge_cost_usd", 0.0) for r in rows),
            "false_abstention_rate": (sum(r["abstained"] for r in answerable) / len(answerable))
            if answerable
            else None,
            "correct_abstention_rate": (sum(r["abstained"] for r in unanswerable) / len(unanswerable))
            if unanswerable
            else None,
            "by_category": categories,
        }

    baseline = cfg.baseline if cfg.baseline in configs else (labels[0] if labels else None)
    comparisons = compare(by_cfg, baseline, cfg) if baseline else []
    return {
        "name": cfg.name,
        "description": cfg.description,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_examples": n_examples if n_examples is not None else len({r["example_id"] for r in records}),
        "judge": judge_name,
        "baseline": baseline,
        "alpha": cfg.alpha,
        "indexes": index_info or {},
        "configs": configs,
        "comparisons": comparisons,
        "pareto": pareto_front(configs),
    }


COMPARE_METRICS = QUALITY_METRICS + ["latency_total_ms", "cost_usd"]
# These depend only on the RAG configuration, not on the LLM.
RAG_SCOPED = {"retrieval_precision", "retrieval_recall", "retrieval_mrr", "context_relevance"}


def compare(by_cfg: dict[str, list[dict[str, Any]]], baseline: str, cfg: ExperimentConfig) -> list[dict[str, Any]]:
    """Paired tests of every config against the baseline.

    Quality metrics form one Holm-Bonferroni family. RAG-scoped metrics are
    tested once per RAG configuration (repeating an identical test per LLM would
    needlessly inflate the family); the result is shown on every config sharing
    that RAG setup. Latency and cost are reported uncorrected - those
    differences are typically large and are not the hypotheses of interest.
    """
    base_rag = by_cfg[baseline][0]["rag"]
    base = {m: {r["example_id"]: r.get(m) for r in by_cfg[baseline]} for m in COMPARE_METRICS}
    rows: list[dict[str, Any]] = []
    shared: dict[tuple[str, str], dict[str, Any]] = {}
    for lab, recs in by_cfg.items():
        if lab == baseline:
            continue
        rag = recs[0]["rag"]
        for m in COMPARE_METRICS:
            if m in RAG_SCOPED:
                if rag == base_rag:
                    continue  # identical retrieval: nothing to test
                if (rag, m) in shared:
                    rows.append({**shared[(rag, m)], "config": lab, "family_member": False})
                    continue
            cand = {r["example_id"]: r.get(m) for r in recs}
            a, b = stats.paired_values(cand, base[m])
            if a.size == 0:
                continue
            diff = stats.paired_bootstrap_diff(a, b, n_resamples=cfg.bootstrap_resamples, seed=cfg.seed)
            p = stats.paired_permutation_test(a, b, seed=cfg.seed)
            row = {
                "config": lab,
                "rag": rag,
                "metric": m,
                "scope": "rag" if m in RAG_SCOPED else "config",
                "n": int(a.size),
                "delta": diff.mean,
                "lo": diff.lo,
                "hi": diff.hi,
                "p": p,
                "family_member": m in QUALITY_METRICS,
            }
            rows.append(row)
            if m in RAG_SCOPED:
                shared[(rag, m)] = row

    family = [i for i, r in enumerate(rows) if r["family_member"]]
    adjusted = dict(zip(family, stats.holm_bonferroni([rows[i]["p"] for i in family])))
    for i, r in enumerate(rows):
        r["p_adj"] = adjusted.get(i, r["p"])
    for r in rows:  # propagate shared RAG-scoped results
        if r["scope"] == "rag" and not r["family_member"]:
            r["p_adj"] = shared[(r["rag"], r["metric"])]["p_adj"]
        r["significant"] = r["p_adj"] < cfg.alpha
    return rows


def pareto_front(configs: dict[str, Any]) -> list[str]:
    """Configs not dominated on (answer correctness up, cost down, p95 latency down)."""
    pts = {}
    for lab, c in configs.items():
        acc = c["metrics"]["answer_correctness"]["mean"]
        cost = c["metrics"]["cost_usd"]["mean"]
        lat = c["latency_p95_ms"]
        if None not in (acc, cost, lat):
            pts[lab] = (acc, -cost, -lat)
    front = []
    for lab, p in pts.items():
        dominated = any(
            all(q[j] >= p[j] for j in range(3)) and any(q[j] > p[j] for j in range(3))
            for o, q in pts.items()
            if o != lab
        )
        if not dominated:
            front.append(lab)
    return front


def load_run(run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = Path(run_dir)
    records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    summary = json.loads((run_dir / "summary.json").read_text())
    return records, summary
