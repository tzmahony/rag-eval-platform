import json

import pytest

from ragbench.cli import main
from ragbench.experiment import ExperimentConfig, pareto_front, run_experiment


def _config(examples_dir, tmp_path, **over):
    data = {
        "name": "test",
        "corpus": str(examples_dir / "corpus"),
        "dataset": str(examples_dir / "dataset.jsonl"),
        "output_dir": str(tmp_path / "runs"),
        "cache": False,
        "rag": [
            {
                "name": "bm25",
                "chunker": {"type": "sentence", "size": 3, "overlap": 1},
                "retriever": {"type": "bm25"},
                "k": 4,
            },
            {
                "name": "tfidf-k1",
                "chunker": {"type": "sentence", "size": 2, "overlap": 0},
                "retriever": {"type": "tfidf"},
                "k": 1,
            },
        ],
        "llms": ["mock:extractive", "mock:hallucinating"],
        "bootstrap_resamples": 300,
    }
    data.update(over)
    return ExperimentConfig.from_dict(data)


def test_end_to_end_run_writes_all_outputs(examples_dir, tmp_path):
    res = run_experiment(_config(examples_dir, tmp_path), quiet=True)
    out = res.output_dir
    for f in ("results.jsonl", "summary.json", "summary.csv", "report.md", "report.html", "config.yaml"):
        assert (out / f).exists(), f
    n_examples = res.summary["n_examples"]
    assert len(res.records) == n_examples * 4
    s = res.summary["configs"]
    assert set(s) == {
        "bm25 / mock:extractive",
        "bm25 / mock:hallucinating",
        "tfidf-k1 / mock:extractive",
        "tfidf-k1 / mock:hallucinating",
    }
    # the hallucinating mock must be measurably less faithful
    assert (
        s["bm25 / mock:hallucinating"]["metrics"]["faithfulness"]["mean"]
        < s["bm25 / mock:extractive"]["metrics"]["faithfulness"]["mean"]
    )
    # identical retrieval across LLMs
    assert (
        s["bm25 / mock:hallucinating"]["metrics"]["retrieval_recall"]
        == s["bm25 / mock:extractive"]["metrics"]["retrieval_recall"]
    )
    # k=1 retrieves less evidence than k=4
    assert (
        s["tfidf-k1 / mock:extractive"]["metrics"]["retrieval_recall"]["mean"]
        < s["bm25 / mock:extractive"]["metrics"]["retrieval_recall"]["mean"]
    )
    rec = json.loads((out / "results.jsonl").read_text().splitlines()[0])
    for key in ("answer", "retrieved", "faithfulness", "latency_total_ms", "cost_usd", "judge"):
        assert key in rec
    assert "## Leaderboard" in (out / "report.md").read_text()


def test_comparisons_and_rag_scoped_tests(examples_dir, tmp_path):
    res = run_experiment(_config(examples_dir, tmp_path), quiet=True, write=False)
    comps = res.summary["comparisons"]
    assert res.summary["baseline"] == "bm25 / mock:extractive"
    # no retrieval tests for configs sharing the baseline's RAG setup
    assert not [c for c in comps if c["config"] == "bm25 / mock:hallucinating" and c["metric"] == "retrieval_recall"]
    shared = [c for c in comps if c["metric"] == "retrieval_recall"]
    assert len(shared) == 2 and shared[0]["p_adj"] == shared[1]["p_adj"]
    assert all(0 <= c["p_adj"] <= 1 for c in comps)


def test_limit_and_unknown_keys(examples_dir, tmp_path):
    cfg = _config(examples_dir, tmp_path, limit=5)
    res = run_experiment(cfg, quiet=True, write=False)
    assert res.summary["n_examples"] == 5
    with pytest.raises(ValueError):
        _config(examples_dir, tmp_path, rag=[{"name": "x", "bogus": 1}])


def test_pareto_front():
    def c(acc, cost, lat):
        return {"metrics": {"answer_correctness": {"mean": acc}, "cost_usd": {"mean": cost}}, "latency_p95_ms": lat}

    front = pareto_front({"a": c(0.9, 1.0, 100), "b": c(0.8, 0.5, 100), "c": c(0.7, 0.6, 120)})
    assert set(front) == {"a", "b"}


def test_cli_quickstart(examples_dir, tmp_path, capsys):
    assert (
        main(
            [
                "run",
                str(examples_dir / "configs" / "quickstart.yaml"),
                "--limit",
                "4",
                "--no-cache",
                "-q",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "report.md" in capsys.readouterr().out
    run_dir = next(tmp_path.glob("quickstart/*"))
    assert main(["report", str(run_dir)]) == 0


def test_cli_calibrate(examples_dir, capsys):
    assert main(["calibrate", str(examples_dir / "judge_calibration.jsonl")]) == 0
    assert "faithfulness" in capsys.readouterr().out
