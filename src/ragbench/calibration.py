"""Judge calibration: how well does an automatic judge agree with human labels?

An LLM-as-judge number is only as trustworthy as the judge. Before relying on a
judge, label a sample by hand and measure agreement. Input JSONL rows::

    {"question": ..., "answer": ..., "contexts": [...], "reference_answer": ...,
     "answerable": true, "human_faithfulness": 1, "human_correctness": 0}

Human labels may be binary (0/1) or graded in [0, 1]; any subset may be present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragbench.judges import Judge
from ragbench.stats import cohens_kappa, spearman


def calibrate(judge: Judge, labels_path: str | Path, threshold: float = 0.5) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(labels_path).read_text().splitlines() if line.strip()]
    pairs: dict[str, list[tuple[float, float]]] = {
        "faithfulness": [],
        "answer_correctness": [],
        "context_relevance": [],
    }
    for r in rows:
        ctx = r.get("contexts", [])
        if "human_faithfulness" in r:
            pairs["faithfulness"].append((r["human_faithfulness"], judge.faithfulness(r["answer"], ctx).score))
        if "human_correctness" in r:
            res = judge.answer_correctness(r["question"], r["answer"], r["reference_answer"], r.get("answerable", True))
            pairs["answer_correctness"].append((r["human_correctness"], res.score))
        if "human_context_relevance" in r:
            pairs["context_relevance"].append(
                (r["human_context_relevance"], judge.context_relevance(r["question"], ctx).score)
            )

    report: dict[str, Any] = {"judge": judge.name, "n_rows": len(rows), "metrics": {}}
    for metric, ps in pairs.items():
        ps = [(h, j) for h, j in ps if j is not None]
        if not ps:
            continue
        hb = [int(h >= threshold) for h, _ in ps]
        jb = [int(j >= threshold) for _, j in ps]
        report["metrics"][metric] = {
            "n": len(ps),
            "accuracy": sum(a == b for a, b in zip(hb, jb)) / len(ps),
            "cohens_kappa": cohens_kappa(hb, jb),
            "spearman": spearman([h for h, _ in ps], [j for _, j in ps]),
            "mean_abs_error": sum(abs(h - j) for h, j in ps) / len(ps),
        }
    return report


def format_calibration(report: dict[str, Any]) -> str:
    lines = [f"Judge: {report['judge']}  (rows: {report['n_rows']})", ""]
    lines.append(f"{'metric':20} {'n':>4} {'accuracy':>9} {'kappa':>7} {'spearman':>9} {'MAE':>6}")
    for m, v in report["metrics"].items():

        def f(x):
            return "   –" if x is None else f"{x:.3f}"

        lines.append(
            f"{m:20} {v['n']:>4} {f(v['accuracy']):>9} {f(v['cohens_kappa']):>7} {f(v['spearman']):>9} {f(v['mean_abs_error']):>6}"
        )
    lines.append("")
    lines.append(
        "Rule of thumb: kappa > 0.6 is substantial agreement; below 0.4, do not trust the judge for that metric."
    )
    return "\n".join(lines)
