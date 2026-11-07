"""Writes results.jsonl, summary.json, summary.csv, report.md and report.html."""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Any

import yaml

from ragbench.metrics import METRIC_BY_KEY, QUALITY_METRICS

LEADERBOARD = [
    ("answer_correctness", "Correctness"),
    ("faithfulness", "Faithfulness"),
    ("context_relevance", "Ctx relevance"),
    ("retrieval_precision", "Precision@k"),
    ("retrieval_recall", "Recall@k"),
    ("retrieval_mrr", "MRR"),
]


def _delta(metric: str, v: float) -> str:
    if metric == "latency_total_ms":
        return f"{v:+.0f} ms"
    if metric == "cost_usd":
        return f"{v * 1000:+.3f} $/1k"
    return f"{v:+.3f}"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "run"


def write_outputs(result, out_dir: Path | None = None) -> Path:
    cfg = result.config
    if out_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = cfg.output_dir / _slug(cfg.name) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for r in result.records:
            f.write(json.dumps(r, default=str) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(result.summary, indent=2, default=str))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False))
    write_reports(result.records, result.summary, out_dir)
    return out_dir


def write_reports(records: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    write_csv(summary, out_dir / "summary.csv")
    (out_dir / "report.md").write_text(render_markdown(records, summary))
    (out_dir / "report.html").write_text(render_html(records, summary))


def write_csv(summary: dict[str, Any], path: Path) -> None:
    keys = list(METRIC_BY_KEY)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "rag",
                "llm",
                "n",
                "errors",
                *keys,
                *(f"{k}_lo" for k in keys),
                *(f"{k}_hi" for k in keys),
                "latency_p50_ms",
                "latency_p95_ms",
                "total_cost_usd",
                "judge_cost_usd",
                "false_abstention_rate",
                "correct_abstention_rate",
            ]
        )
        for lab, c in summary["configs"].items():
            m = c["metrics"]
            w.writerow(
                [
                    lab,
                    c["rag"],
                    c["llm"],
                    c["n"],
                    c["errors"],
                    *(m[k]["mean"] for k in keys),
                    *(m[k]["lo"] for k in keys),
                    *(m[k]["hi"] for k in keys),
                    c["latency_p50_ms"],
                    c["latency_p95_ms"],
                    c["total_cost_usd"],
                    c["judge_cost_usd"],
                    c["false_abstention_rate"],
                    c["correct_abstention_rate"],
                ]
            )


# ----------------------------------------------------------------------------- formatting helpers


def _fmt(v: float | None, spec: str = "{:.3f}") -> str:
    return "–" if v is None else spec.format(v)


def _est(e: dict[str, Any], spec: str = "{:.2f}") -> str:
    if e["mean"] is None:
        return "–"
    half = max(e["mean"] - e["lo"], e["hi"] - e["mean"])
    return f"{spec.format(e['mean'])} ±{spec.format(half)}"


def _verdict(row: dict[str, Any]) -> str:
    if not row["significant"]:
        return "no sig. difference"
    higher = METRIC_BY_KEY[row["metric"]].higher_is_better
    better = (row["delta"] > 0) == higher
    return "better" if better else "worse"


def _ranked(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        summary["configs"].items(),
        key=lambda kv: -(kv[1]["metrics"]["answer_correctness"]["mean"] or -1),
    )


def _cost_per_1k(c: dict[str, Any]) -> float | None:
    m = c["metrics"]["cost_usd"]["mean"]
    return None if m is None else m * 1000


# ----------------------------------------------------------------------------- markdown


def render_markdown(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"# ragbench report: {summary['name']}\n")
    if summary.get("description"):
        out.append(summary["description"].strip() + "\n")
    out.append(
        f"- **Examples:** {summary['n_examples']}  \n- **Configurations:** {len(summary['configs'])}  \n"
        f"- **Judge:** `{summary['judge']}`  \n- **Baseline:** `{summary['baseline']}`  \n"
        f"- **Generated:** {summary['created_at']}\n"
    )
    out.append(
        "Values are means with 95% bootstrap confidence intervals (±). Comparisons use a paired "
        "permutation test with Holm–Bonferroni correction across all quality metrics and configurations.\n"
    )

    out.append("## Leaderboard\n")
    head = [
        "#",
        "Configuration",
        *(h for _, h in LEADERBOARD),
        "p50 ms",
        "p95 ms",
        "Tokens in/out",
        "$ / 1k queries",
        "Pareto",
    ]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    pareto = set(summary.get("pareto", []))
    for i, (lab, c) in enumerate(_ranked(summary), 1):
        m = c["metrics"]
        cells = [
            str(i),
            f"`{lab}`",
            *(_est(m[k]) for k, _ in LEADERBOARD),
            _fmt(c["latency_p50_ms"], "{:.0f}"),
            _fmt(c["latency_p95_ms"], "{:.0f}"),
            f"{_fmt(m['input_tokens']['mean'], '{:.0f}')} / {_fmt(m['output_tokens']['mean'], '{:.0f}')}",
            _fmt(_cost_per_1k(c), "${:.3f}"),
            "★" if lab in pareto else "",
        ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("\n★ = Pareto-optimal on correctness vs. cost vs. p95 latency.\n")

    out.append("## Retrieval detail\n")
    out.append("| RAG config | Pipeline | Chunks | nDCG@k | Hit@k | Retrieval p50 ms |")
    out.append("|---|---|---|---|---|---|")
    seen = set()
    for c in summary["configs"].values():
        if c["rag"] in seen:
            continue
        seen.add(c["rag"])
        info = summary.get("indexes", {}).get(c["rag"], {})
        rows = [r["latency_retrieval_ms"] for r in records if r["rag"] == c["rag"]]
        rows.sort()
        p50 = rows[len(rows) // 2] if rows else None
        out.append(
            f"| `{c['rag']}` | {info.get('description', '')} | {info.get('n_chunks', '–')} | "
            f"{_est(c['metrics']['retrieval_ndcg'])} | {_est(c['metrics']['retrieval_hit'])} | {_fmt(p50, '{:.2f}')} |"
        )
    out.append("")

    out.append("## Abstention behaviour\n")
    out.append("| Configuration | Correct abstention (unanswerable) | False abstention (answerable) | Errors |")
    out.append("|---|---|---|---|")
    for lab, c in _ranked(summary):
        out.append(
            f"| `{lab}` | {_fmt(c['correct_abstention_rate'], '{:.0%}')} | {_fmt(c['false_abstention_rate'], '{:.0%}')} | {c['errors']} |"
        )
    out.append("")

    cats = sorted({cat for c in summary["configs"].values() for cat in c["by_category"]})
    if len(cats) > 1:
        out.append("## Correctness by question category\n")
        out.append("| Configuration | " + " | ".join(f"{cat}" for cat in cats) + " |")
        out.append("|---|" + "---|" * len(cats))
        for lab, c in _ranked(summary):
            out.append(
                f"| `{lab}` | "
                + " | ".join(
                    f"{_fmt(c['by_category'].get(cat, {}).get('answer_correctness'), '{:.2f}')} (n={c['by_category'].get(cat, {}).get('n', 0)})"
                    for cat in cats
                )
                + " |"
            )
        out.append("")

    comps = summary.get("comparisons", [])
    if comps:
        order = [*QUALITY_METRICS, "latency_total_ms", "cost_usd"]
        metrics = sorted({r["metric"] for r in comps}, key=order.index)
        out.append(f"## Significance vs. baseline `{summary['baseline']}`\n")
        out.append(
            "Cells show the paired difference (candidate − baseline). **▲** significantly better, "
            "**▼** significantly worse (Holm-adjusted p < "
            f"{summary.get('alpha', 0.05)}); unmarked differences are not distinguishable from noise. "
            "Full CIs and p-values are in `summary.json`.\n"
        )
        out.append("| Configuration | " + " | ".join(METRIC_BY_KEY[m].label for m in metrics) + " |")
        out.append("|---|" + "---|" * len(metrics))
        by_cfg: dict[str, dict[str, Any]] = {}
        for r in comps:
            by_cfg.setdefault(r["config"], {})[r["metric"]] = r
        for lab, row in by_cfg.items():
            cells = []
            for m in metrics:
                r = row.get(m)
                if r is None:
                    cells.append("–")
                    continue
                v = _verdict(r)
                mark = {"better": " **▲**", "worse": " **▼**"}.get(v, "")
                cells.append(_delta(m, r["delta"]) + mark)
            out.append(f"| `{lab}` | " + " | ".join(cells) + " |")
        out.append("")

    out.append("## Failure analysis\n")
    unfaithful = sorted(
        (r for r in records if r.get("faithfulness") is not None and r["faithfulness"] < 1),
        key=lambda r: r["faithfulness"],
    )[:8]
    out.append(
        f"### Least faithful answers ({sum(1 for r in records if (r.get('faithfulness') or 1) < 1)} total below 1.0)\n"
    )
    if not unfaithful:
        out.append("_None — every answer was fully supported by its retrieved context._\n")
    for r in unfaithful:
        claims = [c for c in r.get("judge", {}).get("faithfulness", {}).get("claims", []) if not c.get("supported")]
        out.append(f"- **{r['example_id']}** `{r['config']}` (faithfulness {r['faithfulness']:.2f}): {r['question']}")
        for c in claims[:3]:
            out.append(f"  - unsupported: _{c.get('claim', '')}_")
    out.append("")
    misses = [r for r in records if r.get("retrieval_recall") == 0]
    seen_m = set()
    out.append(f"### Retrieval misses (recall@k = 0): {len({(r['rag'], r['example_id']) for r in misses})}\n")
    for r in misses:
        key = (r["rag"], r["example_id"])
        if key in seen_m or len(seen_m) >= 8:
            continue
        seen_m.add(key)
        got = ", ".join(sorted({h["doc_id"] for h in r["retrieved"]}))
        out.append(f"- **{r['example_id']}** `{r['rag']}`: {r['question']} — retrieved: {got}")
    out.append("")

    total_gen = sum(c["total_cost_usd"] for c in summary["configs"].values())
    total_judge = sum(c["judge_cost_usd"] for c in summary["configs"].values())
    out.append("## Run cost\n")
    out.append(f"Generation: ${total_gen:.4f} · Judging: ${total_judge:.4f} · Total: ${total_gen + total_judge:.4f}\n")
    return "\n".join(out)


# ----------------------------------------------------------------------------- html


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3df;--card:#fff;--accent:#2f6fdf;--good:#1f8a4c;--bad:#c2412d;--dot:#2f6fdf}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#9d9c95;--line:#2d2d2a;--card:#1c1c1a;--accent:#7aa7ff;--good:#4cc27e;--bad:#ff7a63;--dot:#7aa7ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}
.meta{color:var(--muted);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}th{color:var(--muted);font-weight:600}code{font-size:12px}
.good{color:var(--good)}.bad{color:var(--bad)}.ns{color:var(--muted)}.ci{color:var(--muted);font-size:11px}
svg text{fill:var(--muted);font-size:11px}svg .lab{fill:var(--fg)}svg line.axis{stroke:var(--line)}svg .pt{fill:var(--dot)}svg .err{stroke:var(--dot);opacity:.5}
"""


def _h(s: Any) -> str:
    return html.escape(str(s))


def _cell(e: dict[str, Any]) -> str:
    if e["mean"] is None:
        return "<td>–</td>"
    return f"<td>{e['mean']:.3f}<br><span class=ci>[{e['lo']:.2f}, {e['hi']:.2f}]</span></td>"


def _scatter(summary: dict[str, Any]) -> str:
    rank = {lab: i for i, (lab, _) in enumerate(_ranked(summary), 1)}
    pts = [
        (lab, c["metrics"]["cost_usd"]["mean"] * 1000, c["metrics"]["answer_correctness"])
        for lab, c in summary["configs"].items()
        if c["metrics"]["cost_usd"]["mean"] is not None and c["metrics"]["answer_correctness"]["mean"] is not None
    ]
    if not pts:
        return ""
    w, h, pad = 760, 340, 56
    xmax = max(p[1] for p in pts) * 1.15 or 1.0
    x = lambda v: pad + (w - 2 * pad) * v / xmax  # noqa: E731
    y = lambda v: h - pad + (pad * 2 - h) * v  # noqa: E731  (0..1 -> bottom..top)
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Correctness versus cost">',
        f"<line class=axis x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} /><line class=axis x1={pad} y1={pad} x2={pad} y2={h - pad} />",
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f"<text x={pad - 8} y={y(t) + 4} text-anchor=end>{t:.2f}</text>")
    for i in range(5):
        v = xmax * i / 4
        parts.append(f"<text x={x(v)} y={h - pad + 18} text-anchor=middle>${v:.3f}</text>")
    parts.append(f"<text x={w / 2} y={h - 12} text-anchor=middle>Cost per 1k queries (USD)</text>")
    parts.append(
        f'<text x=14 y={h / 2} transform="rotate(-90 14 {h / 2})" text-anchor=middle>Answer correctness</text>'
    )
    for lab, cx, est in pts:
        parts.append(f"<line class=err x1={x(cx)} x2={x(cx)} y1={y(est['lo'])} y2={y(est['hi'])} stroke-width=2 />")
        parts.append(
            f"<circle class=pt cx={x(cx)} cy={y(est['mean'])} r=5><title>{_h(lab)}: {est['mean']:.3f}</title></circle>"
        )
        parts.append(f"<text class=lab x={x(cx) + 7} y={y(est['mean']) - 7}>{rank[lab]}</text>")
    parts.append("</svg>")
    return "".join(parts)


def render_html(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    pareto = set(summary.get("pareto", []))
    rows = []
    for i, (lab, c) in enumerate(_ranked(summary), 1):
        m = c["metrics"]
        rows.append(
            f"<tr><td>{i}</td><td class=l><code>{_h(lab)}</code>{' ★' if lab in pareto else ''}</td>"
            + "".join(_cell(m[k]) for k, _ in LEADERBOARD)
            + f"<td>{_fmt(c['latency_p50_ms'], '{:.0f}')}</td><td>{_fmt(c['latency_p95_ms'], '{:.0f}')}</td>"
            f"<td>{_fmt(m['input_tokens']['mean'], '{:.0f}')} / {_fmt(m['output_tokens']['mean'], '{:.0f}')}</td>"
            f"<td>{_fmt(_cost_per_1k(c), '${:.3f}')}</td></tr>"
        )
    comps = summary.get("comparisons", [])
    order = [*QUALITY_METRICS, "latency_total_ms", "cost_usd"]
    metrics = sorted({r["metric"] for r in comps}, key=order.index)
    grid: dict[str, dict[str, Any]] = {}
    for r in comps:
        grid.setdefault(r["config"], {})[r["metric"]] = r
    comp_rows = []
    for lab, row in grid.items():
        cells = []
        for m in metrics:
            r = row.get(m)
            if r is None:
                cells.append("<td class=ns>–</td>")
                continue
            v = _verdict(r)
            cls = {"better": "good", "worse": "bad"}.get(v, "ns")
            mark = {"better": " ▲", "worse": " ▼"}.get(v, "")
            tip = f"95% CI [{r['lo']:+.4g}, {r['hi']:+.4g}], Holm p={r['p_adj']:.4f}"
            cells.append(f"<td class={cls} title='{tip}'>{_delta(m, r['delta'])}{mark}</td>")
        comp_rows.append(f"<tr><td class=l><code>{_h(lab)}</code></td>{''.join(cells)}</tr>")
    head = "".join(f"<th>{_h(hd)}</th>" for _, hd in LEADERBOARD)
    comp_html = (
        f"<h2>Significance vs. baseline <code>{_h(summary['baseline'])}</code></h2><div class=card><table>"
        "<tr><th class=l>Configuration</th>"
        + "".join(f"<th>{_h(METRIC_BY_KEY[m].label)}</th>" for m in metrics)
        + "</tr>"
        + "".join(comp_rows)
        + "</table><p class=meta>Paired difference (candidate − baseline). ▲/▼ significantly better/worse after "
        "Holm–Bonferroni correction; hover a cell for its CI and p-value.</p></div>"
        if comp_rows
        else ""
    )
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ragbench report</title><style>{CSS}</style></head><body><main>
<h1>{_h(summary["name"])}</h1>
<div class=meta>{summary["n_examples"]} examples · {len(summary["configs"])} configurations · judge <code>{_h(summary["judge"])}</code> · {summary["created_at"]}</div>
<p class=meta>{_h(summary.get("description", ""))}</p>
<h2>Leaderboard</h2><div class=card><table>
<tr><th>#</th><th class=l>Configuration</th>{head}<th>p50 ms</th><th>p95 ms</th><th>Tokens in/out</th><th>$ / 1k</th></tr>
{"".join(rows)}</table><p class=meta>Means with 95% bootstrap CIs. ★ Pareto-optimal on correctness, cost and p95 latency.</p></div>
<h2>Correctness vs. cost</h2><div class=card>{_scatter(summary)}<p class=meta>Points are numbered by leaderboard rank; bars are 95% CIs.</p></div>
{comp_html}
</main></body></html>"""
