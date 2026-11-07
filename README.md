# ragbench

**Rigorous evaluation of RAG configurations and LLMs.**

Building a retrieval-augmented generation (RAG) app is the easy part. Showing that it works, and that
configuration B really is better than configuration A, is harder. ragbench runs every combination of
retrieval setup × LLM against a labelled dataset and reports:

| Dimension | Metric(s) | How it is measured |
|---|---|---|
| **Retrieval quality** | Precision@k, Recall@k, MRR, nDCG@k, Hit@k | Retrieved chunks are checked against labelled evidence passages, or against relevant document ids when no evidence is labelled |
| **Context relevance** | Share of retrieved context that helps answer the question | A judge rates each retrieved passage (LLM judge) or sentence (heuristic judge) |
| **Faithfulness** | Share of the answer's claims that the retrieved context supports | The answer is split into claims and each claim is checked against the context |
| **Answer correctness** | Agreement with the reference answer (0–1), plus token F1 | 1–5 rubric (LLM judge) or key-fact recall (heuristic). Unanswerable questions are scored on abstention |
| **Latency** | Retrieval, generation and total time: mean, p50, p95 | Wall-clock timing around each stage |
| **Tokens & cost** | Input/output tokens, $ per query, $ per 1k queries, judge cost | Provider-reported usage × a price table you can override |

These numbers are only useful if you can trust them, so ragbench also provides:

- **95% bootstrap confidence intervals** on every metric.
- **Paired significance tests** against a baseline. Every configuration answers the same questions, so
  the comparison is paired (sign-flip permutation test).
- **Holm–Bonferroni correction**, so that running 50 comparisons doesn't produce false "wins".
- **Pareto frontier** on correctness, cost and p95 latency, which shows which configurations are worth
  considering at all.
- **Judge calibration** (`ragbench calibrate`): measures how well a judge agrees with human labels
  (accuracy, Cohen's κ, Spearman) before you rely on it.
- **Auditable judgments**: every per-claim and per-passage verdict is stored in `results.jsonl`.
- **Abstention tracking**: unanswerable questions measure whether the system says "I don't know"
  instead of making up an answer, and false abstentions on answerable questions are counted too.
- **Reproducibility**: seeded statistics, a SQLite response cache, and a copy of the resolved config
  saved with every run.

## Quickstart (offline, no API keys)

```bash
pip install -e .
ragbench run examples/configs/quickstart.yaml
```

This evaluates 5 retrieval setups × 3 deterministic mock models (15 configurations) on 32 questions
about a fictional company handbook, in about a second. Each run writes the following to
`runs/<name>/<timestamp>/`:

| File | Contents |
|---|---|
| `report.md` / `report.html` | Leaderboard, retrieval detail, abstention behaviour, per-category correctness, significance matrix, failure analysis |
| `results.jsonl` | One row per (configuration, question): answer, retrieved chunks with relevance labels, every metric, judge verdicts |
| `summary.json` / `summary.csv` | Aggregates with CIs, every pairwise test, Pareto set |
| `config.yaml` | The exact config that produced the run |

An excerpt from the quickstart report:

| # | Configuration | Correctness | Faithfulness | Ctx relevance | Precision@k | Recall@k | p95 ms | $ / 1k queries |
|---|---|---|---|---|---|---|---|---|
| 3 | `tfidf-sent2-k2 / mock:verbose` | 0.75 ±0.14 | 1.00 ±0.00 | 0.55 ±0.11 | 0.52 ±0.07 | 0.91 ±0.09 | 2147 | $0.918 |
| 7 | `tfidf-sent2-k2 / mock:hallucinating` | 0.73 ±0.14 | 0.89 ±0.08 | 0.55 ±0.11 | 0.52 ±0.07 | 0.91 ±0.09 | 453 | $0.047 |
| 8 | `bm25-sent3-k4 / mock:extractive` | 0.69 ±0.14 | 1.00 ±0.00 | 0.30 ±0.06 | 0.40 ±0.05 | 0.98 ±0.04 | 646 | $0.199 |

The report also points out things a plain leaderboard hides:

- `mock:hallucinating` adds an unsupported sentence to some answers. Faithfulness catches it (0.89 vs
  1.00), but on 32 questions the drop is **not significant after correcting for multiple comparisons**.
  A larger dataset is needed before claiming a difference.
- Smaller chunks with a smaller k (`tfidf-sent2-k2`) **significantly** improve context relevance and cut
  input tokens by half, at a small, non-significant cost in recall.
- Bigger, slower models (`mock:verbose`) are significantly worse on latency and cost, and not
  significantly better on correctness.

## Evaluating real models

```bash
pip install -e '.[anthropic]'          # or '.[openai]', '.[all]'
export ANTHROPIC_API_KEY=...
ragbench run examples/configs/claude.yaml --limit 5   # small first run to check cost
ragbench run examples/configs/claude.yaml
```

Supported providers are `anthropic:<model>`, `openai:<model>` (including any OpenAI-compatible server
via `base_url`) and `mock:<extractive|verbose|hallucinating>`. Run `ragbench models` to list built-in
prices.

## Configuration

```yaml
name: my-experiment
corpus: ../corpus              # directory of .md/.txt files, or JSONL with id/text
dataset: ../dataset.jsonl      # paths are relative to this file
output_dir: runs
seed: 7
concurrency: 8
cache: true                    # or a path to a sqlite file; false to disable
alpha: 0.05
relevance_threshold: 0.6       # share of evidence tokens a chunk must contain to count as relevant

judge:
  type: llm                    # or: heuristic
  model: anthropic:claude-sonnet-5-5
  params: {effort: low}

rag:
  - name: bm25-sent3-k4
    chunker: {type: sentence, size: 3, overlap: 1}    # fixed | sentence | paragraph | document
    retriever: {type: bm25, k1: 1.5, b: 0.75}         # bm25 | tfidf | lsa | dense | hybrid
    k: 4
  - name: hybrid
    chunker: {type: fixed, size: 120, overlap: 20}
    retriever:
      type: hybrid                                    # reciprocal-rank fusion
      retrievers: [{type: bm25}, {type: dense, embedder: {type: openai, model: text-embedding-3-small}}]
      weights: [1.0, 1.0]
    k: 6
    # optional: system_prompt, prompt_template ({context}, {question}), passage_format

llms:
  - anthropic:claude-haiku-5-5
  - id: anthropic:claude-opus-5-5
    name: opus-low                # display name, so one model can appear with different settings
    params: {effort: low, max_tokens: 4096}

baseline: bm25-sent3-k4 / opus-low   # "<rag name> / <llm name>"; defaults to the first configuration

pricing:                             # USD per 1M tokens; overrides the built-in table
  anthropic:claude-haiku-5-5: {input: 0.10, output: 0.50}
```

### Dataset format

One JSON object per line:

```json
{"id": "q27", "question": "What would one K9 plus two spare batteries cost?",
 "reference_answer": "34,300 US dollars.", "relevant_doc_ids": ["product-k9-applicator"],
 "evidence": ["The list price of the K9 Applicator is 32,000 US dollars", "Spare batteries cost 1,150 US dollars each"],
 "answerable": true, "metadata": {"category": "reasoning"}}
```

`evidence` is optional but recommended. With it, retrieval is scored at chunk level (did we fetch the
passage?) instead of document level (did we fetch the right file?). Set `"answerable": false` with
empty `relevant_doc_ids` for questions the corpus cannot answer. `metadata.category` adds a
per-category breakdown to the report.

## Judges and calibration

- **`heuristic`**: deterministic lexical checks. Free, instant and reproducible, so it suits CI and
  quick iteration. It misses paraphrase and can be fooled by number reuse.
- **`llm`**: per-passage and per-claim verdicts from any supported model, returned as strict JSON and
  stored for auditing. Use a judge from a different model family, or at least a different model, than
  the ones under test to reduce self-preference bias.

Measure judge agreement with human labels before you rely on a judge:

```bash
ragbench calibrate examples/judge_calibration.jsonl                              # heuristic
ragbench calibrate examples/judge_calibration.jsonl --judge anthropic:claude-sonnet-5-5
```

```
metric                  n  accuracy   kappa  spearman    MAE
faithfulness           14     0.786   0.588     0.645  0.214
answer_correctness     14     0.643   0.186     0.588  0.235
```

With κ = 0.19 on correctness, the heuristic judge shouldn't be trusted for final correctness numbers.
That is the kind of finding this command exists to surface.

## Methodology

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for metric definitions, the statistical tests, and
known limitations.

## Extending

```python
from ragbench.retrieval.base import Retriever
from ragbench.llm.base import LLM, Completion

class MyRetriever(Retriever):
    def _build(self, chunks): ...
    def scores(self, query): ...     # one float per chunk

class MyLLM(LLM):
    provider = "mine"
    def _complete(self, prompt, system, hints):
        ...
        return Completion(text, input_tokens, output_tokens, stop_reason)
```

Latency measurement, cost, caching and error capture come from the base classes. You can also drive
everything from Python with `ragbench.experiment.ExperimentConfig.from_dict(...)` and
`run_experiment(cfg)`.

## Project layout

```
src/ragbench/
  chunking.py        fixed / sentence / paragraph / document chunkers
  retrieval/         BM25, TF-IDF, LSA, embedding (sentence-transformers, OpenAI), hybrid RRF
  llm/               provider adapters, pricing, SQLite cache, offline mock models
  judges/            heuristic and LLM-as-judge implementations
  metrics/           retrieval metrics, answer metrics, metric registry
  stats.py           bootstrap CIs, paired permutation test, Holm, kappa, Spearman
  experiment.py      config, grid execution, aggregation, comparisons, Pareto front
  report.py          Markdown / HTML / CSV / JSON outputs
  calibration.py     judge vs. human agreement
examples/            fictional corpus, 32-question dataset, configs, calibration labels
tests/               pytest suite (runs offline)
```

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check src tests
```

## License

MIT
