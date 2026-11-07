# Methodology

## Units of evaluation

A **configuration** is a (RAG setup, LLM) pair. A RAG setup is a chunker, a retriever, a top-k, and a
prompt. Every configuration answers every question in the dataset, so every comparison is **paired**.

Retrieval and context relevance depend only on the RAG setup. ragbench computes them once per RAG
setup and shares them across LLMs. This saves judge cost and guarantees that two LLMs on the same RAG
setup received identical context.

## Retrieval metrics

Let `R` be the top-k retrieved chunks.

- **Relevance of a chunk**: if the example lists `evidence` passages, a chunk is relevant when it
  contains at least `relevance_threshold` (default 60%) of the content tokens of any evidence passage.
  The threshold tolerates evidence that is split across a chunk boundary. Without evidence, a chunk is
  relevant when it comes from a document in `relevant_doc_ids`.
- **Precision@k** = relevant chunks in `R` / k.
- **Recall@k** = share of evidence passages found in `R` (or share of relevant documents when there is no
  evidence).
- **MRR** = 1 / rank of the first relevant chunk (0 if none).
- **nDCG@k** uses binary gains. The ideal DCG uses the true number of relevant chunks in the index
  (capped at k).
- **Hit@k** = 1 if any relevant chunk is retrieved.

Retrieval metrics are undefined (`None`) for unanswerable questions and are excluded from their means.
`n` is reported alongside every mean.

Precision@k is bounded above by (number of relevant chunks / k). A setup with large chunks and small k
can show high precision while sending many irrelevant tokens to the model. Read it together with
context relevance and input tokens.

## Model-graded metrics

| Metric | LLM judge | Heuristic judge |
|---|---|---|
| Context relevance | Share of retrieved passages judged to help answer the question | Share of context sentences that share at least 30% of the question's content terms |
| Faithfulness | Answer split into claims; share judged supported by the context. No claims (abstention) → 1.0 | Share of answer sentences whose content tokens are ≥75% present in the context **and** whose numbers all appear in the context |
| Answer correctness | 1–5 rubric mapped to 0–1; unanswerable questions: abstaining scores 1, asserting anything scores 0 | Reference-token recall (70% recall counts as 1.0); missing a number from the reference caps the score at 0.5; same abstention rule |

Faithfulness and correctness are separate measurements. An answer can be faithful to bad context and
wrong, or correct from the model's background knowledge but unfaithful. The example corpus describes a
fictional company, so background knowledge cannot help and the two separate cleanly.

**Token F1** (SQuAD-style) is also reported as a judge-free reference point.

## Latency, tokens and cost

- Retrieval latency is measured around `retriever.retrieve`. Generation latency is measured around the
  provider call, including network time. Total latency is their sum. p50 and p95 are computed over
  successful calls.
- Token counts are the provider's reported usage. For reasoning models, thinking tokens are billed as
  output and are included.
- Cost = tokens × price per token, from the built-in table or the config's `pricing:` block. Judge cost is
  tracked separately so you can see what the evaluation itself costs.
- Cached responses keep the latency recorded when they were first produced and are flagged
  `cached: true`. To benchmark latency, run with `--no-cache`.

## Statistics

- **Confidence intervals**: percentile bootstrap (default 2,000 resamples, seeded) for each metric mean.
- **Comparisons**: for each configuration and metric, the per-question difference from the baseline is
  computed on the questions where both values are defined. Then:
  - 95% CI of the mean difference by paired bootstrap;
  - two-sided **sign-flip permutation test**. It is exact for n ≤ 16 and uses 10,000 Monte Carlo
    resamples otherwise, with +1 smoothing so p-values are never exactly 0. It makes no normality
    assumption, which matters for bounded, often bimodal scores.
- **Multiple comparisons**: all quality-metric tests form one Holm–Bonferroni family. Tests on RAG-scoped
  metrics run once per RAG setup, not once per LLM, so the family is not inflated by duplicate
  hypotheses. Latency and cost differences are reported uncorrected because they are usually large and
  are rarely the hypothesis being tested.
- **Pareto frontier**: configurations that no other configuration beats or matches on all three of
  correctness (higher is better), mean cost and p95 latency (lower is better).

### How many questions do you need?

The paired design helps, but small datasets still give wide intervals. The quickstart's
hallucinating model loses 0.11 faithfulness, and with 32 questions and ~40 corrected tests that is not
significant. As a rough guide, detecting a 5-point difference in a 0/1 metric with ~50% variance in
the per-question differences takes on the order of several hundred questions. Report the CIs and let
them guide how much data to label.

## Known limitations

- The heuristic judge is lexical. It misses paraphrase and can be fooled when a wrong answer reuses
  numbers that appear elsewhere in the context. Use `ragbench calibrate` to quantify this, and use an LLM
  judge for final numbers.
- LLM judges have their own biases (verbosity, self-preference). Calibrate them as well, and pin the
  judge model and prompt for every run you intend to compare.
- Mock models are sentence extractors, not language models. They exist to exercise the pipeline and
  metrics with known behaviour.
- Single-run measurements: provider nondeterminism and network jitter are not averaged over repeats.
  For latency claims, run several times with `--no-cache`.
