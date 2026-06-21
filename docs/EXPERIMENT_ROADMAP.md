# Experiment Roadmap

## Purpose and scope

This document records the implemented QASPER/Qdrant pipeline and the experiments that remain to be built. The current repository is the source of truth: items are called **completed** only when supported by the present code or generated artifacts. Proposed schemas, evaluators, router components, and analyses are explicitly marked **planned**.

The research objective is to measure how fixed token-level chunk granularity affects evidence retrieval, establish a ground-truth-informed oracle upper bound, and test whether a router can choose a useful granularity from a question embedding at inference time.

## Metric and similarity vocabulary

Four quantities must remain distinct throughout implementation and reporting:

| Quantity | Inputs | Meaning | Current status |
|---|---|---|---|
| **Query-to-chunk retrieval similarity** | Question embedding and chunk embedding | Qdrant cosine score used to rank chunks for a question. This is an inference-time retrieval signal and does not use ground-truth evidence. | **Implemented.** Stored as `topk_scores`, with `avg_score_topk` and `best_score_topk`. |
| **Evidence-to-chunk oracle similarity** | Ground-truth evidence embedding and chunk embedding | Measures how directly a chunk aligns with annotated evidence. Because it uses evidence unavailable at deployment, it is an oracle/analysis signal rather than a deployable retrieval rule. | **Not implemented.** |
| **Per-chunk F1** | One retrieved chunk's text and the ground-truth evidence text | Token-overlap F1 for each individual retrieved chunk. It can expose whether one strong chunk is hidden inside a weak top-K set. | **Not implemented.** |
| **Top-K set-level F1** | All top-K retrieved chunk texts joined together and all evidence texts joined together | Aggregate token-overlap quality of the retrieved set. In the current code this is `f1_joined_topk`; it is not an average of per-chunk F1 values. | **Implemented.** |

The current token-level F1 normalizes case, punctuation, and whitespace, tokenizes with the configured Hugging Face tokenizer, and uses multiset token overlap. The current evaluator calls this once on the joined top-K retrieval and joined evidence.

**Oracle granularity selection must never be presented as a deployable inference strategy.** The oracle uses ground-truth highlighted evidence to score or select a granularity. That evidence is available for supervised experiment construction and evaluation, but it will not be available for a new question at deployment. Oracle performance is therefore an upper bound and a source of training labels/diagnostics. A deployable router must make its decision from inference-time features such as the question embedding alone.

## Completed pipeline

### 1. QASPER ingestion

`prepare_dataset.py` loads `allenai/qasper` with Hugging Face Datasets and processes the requested train, validation, and test splits. Each paper is flattened in this order:

1. title;
2. abstract;
3. section name;
4. section paragraphs.

Deterministic UUID-5 identifiers make repeated Qdrant upserts idempotent.

### 2. Filtering questions without highlighted evidence

Before question ingestion, all annotations are inspected for non-empty `highlighted_evidence`. Questions with no usable highlighted evidence are excluded from `PaperQuestion` and consequently from evidence ingestion and evaluation. Evidence strings from multiple annotators are deduplicated per question before storage.

### 3. Fixed non-overlapping chunking

The flattened paper is tokenized using the configured fast Hugging Face tokenizer (`gpt2` by default) and divided into non-overlapping chunks at five fixed target sizes:

| Granularity level | Target tokens |
|---:|---:|
| 1 | 10 |
| 2 | 20 |
| 3 | 40 |
| 4 | 80 |
| 5 | 160 |

The final chunk may be shorter. Payloads retain document metadata, chunk index, total chunks, actual token count, granularity level, content, and character-span metadata.

### 4. OpenAI embeddings

Chunks, answerable questions, and evidence strings are embedded through the OpenAI API. The default configuration is `text-embedding-3-small`, with 1,536 dimensions expected by the Qdrant schema. Embedding requests support batching, concurrent batch execution, output-order restoration, empty-string protection, and exponential-backoff retries.

### 5. Qdrant persistence

The active persistence layer is Qdrant. It contains three collections:

- `PaperChunk`: embedded chunks at all five granularities;
- `PaperQuestion`: embedded answerable questions and dataset split metadata;
- `PaperEvidence`: embedded, deduplicated highlighted evidence linked to question UUIDs.

The collections use cosine distance, on-disk vectors, and on-disk HNSW configuration. Payload indexes support filters on document, granularity, split, and question identifiers. The retained `weaviate_schema.py` is legacy code and is not part of the active pipeline.

### 6. Checkpointed parallel ingestion

Ingestion is parallelized at two levels:

- papers are processed through a thread pool;
- embedding batches are sent through another configurable thread pool.

`checkpoint.json` records completion separately for each paper's five chunk sizes, questions, and evidence. Checkpoint writes are lock-protected and use a temporary file replacement. A failed run can resume its missing stages; `--recreate` clears both the checkpoint and Qdrant collections.

### 7. Fixed-separate top-K retrieval

The implemented baseline evaluates each granularity separately. For each question and granularity level, Qdrant searches `PaperChunk` using the stored question embedding while filtering by:

- the question's `document_id`; and
- exactly one `granularity_level`.

The evaluator retrieves the top K chunks, currently defaulting to K=5. Qdrant's cosine scores are the **query-to-chunk retrieval similarities**. Retrieval at one granularity does not compete with chunks from another granularity.

### 8. Aggregate token-level F1 evaluation

All retrieved texts at a given granularity are joined, all stored evidence passages for the question are joined, and one token-level F1 is computed between the two combined strings. This is the **top-K set-level F1** recorded as `f1_joined_topk`.

The evaluator also records latency, chunk IDs and indices, token counts, individual Qdrant scores, mean query-to-chunk score, and best query-to-chunk score. It does not currently calculate a separate F1 for every retrieved chunk.

### 9. JSONL outputs

Optional ingestion exports are written to:

- `json_output/chunks.jsonl`;
- `json_output/questions.jsonl`;
- `json_output/evidence.jsonl`.

Evaluation writes timestamped files matching:

```text
outputs/RetrievalEvalFixedSeparate_<YYYYMMDD_HHMMSS>.jsonl
```

Full retrieved and evidence text is omitted by default and can be included with `--store-text`.

## Current repository snapshot and known limitations

As inspected on 2026-06-21, the local checkpoint represents 1,585 papers, but not every stage is complete:

| Checkpoint stage | Papers marked complete |
|---|---:|
| 10-token chunks | 1,585 |
| 20-token chunks | 1,585 |
| 40-token chunks | 1,585 |
| 80-token chunks | 1,583 |
| 160-token chunks | 1,583 |
| Questions | 1,581 |
| Evidence | 1,581 |

Additional limitations are:

- Only a smoke-test evaluation exists. The sole populated evaluation file has 15 records: three questions evaluated at five granularities. The other existing evaluation file is empty.
- No full-split or full-dataset baseline result has been produced and checked into a reproducible analysis workflow.
- No per-retrieved-chunk F1 is calculated or stored.
- No evidence-to-chunk cosine similarity is calculated or stored.
- Evaluation records exist only as JSONL; there is no persistent Qdrant evaluation collection or other evaluation database schema.
- No mixed-granularity retrieval evaluator exists.
- No oracle-label dataset builder exists.
- No router dataset, router model, training procedure, or routed evaluator exists.
- No automated unit, integration, or end-to-end tests exist.
- There is no complete reproducibility guide covering environment setup, service lifecycle, ingestion verification, experiments, analysis, and artifact provenance.

## Experimental principles

The following constraints apply to all planned work:

1. Preserve QASPER's train/validation/test boundaries. Fit router parameters and preprocessing only on train data; use validation for model selection and test once for final reporting.
2. Keep retrieval candidates restricted to the question's source document unless a separately named experiment deliberately changes that assumption.
3. Record K, tokenizer, chunk sizes, embedding model, collection configuration, code revision, split, random seed, and aggregation rules with every experiment.
4. Do not conflate Qdrant's question-to-chunk score with evidence-based oracle similarity or token F1.
5. Define tie-breaking before producing oracle labels. At minimum, report tie frequency and use a deterministic rule.
6. Keep the fixed-separate baseline immutable as a comparison point; add new methods rather than silently changing its meaning.
7. Treat evidence-derived labels and metrics as evaluation/supervision only. They must not enter the routed test-time retrieval path.

## Milestones

### Milestone 0 — Complete and validate ingestion

**Status:** Planned completion work; most ingestion is already present.

**Goal:** Bring all intended QASPER papers to a consistent checkpoint state and verify Qdrant/JSONL integrity before running expensive experiments.

**Dependencies:** Running Qdrant, valid OpenAI credentials, accessible QASPER dataset, current tokenizer and embedding configuration.

**Expected artifacts:**

- checkpoint with all expected stages complete for every intended paper;
- collection counts and per-split question counts;
- validation report covering missing stages, UUID uniqueness, payload completeness, vector dimensions, and JSONL/Qdrant consistency;
- recorded environment and configuration snapshot.

**Acceptance criteria:**

- every intended paper is either fully complete or explicitly documented as excluded with a reason;
- no checkpoint stage is marked complete when its required Qdrant records are absent;
- sampled chunks reproduce the configured tokenization and non-overlapping span order;
- every evaluated question has linked evidence;
- rerunning without `--recreate` performs no unintended duplicate insertion.

**Risks:** OpenAI cost/rate limits, partial Qdrant writes preceding checkpoint updates, shared JSONL writers under parallel paper processing, stale artifacts created with different source revisions, and exact evidence text not mapping uniquely back to flattened paper text.

### Milestone 1 — Separate-granularity oracle evaluation

**Status:** Planned. The existing fixed-separate evaluator supplies only part of this experiment.

**Goal:** Evaluate every question independently at all five fixed granularities and derive a ground-truth-informed best granularity per question.

The extended evaluation should retain current query-to-chunk scores and top-K set-level F1, then add:

- token F1 for every retrieved chunk against joined ground-truth evidence;
- evidence-to-chunk cosine similarity, with an explicitly documented aggregation when a question has multiple evidence vectors;
- deterministic oracle labels based on a predeclared evidence-derived objective;
- tie indicators and the complete score vector over all granularities.

The primary oracle objective must be fixed before evaluation. A defensible default is maximum top-K set-level F1, with deterministic tie-breaking documented. Evidence-to-chunk cosine and per-chunk F1 should remain separate diagnostic/oracle features rather than being silently blended into the same number.

**Dependencies:** Milestone 0; existing fixed-separate evaluator; stored question, chunk, and evidence vectors; agreed K values and oracle-label rule.

**Expected artifacts:**

- full per-question/per-granularity evaluation JSONL;
- per-retrieved-chunk metrics artifact;
- per-question oracle-label table containing all granularity scores and the selected label;
- aggregate tables by granularity and split;
- label distribution and tie analysis.

**Acceptance criteria:**

- all four metric/similarity concepts in this document have separate named fields;
- one record exists for every eligible question/granularity pair, or a documented skip reason exists;
- recomputed aggregates match record-level data;
- oracle labels are deterministic and use ground-truth evidence only in this offline stage;
- results are reported as an oracle upper bound, never as deployable performance.

**Risks:** Oracle-label instability across K or metric choice, longer chunks gaining recall merely by containing more tokens, duplicated evidence passages, ambiguous aggregation over multiple evidence items, and accidental use of test-derived labels during router development.

### Milestone 2 — Mixed-granularity retrieval

**Status:** Planned; no mixed evaluator exists.

**Goal:** Let chunks from all five granularities compete in one retrieval operation and compare the resulting top-K set with fixed-separate baselines.

The experiment must define how duplicate or nested content is handled. Because the same source span appears at multiple resolutions, naive top-K retrieval may return redundant overlapping chunks. At least one clearly specified policy should be evaluated, such as raw global ranking or overlap-aware deduplication; different policies must be named as different methods.

**Dependencies:** Milestone 0; metric extensions from Milestone 1; explicit candidate, score, overlap, and tie policies.

**Expected artifacts:**

- mixed-granularity per-question JSONL results;
- selected-granularity composition of each top-K set;
- redundancy/overlap diagnostics;
- aggregate comparison with each fixed granularity.

**Acceptance criteria:**

- all five granularity levels participate in the candidate pool;
- the candidate and deduplication policy is fully specified and deterministic;
- query-to-chunk score, per-chunk F1, and top-K set-level F1 remain separately reported;
- the same eligible questions and K values are used for paired baseline comparisons.

**Risks:** Strong score calibration differences between chunk lengths, domination by one granularity, nested duplicate results, increased search cost, and unfair comparisons caused by unequal retrieved-token budgets.

### Milestone 3 — Router dataset and training from question embeddings

**Status:** Planned; no router data or model code exists.

**Goal:** Train a model that predicts the oracle granularity label from a question embedding without access to evidence at inference time.

The input is the stored question embedding. Labels come from the train-split oracle analysis in Milestone 1. Validation data selects model configuration and stopping criteria. Test labels are reserved for final evaluation and must not influence training or tuning.

Because oracle labels may be noisy, imbalanced, or tied, the experiment should compare a simple reproducible baseline with any more complex router. Candidate framing—single-label classification, soft labels from normalized oracle scores, or cost-sensitive prediction—must be decided and documented before implementation.

**Dependencies:** Stable oracle definition; leakage-safe split assignments; complete question embeddings; recorded seeds; agreed model-selection metric.

**Expected artifacts:**

- versioned router dataset manifest with question UUID, split, embedding reference, oracle scores, label, and tie metadata;
- label-distribution report;
- trained model checkpoint and preprocessing metadata;
- training/validation curves and configuration;
- classification metrics including confusion matrix and per-class results.

**Acceptance criteria:**

- no evidence text/vector or evidence-derived test feature is supplied to the router;
- train, validation, and test question IDs are disjoint;
- a fixed-seed training run is repeatable within declared tolerance;
- the trained artifact records its oracle-label version, embedding model, feature dimension, and code revision;
- router accuracy is accompanied by downstream retrieval metrics, not treated as the sole success criterion.

**Risks:** Class imbalance, unstable oracle targets, question embeddings lacking sufficient granularity signal, overfitting a small class, data leakage through preprocessing, and optimizing label accuracy when several granularities have nearly identical retrieval quality.

### Milestone 4 — Router-based retrieval

**Status:** Planned; no routed evaluator exists.

**Goal:** For each question, predict one granularity from the question embedding, retrieve top K chunks only at that granularity, and evaluate the result without using evidence until scoring.

The deployable path is:

```text
question embedding -> router prediction -> selected granularity -> Qdrant top-K retrieval
```

Ground-truth evidence enters only after retrieval to compute evaluation metrics.

**Dependencies:** Trained router from Milestone 3; fixed-separate evaluator interfaces; frozen retrieval configuration; Milestone 1 metrics.

**Expected artifacts:**

- routed per-question JSONL results including predicted granularity and router confidence/score when available;
- paired oracle, routed, fixed-granularity, and mixed-granularity records;
- router inference and retrieval latency measurements;
- failure analysis by predicted and oracle granularity.

**Acceptance criteria:**

- the evaluator demonstrably does not access evidence before routing/retrieval;
- every routed result can be reproduced from the recorded model and configuration;
- results use the same eligible questions, K, embeddings, and F1 implementation as baselines;
- routed performance is reported alongside the oracle gap and strongest fixed baseline.

**Risks:** Router overhead, low-confidence predictions, missing-granularity chunks for incomplete papers, train/evaluation configuration drift, and error propagation from router misclassification.

### Milestone 5 — Final comparison and statistical analysis

**Status:** Planned; no analysis framework currently exists.

**Goal:** Compare fixed separate granularities, oracle selection, mixed retrieval, and routed retrieval on matched questions with uncertainty and statistical testing.

**Dependencies:** Frozen outputs from Milestones 1–4; predefined primary outcome and comparison family; stable analysis environment.

**Expected artifacts:**

- consolidated, versioned results table;
- macro means and distribution summaries for top-K set-level F1, retrieval similarity, latency, and retrieved-token count;
- paired confidence intervals and appropriate paired significance tests;
- multiple-comparison correction where needed;
- effect sizes, oracle-gap analysis, plots, and qualitative error cases;
- final reproducibility manifest and thesis-ready tables/figures.

**Acceptance criteria:**

- comparisons are paired on identical eligible questions;
- the primary metric and statistical tests are declared before inspecting final test results;
- confidence intervals and effect sizes accompany p-values;
- missing records and exclusions are enumerated;
- claims distinguish statistical significance, practical magnitude, and the non-deployable oracle upper bound;
- every published table or figure traces back to immutable record-level artifacts.

**Risks:** Multiple-testing inflation, overinterpreting the three-question smoke test, split contamination, selection of favorable K after test inspection, non-independent annotations, and confounding quality with retrieved-token budget or latency.

## Existing reproducible commands

These commands already correspond to current repository entry points. They require a configured `.env`, installed dependencies, and—except for purely local inspection—a running Qdrant service. They document existing behavior only; commands for unfinished milestones should be added when those components exist rather than invented now.

### Environment and Qdrant

```powershell
python -m pip install -r requirements.txt
docker compose up -d
```

### Ingestion

Resume all available splits from `checkpoint.json`:

```powershell
python prepare_dataset.py
```

Run a small ingestion test:

```powershell
python prepare_dataset.py --limit 5
```

Choose splits and concurrency:

```powershell
python prepare_dataset.py --splits train validation --paper-max-workers 4 --embedding-max-workers 8
```

Destructively recreate the three Qdrant collections and clear the checkpoint:

```powershell
python prepare_dataset.py --recreate
```

### Retrieval demonstration

```powershell
python retrieval_example.py
```

### Current fixed-separate evaluation

Small smoke test:

```powershell
python evaluate.py --method fixed-separate --limit 50
```

Evaluate one split with K=10:

```powershell
python evaluate.py --method fixed-separate --split test --top-k 10
```

Include full retrieved and evidence texts in JSONL:

```powershell
python evaluate.py --method fixed-separate --split test --store-text
```

## Reproducibility work still required

A later reproducibility guide should add, at minimum:

- supported Python, Docker, and operating-system versions;
- `.env.example` without credentials;
- exact package lock or fully pinned environment;
- Qdrant startup, health check, backup, and restore procedures;
- expected QASPER revision and split counts;
- configuration and secret-handling instructions;
- collection-integrity and checkpoint-validation commands;
- experiment manifests containing source revision, timestamps, seeds, K, model names, and artifact hashes;
- commands for each planned evaluator and router stage once implemented;
- automated tests and a clean end-to-end reproduction path.

## Completion definition

The roadmap is complete when ingestion is validated, all planned evaluators and router components exist with automated tests, full split-safe experiments have immutable record-level artifacts, and the final statistical comparison can be reproduced from a clean environment without undocumented manual steps. Until then, the only implemented evaluation method is fixed-separate retrieval with query-to-chunk cosine scores and aggregate top-K set-level token F1.
