# Experiment Roadmap

## Purpose and scope

This document records the implemented QASPER/Qdrant pipeline and the experiments that remain to be built. The current repository is the source of truth: items are called **completed** only when supported by the present code or generated artifacts. Proposed schemas, evaluators, router components, and analyses are explicitly marked **planned**.

The research objective is to measure how fixed token-level chunk granularity
affects evidence retrieval, establish ground-truth-informed supervision, and
test whether a router can choose a useful granularity from question-only
inference input. The legacy Logistic Regression and MLP routers use the stored
question embedding; the Qwen routers use only a fixed instruction and the
original question text.

## Metric and similarity vocabulary

Four quantities must remain distinct throughout implementation and reporting:

| Quantity | Inputs | Meaning | Current status |
|---|---|---|---|
| **Query-to-chunk retrieval similarity** | Question embedding and chunk embedding | Qdrant cosine score used to rank chunks for a question. This is an inference-time retrieval signal and does not use ground-truth evidence. | **Implemented.** Stored as `topk_scores`, with `avg_score_topk` and `best_score_topk`. |
| **Evidence-to-chunk oracle similarity** | Ground-truth evidence embedding and chunk embedding | Measures how directly a chunk aligns with annotated evidence. Because it uses evidence unavailable at deployment, it is an oracle/analysis signal rather than a deployable retrieval rule. | **Implemented in fixed-separate evaluation.** Stored separately for every chunk/evidence pair with per-chunk and top-K aggregates. |
| **Per-chunk F1** | One retrieved chunk's text and each ground-truth evidence passage | Token-overlap F1 for each individual chunk/evidence pair. It can expose whether one strong chunk is hidden inside a weak top-K set. | **Implemented in fixed-separate evaluation.** |
| **Top-K set-level F1** | All top-K retrieved chunk texts joined together and all evidence texts joined together | Aggregate token-overlap quality of the retrieved set. In the current code this is `f1_joined_topk`; it is not an average of per-chunk F1 values. | **Implemented.** |

The current token metrics normalize case, punctuation, and whitespace, tokenize with the configured Hugging Face tokenizer, and use multiset token overlap. The evaluator records per-chunk/per-evidence F1 plus precision, recall, and F1 for joined top-K retrieval against joined unique evidence.

**Oracle granularity selection must never be presented as a deployable
inference strategy.** Both saved Oracle definitions use ground-truth
highlighted evidence, which is available for supervised experiment
construction but not for a new deployment question. The legacy Oracle selects
the granularity with the highest joined retrieval F1. The later Qwen Oracle
selects the class nearest to the GPT-2 token length of complete deduplicated
evidence, with the smaller class winning exact midpoint ties and values clipped
to 10/160 at the range boundaries. The latter is independent of retrieval F1,
embedding quality, cosine similarity, retrieved chunks, and router
performance. Results using these two label definitions are not directly
comparable. A deployable router must decide from question-only input.

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

### 8. Per-chunk and aggregate evaluation

Every retrieved chunk is compared with every unique evidence passage using stored vectors and token F1. Query-to-chunk retrieval similarity and evidence-to-chunk cosine similarity remain separate. Per-chunk maximum/mean evidence similarity and maximum F1 are retained.

All retrieved texts at a given granularity are also joined, as are the unique evidence passages. Set-level token precision, recall, and F1 are computed between those combined strings. The legacy **top-K set-level F1** field `f1_joined_topk` remains as an alias of `set_level_f1`.

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

As inspected through 2026-06-22, the legacy local ingestion checkpoint
represents 1,585 papers, but not every stage is complete. This dated snapshot
does not supersede the later, file-backed Qwen Phase 1/Phase 2/Phase 2B/Phase
2C/Phase 2D/Phase 2E results:

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

- Only a legacy fixed-separate smoke-test evaluation exists in that dated
  ingestion workflow. Its sole populated evaluation file has 15 records: three
  questions evaluated at five granularities. The other evaluation file is
  empty.
- No full-split or full-dataset legacy fixed-separate baseline result has been
  produced and checked into a reproducible analysis workflow.
- Full-dataset legacy fixed-separate evaluation has not yet been run, although
  optional batched persistence exists through `RetrievalEvaluation`.
- Mixed-granularity raw and overlap-deduplicated evaluators now exist; full-split comparison remains pending.
- The old-Oracle dataset builder and leakage-safe embedding-router trainer
  exist, but no full-split live-collection embedding-router dataset has been
  generated. This is separate from the preserved evidence-length-Oracle files
  used by Qwen.
- No unified legacy embedding-router routed evaluator exists. The separate Qwen
  routed evaluator is implemented and completed for Phases 1, 2, both
  Phase 2B variants, Phase 2C, Phase 2D, and Phase 2E.
- Focused unit tests exist, but no live-Qdrant integration or complete end-to-end experiment test exists.
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

**Risks:** OpenAI cost/rate limits, partial Qdrant writes preceding checkpoint updates, stale artifacts created with different source revisions, and exact evidence text not mapping uniquely back to flattened paper text. Shared ingestion JSONL writes are now lock-protected.

### Milestone 1 — Separate-granularity oracle evaluation

**Status:** Implemented and smoke-tested; full-split execution remains planned.

**Goal:** Evaluate every question independently at all five fixed granularities and derive a ground-truth-informed best granularity per question.

The evaluator now retains query-to-chunk scores and top-K set-level metrics and implements:

- token F1 for every retrieved chunk against every unique evidence passage;
- evidence-to-chunk cosine similarity for every pair, with maximum and arithmetic-mean aggregates;

The implementation additionally provides:

- deterministic oracle labels based on a predeclared evidence-derived objective;
- tie indicators and the complete score vector over all granularities.

The primary target maximizes joined top-K F1, breaks epsilon ties with mean per-chunk maximum evidence similarity, then prefers the smaller chunk size. Evidence-to-chunk cosine and per-chunk F1 remain separate diagnostic/oracle features rather than being silently blended into the same number. See `docs/EVALUATION_PIPELINE.md` for the complete schema and commands.

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

**Status:** Implemented, unit-tested, and live-smoke validated with JSONL plus Qdrant persistence; full-split comparison remains pending.

**Goal:** Let chunks from all five granularities compete in one retrieval operation and compare the resulting top-K set with fixed-separate baselines.

The implementation exposes two separately named policies: `mixed-raw` preserves global score ranking, while `mixed-deduplicated` suppresses candidates whose character-span intersection divided by the shorter span length meets the configured threshold (default `0.8`). Because the same source span appears at multiple resolutions, the raw result may contain redundant nested chunks. Deduplication reduces this redundancy but can suppress a lower-ranked, more precisely bounded chunk; both policies must be reported separately.

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

**Status:** The legacy embedding-router code is implemented and tested; its
live-collection rerun remains blocked by the old-Oracle `RouterDataset`
snapshot. The separate Qwen evidence-length-Oracle path completed full Phase 2
and both Phase 2B training and validation runs, plus the Phase 2C, Phase 2D,
and Phase 2E Base-model sequence-classification runs, from preserved files.

**Goal:** Train a model that predicts the oracle granularity label from a question embedding without access to evidence at inference time.

The input is the stored question embedding. Labels come from the train-split oracle analysis in Milestone 1. Validation data selects model configuration and stopping criteria. Test labels are reserved for final evaluation and must not influence training or tuning.

The implemented framing is fixed five-class classification from question embeddings only. It compares a majority baseline with multinomial logistic regression. A small MLP is optional and becomes primary only when its validation macro F1 improves on logistic regression by the configured minimum. Details and commands are in `docs/GRANULARITY_ROUTER.md`.

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

**Status:** Partially complete. The Qwen routed evaluator is implemented and
has completed same-paper top-five retrieval for Phase 1, Phase 2, both
Phase 2B variants, Phase 2C, Phase 2D, and the locked Phase 2E winner. A
unified legacy embedding-router comparison evaluator remains planned.

**Goal:** For each question, predict one granularity from question-only input,
retrieve top K chunks only at that granularity, and evaluate the result without
using evidence until scoring. The legacy router uses the stored question
embedding. The completed Qwen evaluator uses the fixed instruction plus the
original question text.

The deployable path is:

```text
question-only router input -> router prediction -> selected granularity -> Qdrant top-K retrieval
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

### Qwen router phases

Phase 1 is complete for the pretrained `Qwen/Qwen3.5-0.8B`: separate
evidence-length Oracle generation, 924-question zero-shot validation,
classification reporting, and unchanged end-to-end retrieval are finished.
Accuracy/macro-F1/weighted F1 is 0.040043/0.049046/0.032613, all outputs are
valid, and mean joined retrieval F1 is 0.239109 at 100% coverage. The new
Oracle is strongly imbalanced toward class 160 (420/924, 45.45%), whereas Qwen
predicts class 10 for 767/924 examples. See `docs/QWEN_PHASE1_RESULTS.md`.

Phase 2 is complete. It fine-tuned the exact same model and revision using the
same fixed instruction, preserved 2,245/924 train/validation questions, and
same evidence-length Oracle as Phase 1. It remains a separate artifact tree and
does not overwrite or merge with the pretrained Phase 1 baseline.

The run `qwen-phase2-full-parameter-20260802-seed42-v2` used full-parameter
supervised fine-tuning: all 852,985,920 model parameters were trainable, with
no LoRA, QLoRA, adapters, prompt tuning, separate classification head,
quantization, evidence input, retrieval input, or handcrafted features. Loss
was restricted to the assistant target tokens. The deterministic configuration
was Python 3.10.7, Transformers `5.15.0.dev0` at commit
`2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`, PyTorch `2.8.0+cu128`, CUDA
BF16 on one `NVIDIA A100-SXM4-40GB`, seed 42, three epochs, batch size 4,
gradient accumulation 8 (effective batch size 32), learning rate 2e-5,
weight decay 0.01, cosine scheduling with 5% warmup, and gradient clipping
at 1.0. Training executed 213 parameter-update steps and evaluated the full
924-example validation split at the end of every epoch. `step-000213` was
selected by validation macro-F1.

Phase 2 validation accuracy/macro-F1/weighted F1/balanced accuracy is
0.4318181818181818/0.16502267760462996/0.32805741427623947/
0.20697865353037764. All 924 outputs are valid; top-2 accuracy remains
unavailable because generated text supplies no comparable five-class scores.
The Oracle distribution is 13/81/178/232/420 for 10/20/40/80/160, while the
prediction distribution is 0/0/0/149/775. The 160 class is 45.45% of
validation, and the fine-tuned router still collapses to the two largest
classes. It remains below the 160-majority accuracy baseline of
0.45454545454545453, although its macro-F1 exceeds the majority baseline of
0.125.

Unchanged downstream retrieval covered 924/924 predictions. Mean/median joined
retrieval F1 is 0.22658488852813854/0.19615549999999998 with `top_k=5` and
source-paper restriction. This downstream token-overlap metric is distinct
from Oracle-label classification metrics. Compared with Phase 1 under the
same new Oracle, Phase 2 improves accuracy, macro-F1, and weighted F1, but
decreases balanced accuracy and mean joined retrieval F1 (Phase 1 mean:
0.23910868506493507). No old-Oracle Logistic/MLP classification result is
presented as directly comparable.

Training, including epoch validation/checkpointing, took
2,107.3131887838244 seconds; reloaded final validation took
299.0685129035264 seconds including model loading; retrieval took
178.12831589998677 seconds. Peak allocated training GPU memory was
10.660949230194092 GiB. The TensorBoard audit reconciled all 213 training
steps and three validation events with zero required-value mismatches, the
selected checkpoint passed deterministic repeat verification, and the final
integrity audit passed every recorded check.

Authoritative results and provenance are in
`outputs/qwen_finetuned_router_evidence_length_oracle/final_summary.json`,
`outputs/qwen_finetuned_router_evidence_length_oracle/integrity_audit.json`,
and the run directory under
`outputs/qwen_finetuned_router_evidence_length_oracle/runs/`. Human-readable
results are in `docs/QWEN_PHASE2_RESULTS.md` and
`reports/qwen_finetuned_router_evidence_length_oracle/experiment_report.md`.

Recorded reproduction sequence:

```bash
.venv-qwen/bin/python qwen_phase2.py inspect-data
.venv-qwen/bin/python qwen_phase2.py train --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.venv-qwen/bin/python qwen_phase2.py audit-tensorboard --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.venv-qwen/bin/python qwen_phase2.py verify-checkpoint --run-id qwen-phase2-full-parameter-20260802-seed42-v2 --checkpoint outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoints/step-000213
.venv-qwen/bin/python qwen_phase2.py final-validation --run-id qwen-phase2-full-parameter-20260802-seed42-v2
```

With the unchanged local Qdrant service available:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2.py evaluate-retrieval --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.\.venv-qwen\Scripts\python.exe qwen_phase2.py audit-final --run-id qwen-phase2-full-parameter-20260802-seed42-v2
```

Phase 2B is also complete as two isolated full-parameter restricted-alias
experiments. Both preserve the same evidence-length Oracle, 2,245/924
train/validation questions, exact model revision, question-only input,
three-epoch/213-step schedule, and downstream retrieval setup. They replace
the variable-length numeric target with verified one-token aliases
`1→10`, `2→20`, `3→40`, `4→80`, and `5→160`; classification uses
deterministic argmax over exactly these five next-token logits. Phase 2B-A is
unweighted. Phase 2B-B changes only the loss weights within this pair, using
effective-number weights with `beta=0.999` derived from the training split.

| Metric | Phase 2B-A alias-unweighted | Phase 2B-B alias-classbalanced |
|---|---:|---:|
| Selected checkpoint | `step-000213` | `step-000142` |
| Accuracy | 0.35064935064935066 | 0.37012987012987014 |
| Macro-F1 | **0.20922603632601472** | 0.16836616836616836 |
| Weighted F1 | **0.3406050804511769** | 0.3142183142183142 |
| Balanced accuracy | **0.2383201416948027** | 0.20607553366174058 |
| Top-2 accuracy | 0.6071428571428571 | 0.7056277056277056 |
| Prediction distribution, 10/20/40/80/160 | 0/0/427/189/308 | 0/0/0/434/490 |
| Mean joined retrieval F1 | **0.28646775432900434** | 0.24962774025974027 |
| Median joined retrieval F1 | **0.2748425** | 0.223194 |

Both variants have 924/924 valid outputs and 100% retrieval coverage. Neither
predicts 10 or 20, and neither exceeds the 160-majority accuracy baseline of
0.45454545454545453. The weighted variant is worse than unweighted alias
training on the primary macro-F1, weighted F1, balanced accuracy, and mean
joined retrieval F1. This is a negative result for this predeclared weighting
scheme, not a general result about all class-imbalance methods.

The four-way same-new-Oracle comparison shows Phase 2 has the highest accuracy
(0.4318181818181818), whereas Phase 2B-A has the highest macro-F1, weighted
F1, balanced accuracy, and mean joined retrieval F1. Phase 2B-B has the highest
available top-2 accuracy; top-2 remains unavailable for Phase 1 and Phase 2.
Classification and downstream joined retrieval F1 remain distinct outcomes.
Phase 2B also changes prompt output schema, target encoding, and decision rule,
so Phase 2-to-2B differences cannot be attributed only to alias symbols.

These are single-seed experiments with checkpoint selection and final
reporting on the same validation split; the QASPER test split was not loaded.
The earlier Logistic Regression/MLP results use the old retrieval-F1 Oracle
and remain not directly comparable. Exact results and caveats are in
`docs/QWEN_PHASE2B_RESULTS.md`; authoritative run summaries are under
`outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/` and
`outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/`, with the
four-way comparison at
`outputs/qwen_phase2b_comparison_evidence_length_oracle/four_way_comparison.json`.

Recorded Phase 2B reproduction sequence:

```bash
.venv-qwen/bin/python qwen_phase2b.py inspect --variant alias-unweighted
.venv-qwen/bin/python qwen_phase2b.py inspect --variant alias-classbalanced
.venv-qwen/bin/python qwen_phase2b.py train --variant alias-unweighted --mode full --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py train --variant alias-classbalanced --mode full --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py final-validation --variant alias-unweighted --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py final-validation --variant alias-classbalanced --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
```

Against the unchanged local Qdrant service:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py evaluate-retrieval --variant alias-unweighted --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py evaluate-retrieval --variant alias-classbalanced --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py compare --output outputs\qwen_phase2b_comparison_evidence_length_oracle\four_way_comparison.json
```

Manual post-transfer checksum dry-runs verified both copied Phase 2B trees.
Phase 2B-A had no differences; the two Phase 2B-B preflight copies that
differed only by generated timestamps were replaced from the GPU source, after
which the targeted checksum rerun returned no differences. Retrieval replay
revalidated all 924 records per run, and Qdrant collection counts were
identical before and after. The selected A/B checkpoints contain 11 files
totaling 4,735,895,574/4,735,895,530 bytes. No standalone Phase 2B hash
inventory was saved, so no such artifact is claimed.

Phase 2C is complete. It implements the supervisor-motivated direct
sequence-classification path with the exact Base checkpoint
`Qwen/Qwen3.5-0.8B-Base`, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, rather than a chat-model
generation target. Its exact fixed instruction is:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = very short context, 2 = short context, 3 = medium context, 4 = long context, 5 = very long context. Return only the number

The model input is this prompt, two newlines, `Question: `, and the original
question text. `AutoModelForSequenceClassification` supplies five directly
comparable logits mapped to 10/20/40/80/160 and uses deterministic argmax.
There is no generation, chat template, assistant-token target, parser,
evidence, retrieval result, metadata, embedding, or handcrafted feature.

The run uses the unchanged 2,245/924 train/validation questions, exact frozen
evidence-length Oracle, seed 42, and downstream retrieval protocol. All
852,991,040 parameters were marked trainable and optimized with uniform
cross-entropy for three epochs and 213 updates. The language backbone and
classifier head received gradients; the 100,592,896-parameter vision tower did
not receive gradients on the text-only path. Epoch 3 `step-000213` was selected
by validation macro-F1.

Phase 2C accuracy/macro-F1/weighted F1/balanced accuracy/top-2 accuracy is
0.34523809523809523/0.21763191244497584/0.3435657773957275/
0.22993634120458348/0.6428571428571429. It predicts
0/20/224/374/306 against Oracle support 13/81/178/232/420. Class 10 has zero
recall. Class 20 recall is only 0.012345679012345678, with one correct example
out of 81. All 924 outputs are valid.

Unchanged same-paper `top_k=5` retrieval covers 924/924 questions. Mean/median
joined retrieval F1 is 0.27914719588744585/0.2607245. Training took
1276.56244828552 seconds; selected-checkpoint loading and isolated inference
took 2.5492455568164587 and 33.99719780869782 seconds; retrieval took
134.9306207000045 seconds. Known training plus final validation and retrieval
time is 1448.0395123510389 seconds.

At the completion of Phase 2C, the five-way evidence-length-Oracle comparison
placed Phase 2C first on saved Qwen macro-F1 at 0.21763191244497584. Phase 2
numeric SFT had the best accuracy at 0.4318181818181818. Phase 2B-A had the
best mean joined retrieval F1 at 0.28646775432900434, compared with Phase 2C
at 0.27914719588744585. These outcomes measure different stages; the Phase 2D
section below reports the expanded six-run comparison.

Phase 2C changes the checkpoint family, classifier formulation, and revised
prompt simultaneously. It is comparable on the preserved benchmark but is
not a clean causal architecture ablation. This is one seed with checkpoint
selection and reporting on validation; controlled one-factor follow-ups and
multiple seeds remain necessary. Old-Oracle Logistic Regression/MLP
classification results are not directly comparable.

Recorded Phase 2C execution sequence on the CUDA host:

```bash
.venv-qwen/bin/python qwen_phase2c_sequence_classifier.py --output-root /dev/shm/qwen_phase2c_sequence_classifier_evidence_length_oracle inspect
.venv-qwen/bin/python qwen_phase2c_sequence_classifier.py --output-root /dev/shm/qwen_phase2c_sequence_classifier_evidence_length_oracle train --mode full --run-id qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1
.venv-qwen/bin/python qwen_phase2c_sequence_classifier.py --output-root /dev/shm/qwen_phase2c_sequence_classifier_evidence_length_oracle final-validation --run-id qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1
```

Against the unchanged local Qdrant service:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2c_posttraining.py evaluate-retrieval --run-id qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2c_posttraining.py compare --output outputs\qwen_phase2c_comparison_evidence_length_oracle\five_way_comparison.json
```

Authoritative results are
`outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/final_summary.json`
and
`outputs/qwen_phase2c_comparison_evidence_length_oracle/five_way_comparison.json`;
human-readable reports are `docs/QWEN_PHASE2C_RESULTS.md` and
`reports/qwen_phase2c_sequence_classifier_evidence_length_oracle/experiment_report.md`.

#### Phase 2D exact-token prompt ablation

Phase 2D is complete as a controlled prompt-only follow-up to Phase 2C. It
uses the same `Qwen/Qwen3.5-0.8B-Base` revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, freshly initialized seed-42
five-logit classification head, frozen 2,245/924 examples, evidence-length
Oracle hashes, uniform cross-entropy, three-epoch/213-update schedule,
checkpoint-selection rule, and downstream retrieval identity. The one
semantic change replaces Phase 2C's qualitative context descriptions with
exact candidate token counts:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, 4 = 80 tokens, 5 = 160 tokens. Return only the number

The prompt SHA-256 is
`b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5`.
The saved protocol audit passes and proves equality of the non-prompt training
configuration and frozen dataset metadata. Prompt-caused tokenization changes
are recorded explicitly: Phase 2D train sequences span 95--121 tokens and
validation sequences 96--124, with no truncation at maximum length 128.

Phase 2D runs in Python 3.10.7 with Transformers `5.15.0.dev0` at commit
`2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`, PyTorch `2.8.0+cu128`, CUDA
12.8 BF16 on one `NVIDIA A100-SXM4-40GB`, no quantization, and seed 42. The
input remains the instruction plus original question only. The text backbone
and classifier head received gradients; the unused vision tower did not on
the text-only path. Epoch 3 `step-000213` was selected by validation macro-F1.

| Metric | Phase 2C | Phase 2D |
|---|---:|---:|
| Accuracy | 0.34523809523809523 | 0.36904761904761907 |
| Macro-F1 | 0.21763191244497584 | 0.22994524079282935 |
| Weighted F1 | 0.3435657773957275 | 0.3644656337102369 |
| Balanced accuracy | 0.22993634120458348 | 0.2391812745015638 |
| Top-2 accuracy | 0.6428571428571429 | 0.6341991341991342 |
| Mean joined retrieval F1 | 0.27914719588744585 | 0.2767166677489178 |
| Median joined retrieval F1 | 0.2607245 | 0.2558975 |

Phase 2D predicts 0/16/219/332/357 against Oracle support
13/81/178/232/420. All 924 outputs are valid, but class 10 is never predicted
and class 20 recall is only 0.024691358024691357. The exact-token wording
therefore improves accuracy, macro-F1, weighted F1, and balanced accuracy over
Phase 2C, but it does not resolve the class imbalance. Accuracy also remains
below the 160-majority baseline 0.45454545454545453.

Unchanged same-paper `top_k=5` retrieval covers 924/924 examples and yields
mean/median joined retrieval F1
0.2767166677489178/0.2558975. These retrieval-overlap scores are not
classification metrics: the modest Phase 2D classification improvement is
paired with a small mean retrieval decrease of 0.0024305281385280653 relative
to Phase 2C. At Phase 2D completion, the six-way comparison placed Phase 2D first on macro-F1,
weighted F1, and balanced accuracy; numeric Phase 2 remains first on accuracy,
and Phase 2B-A remains first on mean joined retrieval F1.

Training took 1224.5802961867303 seconds; selected-checkpoint loading and
isolated final inference took 2.7541816290467978 and 34.72815803065896
seconds; retrieval took 151.0063940999098 seconds. Known training, final
validation, and retrieval time is 1413.0690299463458 seconds. This is a clean
Phase 2C-to-Phase 2D one-factor comparison, but it remains a single seed with
checkpoint selection and reporting on validation. No held-out test result or
run-to-run variance is claimed. Earlier Qwen cross-phase differences remain
confounded, and old-Oracle Logistic Regression/MLP classification remains not
directly comparable.

Recorded Phase 2D execution sequence on the CUDA host:

```bash
.venv-qwen/bin/python qwen_phase2d_sequence_classifier.py --output-root /dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle inspect
.venv-qwen/bin/python qwen_phase2d_sequence_classifier.py --output-root /dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle train --mode full --run-id qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1
.venv-qwen/bin/python qwen_phase2d_sequence_classifier.py --output-root /dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle final-validation --run-id qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1
```

Against the unchanged local Qdrant service:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py evaluate-retrieval --run-id qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py compare --output outputs\qwen_phase2d_comparison_evidence_length_oracle\six_way_comparison.json
```

Authoritative Phase 2D results are
`outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/final_summary.json`;
the complete prompt-only protocol audit and six-run comparison are in
`outputs/qwen_phase2d_comparison_evidence_length_oracle/six_way_comparison.json`.
The Phase 2D experiment fingerprint is
`dad60bd9a0530865110c2310f62a896c73350fa383c7812d5c6733e376bc377d`.
The Phase 2D output tree retains configuration, per-epoch and canonical
predictions, classification and retrieval records, runtimes, the selected
checkpoint, and remote/local transfer verification.

#### Phase 2E predeclared learning-rate grid and five-epoch development selection

Phase 2E is complete. It evaluates three independent, fresh-from-Base
sequence-classifier runs at learning rates `5e-6`, `1e-5`, and `2e-5` for
five epochs each. All other within-grid settings are frozen: the Phase 2D
model and revision, exact-token prompt and prompt hash, preserved 2,245/924
examples and evidence-length-Oracle files, question-only input, five-logit
head, uniform cross-entropy, seed 42, maximum length 128, microbatch 4,
gradient accumulation 8, effective batch 32, weight decay 0.01, cosine
schedule, gradient clipping 1.0, BF16 CUDA, and no quantization. Each trial
has 355 optimizer updates, 18 warmup steps, and validation at steps
71/142/213/284/355. No trial continues from Phase 2D or another Phase 2E
trial.

The predeclared selector compares all 15 validation checkpoints
lexicographically by higher macro-F1, accuracy, weighted F1, and balanced
accuracy; lower cross-entropy; earlier optimizer step; and finally lower
numeric learning rate only for an exact remaining tie. The winner was locked
before retrieval. Retrieval could neither select nor revise it.

| Trial | Per-trial selected epoch/step | Selected macro-F1 | Recorded run seconds |
|---|---:|---:|---:|
| `5e-6` | 4 / 284 | 0.22777929657889012 | 2044.1943467836827 |
| `1e-5` | 4 / 284 | 0.21540884371375907 | 2022.7333836276084 |
| `2e-5` | 5 / 355 | 0.2252323080025679 | 2067.4948720689863 |

The global winner is the `5e-6` epoch-4 checkpoint `step-000284`. A clean
reload reproduced all 924 stored checkpoint predictions and metrics exactly.
Its final development-set classification results are:

| Metric | Phase 2E winner |
|---|---:|
| Accuracy | 0.3484848484848485 |
| Macro-F1 | 0.22777929657889012 |
| Weighted F1 | 0.3473258648868964 |
| Balanced accuracy | 0.24232226137689133 |
| Top-2 accuracy | 0.6190476190476191 |
| Uniform validation cross-entropy | 1.3759860497016412 |
| Valid predictions | 924/924 |

Oracle support is 13/81/178/232/420 and the winner predicts
0/15/275/366/268 for classes 10/20/40/80/160. It never predicts class 10;
class-20 recall is 0.037037037037037035. Accuracy remains below the
class-160 majority baseline 0.45454545454545453, while macro-F1 exceeds the
majority macro-F1 0.125. The selected classifier is slightly below Phase 2D
on macro-F1 by 0.0021659442139392304 and accuracy by 0.02056277056277056,
while balanced accuracy is higher by 0.003140986875327545. Phase 2E therefore
does not improve the primary classification metric over Phase 2D.

Only the locked winner received the unchanged downstream evaluation. The
existing Qdrant collections were read without rebuilding or re-indexing;
same-paper, predicted-granularity retrieval used `top_k=5`, the same
1,536-dimensional `text-embedding-3-small` vectors, cosine similarity, chunk
ordering and concatenation, and joined GPT-2 token-level F1. Coverage is
924/924. Mean, median, and coverage-adjusted full-set joined retrieval F1 are
0.2793735097402597, 0.267412, and 0.27937350974026. Retrieval took
282.3799051999813 seconds. Relative to Phase 2D, mean/median retrieval F1 are
higher by 0.00265684199134192/0.0115145; this is descriptive and does not
override the locked classification selection. Classification agreement with
the Oracle and downstream evidence-token overlap are different metrics.

The three recorded training runs total 6134.422602480277 seconds. Winner
loading and isolated inference take 2.47247052565217 and
33.506004774942994 seconds. Known grid training, winner reload/inference, and
retrieval time is 6452.78098298085 seconds (about 1 hour 47 minutes 33
seconds), excluding preflight, packaging, transfer, and documentation. Each
training run peaked at 9.0316162109375 GiB allocated GPU memory and
9.62109375 GiB reserved.

This is a development/model-selection result, not an unbiased final test
estimate. The same 924 official validation examples have been repeatedly
observed in earlier phases and now select among 15 Phase 2E checkpoints.
There is one seed and no variance, confidence interval, significance, or
QASPER-test claim. Within Phase 2E only learning rate differs between trials;
Phase 2E versus Phase 2D is not a pure learning-rate or extra-epoch ablation
because the cosine horizon and warmup changed from 213/11 to 355/18. Earlier
Logistic Regression and MLP classification results use the old retrieval-F1
Oracle and are not directly comparable.

The authoritative grid lock and completed global summary are
`outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/comparison/selected_trial.json`
and `comparison/selected_final_summary.json`; all 15 rows are in
`comparison/lr_grid_metrics.csv`. Detailed results and reproduction commands
are in `docs/QWEN_PHASE2E_RESULTS.md` and
`reports/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/experiment_report.md`.
The retained study also includes all three selected checkpoints, their
content-addressed transfer manifests, canonical winner predictions,
classification artifacts, and 924 record-level retrieval results. The final
post-retrieval audit verifies the transfer bundle 13/13, all checkpoint files
27/27, 62 unchanged transfer-manifest metadata files plus exactly two
authorized retrieval-summary updates, and zero forbidden payloads. It is saved
as `integrity/final_post_retrieval_audit.json`.

The Qwen interpreter audit, separate-environment rationale, minimal dependency
manifest, exact package lock, and recreation commands are recorded in
`docs/QWEN_ENVIRONMENT.md`. The legacy `.venv` remains separate and unchanged.

The roadmap is complete when ingestion is validated, all planned evaluators
and router components exist with automated tests, full split-safe experiments
have immutable record-level artifacts, and the final statistical comparison
can be reproduced from a clean environment without undocumented manual steps.
Fixed-separate, both Oracle-label paths, mixed retrieval, leakage-safe legacy
router code, and complete Qwen Phase 1/Phase 2/Phase 2B/Phase 2C/Phase 2D/Phase 2E
classification and routed retrieval now exist. The broader matched-method
final analysis and any
same-new-Oracle retraining of Logistic Regression/MLP remain unfinished.
