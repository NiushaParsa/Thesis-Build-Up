# Separate-Granularity Evaluation and Oracle Dataset

## Scope and audited behavior

This document describes the schema-v2 QASPER/Qdrant implementation. It covers fixed-separate evaluation, the offline oracle dataset, the explicitly separated `mixed-raw` and `mixed-deduplicated` retrieval strategies, and validation-first routed retrieval. Router training is documented separately in `docs/GRANULARITY_ROUTER.md`.

`PaperChunk`, `PaperQuestion`, and `PaperEvidence` use cosine distance. A Qdrant chunk-search score is therefore question-to-chunk similarity. The evaluator never treats that score as evidence similarity. Evidence similarity is recomputed from the returned chunk vector and each ground-truth evidence vector. The audited local `PaperEvidence` collection contains 1,536-dimensional vectors; when a stored evidence vector is missing, the evaluator calls the configured embedding service and rejects incompatible dimensions with a clear error.

Every configured granularity is evaluated independently inside the question's document. A router record is created only when all configured levels complete successfully.

## Mixed-granularity retrieval

Both mixed variants perform one Qdrant search filtered only by the question's `document_id`; they intentionally omit the granularity filter so all configured chunk sizes compete in the same score ranking.

- `mixed-raw` stores the normal global top K without overlap suppression.
- `mixed-deduplicated` greedily walks globally ranked candidates and retains a candidate only when it does not strongly overlap an already retained higher-scoring chunk. It fetches `K * MIXED_DEDUP_CANDIDATE_MULTIPLIER` candidates so suppressed results can be backfilled.

The overlap ratio is:

```text
intersection character length / shorter span character length
```

The default threshold is `0.8`. Consequently, duplicate spans and a small chunk nested inside a larger chunk have ratio `1.0` and the lower-ranked candidate is suppressed. Partial overlaps below the threshold remain eligible. Chunks with missing or invalid offsets cannot be compared safely and are retained. This policy reduces redundant multi-level results but may remove a smaller chunk whose boundaries are more precise; raw and deduplicated results must therefore be reported as different methods.

Mixed records store `topk_granularity_levels`, `topk_granularity_tokens`, and a five-entry `granularity_composition`. Every composition entry contains the chunk count, final ranks, and original candidate ranks for that size. Deduplicated records additionally store the threshold, candidate-pool details, suppressed count, and suppression relationships.

## Router-selected retrieval

`router-selected` loads a persisted granularity-router artifact and predicts exactly one chunk size from the stored question embedding. It then searches only the source document and only the predicted granularity level. Evidence and oracle labels are not used to choose the granularity; they are loaded after prediction for evaluation metrics and validation analysis only.

Router-selected records use method name `router-selected granularity`, write timestamped `RetrievalEvalRouterSelected_<timestamp>.jsonl` files, and can be persisted to `RetrievalEvaluation` as payload-only records. The routed evaluation configuration hash includes the router artifact hash, selected model type, top K, collection names, metric/tokenizer versions, and the frozen oracle configuration hash used for validation joins.

Additional routed fields include `router_model_path`, `router_model_hash`, `router_model_version`, `predicted_granularity_level`, `predicted_granularity_tokens`, `prediction_confidence`, `class_probabilities`, `router_latency_ms`, `retrieval_latency_ms`, and `total_latency_ms`. When a matching `RouterDataset` oracle record is available, the evaluator also stores `oracle_target_granularity`, `oracle_best_granularity_by_f1`, `oracle_best_granularity_by_evidence_similarity`, `router_oracle_match`, `oracle_best_f1`, `oracle_best_evidence_similarity`, `regret_f1`, and `regret_evidence_similarity`. Missing oracle records are explicit via `oracle_lookup_status`.

## Metric definitions

Text normalization lowercases text, removes punctuation, and collapses whitespace. The configured Hugging Face tokenizer then produces token IDs. Precision, recall, and F1 use multiset token overlap:

```text
overlap = sum(min(predicted_count[token], reference_count[token]))
precision = overlap / predicted_token_count
recall = overlap / reference_token_count
F1 = 2 * precision * recall / (precision + recall)
```

An empty normalized prediction or reference produces zero precision, recall, and F1.

- `query_chunk_similarity` is the Qdrant score from question-vector retrieval.
- Each chunk's `evidence_cosine_similarities` contains cosine similarity against every unique evidence vector. `max_evidence_similarity` and `mean_evidence_similarity` aggregate those values.
- `evidence_token_f1_scores` contains precision, recall, and F1 against every unique evidence passage. `max_chunk_f1`, `mean_chunk_f1`, and the precision/recall associated with the maximum are stored separately.
- `precision_joined_topk`, `recall_joined_topk`, and `f1_joined_topk` compare rank-ordered joined chunk text once with joined unique evidence text. Joined F1 is not an average of chunk F1 values.
- `mean_max_evidence_similarity_topk` is the mean, across returned chunks, of each chunk's maximum evidence similarity. This is the similarity aggregate used for oracle labeling.

## Deterministic evidence handling

Evidence points are ordered by point ID. Empty or whitespace-only text is discarded, then text is deduplicated by exact equality after the evaluator's documented normalization. The first point in deterministic order supplies the retained ID and vector. Records store `raw_evidence_count`, `valid_evidence_count`, `unique_evidence_count`, retained IDs, vector sources, a SHA-256 hash of joined evidence, and its token count. Questions with no valid evidence are skipped and counted without terminating the run.

## Evaluation record schema

Each question/granularity JSON object contains:

- schema, method, evaluation ID, run ID, configuration hash, timestamp;
- question ID, document ID, split, granularity level and token size;
- requested/returned K, Qdrant latency, aligned chunk IDs, indices, ranks, spans, token counts, and query scores;
- the evidence counts, IDs, vector sources, hash, and token count;
- rank-ordered `retrieved_chunks` with query similarity, all evidence similarities, and all per-evidence token metrics;
- query, evidence-similarity, per-chunk-F1, and joined-top-K aggregates;
- embedding model/dimension, tokenizer identity, metric version, and normalization version.

Arrays are aligned by retrieval rank. Fewer than K results, including zero results, produce valid records with the actual `returned_k` and zero-valued empty aggregates.

The deterministic evaluation ID is UUID-5 over:

```text
method|question_id|granularity|evaluation_config_hash
```

The hash covers the method, K, chunk mapping, embedding model/dimension, tokenizer, metric and normalization versions, all collection names, document/granularity filter behavior, schema version, text-storage mode, label version, and tie epsilon. CLI values override `.env`/`config.py`; configuration values override code fallbacks.

## Qdrant collections and JSONL artifacts

- `PaperChunk`: cosine vectors and chunk payloads; searched with exact `document_id` and `granularity_level` filters.
- `PaperQuestion`: cosine question vectors and question payloads. Its existing vector is reused for router records.
- `PaperEvidence`: evidence payloads and normally cosine evidence vectors.
- `RetrievalEvaluation`: optional payload-only schema-v2 evaluation records.
- `RouterDataset`: optional cosine question vectors with question-level oracle payloads.

Collections are created only when persistence is enabled and missing; existing collections are not deleted or recreated. Upserts are batched. A persistence failure is logged and disables further writes for that destination without corrupting the already independent JSONL output.

Every run writes timestamped files:

```text
RetrievalEvalFixedSeparate_<timestamp>.jsonl
RetrievalEvalMixedRaw_<timestamp>.jsonl
RetrievalEvalMixedDeduplicated_<timestamp>.jsonl
RetrievalEvalRouterSelected_<timestamp>.jsonl
RouterDataset_<timestamp>.jsonl
IncompleteEvaluation_<timestamp>.jsonl
```

## Oracle/router labels

For each complete question:

1. `best_granularity_by_f1` is the level with maximum `f1_joined_topk`.
2. `best_granularity_by_evidence_similarity` is the level with maximum `mean_max_evidence_similarity_topk`.
3. `router_target_granularity` first maximizes joined F1; values within `ROUTER_LABEL_TIE_EPSILON` are tied. Evidence similarity breaks that tie, then the smaller chunk size breaks any remaining tie.

Both analytical labels are stored even when they disagree. The reason for the selected target is recorded. This is an oracle label because it uses ground-truth evidence: it is an offline upper bound and supervision source, not a deployable inference strategy. Later routing must use inference-time features such as the question embedding.

## Configuration

See `.env.example`. Evaluation-specific variables are `RETRIEVAL_TOP_K`, `MIXED_DEDUP_OVERLAP_THRESHOLD`, `MIXED_DEDUP_CANDIDATE_MULTIPLIER`, `ROUTER_LABEL_TIE_EPSILON`, `EVALUATION_OUTPUT_DIR`, `EVALUATION_UPSERT_BATCH_SIZE`, `PERSIST_EVALUATIONS`, and `PERSIST_ROUTER_DATASET`. Collection, Qdrant, embedding, tokenizer, dimension, and `CHUNK_SIZES` settings are shared with ingestion.

## PowerShell commands

Smoke test without Qdrant evaluation writes:

```powershell
.\.venv\Scripts\python.exe evaluate.py --limit 1 --output-dir tmp\eval_smoke --no-persist-evaluations --no-persist-router-dataset
```

Smoke test with batched persistent records:

```powershell
.\.venv\Scripts\python.exe evaluate.py --limit 1 --output-dir tmp\persist_smoke --persist-evaluations --persist-router-dataset --upsert-batch-size 25
```

Complete one split:

```powershell
.\.venv\Scripts\python.exe evaluate.py --split train --top-k 5 --persist-evaluations --persist-router-dataset
```

Run the two mixed variants independently:

```powershell
.\.venv\Scripts\python.exe evaluate.py --method mixed-raw --split test --top-k 5 --persist-evaluations
.\.venv\Scripts\python.exe evaluate.py --method mixed-deduplicated --split test --top-k 5 --overlap-threshold 0.8 --persist-evaluations
```

Run validation-only router-selected retrieval against the frozen oracle dataset:

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --method router-selected `
  --split validation `
  --top-k 5 `
  --router-model models\granularity_router\frozen_topk5\router_model.pt `
  --evaluation-config-hash 9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8 `
  --output-dir outputs\router_selected\validation `
  --persist-evaluations `
  --log-every 100
```

Run the final validation comparison report without recomputing retrieval:

```powershell
.\.venv\Scripts\python.exe compare_retrieval_strategies.py `
  --output-dir reports\final_validation_comparison `
  --bootstrap-iterations 2000 `
  --bootstrap-seed 13
```

The comparison script reads validation JSONL/Qdrant records for the frozen fixed-separate oracle hash and routed validation hash. It writes `comparison_summary.md`, `comparison_table.csv`, `per_question_comparison.csv`, `strategy_metrics.json`, and `bootstrap_results.json` when all required validation strategies are available. If required mixed-granularity validation records are missing or incomplete, it stops and writes `missing_inputs.json` plus a `comparison_summary.md` that lists exactly what is absent; it does not fabricate mixed numbers or evaluate the test split.

Validate explicit artifacts, then also check persistent collections:

```powershell
.\.venv\Scripts\python.exe validate_evaluation.py --evaluation-jsonl outputs\RetrievalEvalFixedSeparate_<timestamp>.jsonl --router-jsonl outputs\RouterDataset_<timestamp>.jsonl
.\.venv\Scripts\python.exe validate_evaluation.py --evaluation-jsonl outputs\RetrievalEvalFixedSeparate_<timestamp>.jsonl --router-jsonl outputs\RouterDataset_<timestamp>.jsonl --check-qdrant
```

Run focused tests and syntax checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m py_compile config.py evaluate.py evaluation_utils.py fixed_sized_granularity_separate.py mixed_granularity.py router_selected.py metrics.py qdrant_schema.py validate_evaluation.py
```

The validator checks deterministic IDs, rank-aligned arrays, finite metrics, complete configured levels per router example, vector dimensions, split preservation, and label availability.
