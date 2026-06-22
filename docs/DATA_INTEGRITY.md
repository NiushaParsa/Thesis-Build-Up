# Data Integrity Audit

## Scope

This audit covers the active QASPER/Qdrant implementation as inspected on 2026-06-21:

- Qdrant collection schemas and payload indexes;
- ingestion, deterministic IDs, checkpointing, parallelism, and JSONL export;
- checkpoint contents and live Qdrant records;
- evaluation method registry, token-level F1, and evaluation JSONL format;
- focused automated tests for the core ingestion/evaluation contracts.

The audit and the new `validate_data.py` command are non-destructive. They do not create, recreate, update, or delete Qdrant collections. Existing `checkpoint.json`, `json_output/`, and evaluation outputs were not modified.

## Active data model

All three Qdrant collections use 1,536-dimensional vectors and cosine distance. Vectors and HNSW data are configured on disk.

| Collection | Purpose | Indexed payload fields | Current points |
|---|---|---|---:|
| `PaperChunk` | Fixed-size document chunks | `document_id`, `granularity_level` | 1,701,822 |
| `PaperQuestion` | Answerable QASPER questions | `document_id`, `split` | 4,526 |
| `PaperEvidence` | Deduplicated highlighted evidence | `question_id`, `document_id` | 9,522 |

Chunk point IDs are deterministic UUID-5 values derived from document ID, granularity level, and chunk index. Question point IDs are derived from document ID and the original QASPER question ID. Evidence point IDs are derived from the question UUID and a truncated MD5 of the evidence text.

Qdrant itself enforces point-ID uniqueness. The validator additionally checks for repeated deterministic payload keys and verifies that chunk and evidence IDs match the IDs recomputed from their payloads. Question IDs can only be fully recomputed when `--qasper-check` loads the source dataset, because `original_question_id` is not stored in the Qdrant question payload.

## Ingestion findings and fixes

### Shared JSONL writer

Paper workers previously shared the three optional JSONL file handles without synchronization. Concurrent calls could interleave writes or split one paper's record group across another worker's output.

`prepare_dataset.py` now routes JSONL writes through one lock-protected helper. Each record group is serialized while holding the lock. Qdrant upserts and ignored exports were not rewritten.

### Chunk statistic

`process_paper_chunks` already incremented `chunks_inserted` by the actual number of inserted chunks. The paper worker then incremented the same counter once more for every completed granularity. That second increment was removed, so the statistic now represents chunk points inserted during the current process.

### Checkpoint semantics

The checkpoint tracks seven stages per paper: chunk sizes 10, 20, 40, 80, and 160, followed by `questions` and `evidence`. Writes remain lock-protected and use temporary-file replacement. Ingestion remains resumable and only `--recreate` requests destructive collection recreation.

The standalone schema command is now non-destructive by default:

```powershell
.\.venv\Scripts\python.exe qdrant_schema.py
```

It creates only missing collections. Destructive schema replacement now requires the explicit `qdrant_schema.py --recreate` flag.

## Exact incomplete checkpoint entries

Four of 1,585 documents have incomplete checkpoint stages:

| Document | Split | Missing stages |
|---|---|---|
| `1701.04056` | validation | `questions`, `evidence` |
| `1909.07575` | validation | `80`, `160`, `questions`, `evidence` |
| `1911.00133` | train | `80`, `160`, `questions`, `evidence` |
| `1911.03648` | train | `questions`, `evidence` |

Stage totals are:

| Stage | Complete papers |
|---|---:|
| 10 | 1,585 |
| 20 | 1,585 |
| 40 | 1,585 |
| 80 | 1,583 |
| 160 | 1,583 |
| questions | 1,581 |
| evidence | 1,581 |

`1911.03648` has no QASPER question with non-empty highlighted evidence, so the current filtering logic expects no question or evidence points for it. Its checkpoint markers are nevertheless absent. The other three interrupted documents account for the missing stored records described below.

## Live Qdrant validation results

The full payload scan checked all 1,701,822 chunks, all 4,526 questions, and all 9,522 evidence points.

### Chunks

- 7,921 document/granularity groups are present. This equals 1,585 papers at five levels minus the four missing 80/160 stages listed above.
- No chunk is missing from a group that the checkpoint marks complete.
- No missing chunk index was found inside an existing group.
- No inconsistent `total_chunks`, out-of-range index, malformed payload, repeated deterministic key, or deterministic-ID mismatch was found.

### Questions and evidence

A source-QASPER comparison derives 4,532 expected answerable-question IDs and 9,538 expected evidence IDs. Qdrant is missing exactly six question points and sixteen evidence points; it has no unexpected question or evidence IDs.

Missing questions:

| Document | Split | Question |
|---|---|---|
| `1909.07575` | validation | What are the baselines? |
| `1909.07575` | validation | What is the attention module pretrained on? |
| `1911.00133` | train | What categories does the dataset come from? |
| `1911.00133` | train | Is the dataset balanced across categories? |
| `1911.00133` | train | What supervised methods are used? |
| `1911.00133` | train | What labels are in the dataset? |

The sixteen missing evidence points are distributed as follows:

| Document | Missing evidence points | Explanation |
|---|---:|---|
| `1701.04056` | 2 | Its question point exists, but evidence ingestion did not complete. |
| `1909.07575` | 9 | Both expected question points and their evidence are absent. |
| `1911.00133` | 5 | All four expected question points and their evidence are absent. |

The existing question for `1701.04056` is the sole question currently lacking linked evidence. No evidence point refers to a missing question point.

Forty-six QASPER papers have no question with highlighted evidence. Forty-five of them are checkpointed for question/evidence processing and correctly contain no stored question or evidence. They are not missing-data errors. The remaining one is `1911.03648`, whose checkpoint stages are incomplete despite having no points to ingest.

Two pairs of distinct QASPER questions repeat the same text within one document:

- `2004.03090`: “Which baselines did they compare to?”
- `1912.13109`: “What dataset is used?”

They have distinct source question IDs and deterministic UUIDs, so they are repeated text, not duplicate deterministic IDs.

### Evidence offsets

1,345 of 9,522 evidence points have `span_start == -1` or `span_end == -1`. Their evidence strings were not found by exact substring search in the flattened document text. These records remain usable as evidence text and vectors, but their character-location metadata is unresolved.

The validator reports the total and configurable examples. To emit every affected ID, set `--example-limit` to at least 1345 and optionally write the report with `--json-output`.

### Vectors

- Every collection schema reports the expected dimension of 1,536.
- A sample of 1,000 vectors from each collection contained no missing, wrong-dimension, non-finite, or all-zero vectors.
- This is not a full point-by-point vector proof for the 1.7-million-point chunk collection. `--full-vector-scan` is available, but it transfers all vectors and is intentionally not the default.

The local Python Qdrant client is 1.16.1 while the Docker service is 1.13.2. The client emits a compatibility warning because that minor-version gap is larger than supported. The read-only audit succeeded, but versions should be aligned before relying on long production runs.

## Evaluation audit

The registry exposes `fixed-separate`, `mixed-raw`, and `mixed-deduplicated`. Fixed-separate retrieves top K independently per level. Both mixed methods use a source-document-only filter so all levels compete globally; the deduplicated method applies the documented character-span overlap policy.

The metric implementation normalizes case, punctuation, and whitespace, tokenizes with the configured Hugging Face tokenizer, and computes multiset token precision/recall/F1. Schema version 2 adds per-chunk comparison against every unique evidence passage while retaining the original aggregate fields.

### Fixed-separate JSONL schema version 2

Each line remains one question/granularity result. Existing schema-version-1 fields are preserved so current CLI consumers continue to work:

```text
eval_id, method_name, question_id, document_id, split,
granularity_level, granularity_tokens, k_requested, retrieved_k,
retrieval_time_ms, evidence_hash, evidence_token_count,
retrieved_joined_token_count, topk_chunk_ids, topk_chunk_indices,
topk_scores, f1_joined_topk, avg_score_topk, best_score_topk
```

Version 2 adds these top-level fields:

| Field | Definition |
|---|---|
| `schema_version` | Integer `2`. Records without this field are treated as legacy version 1. |
| `returned_k` | Number of chunks actually returned; alias of `retrieved_k`. May be smaller than `k_requested`. |
| `retrieval_latency_ms` | Qdrant query latency; alias of `retrieval_time_ms`. |
| `unique_evidence_count` / `unique_evidence_ids` | Non-empty evidence passages after normalization-based deduplication and deterministic point-ID ordering. |
| `joined_unique_evidence_token_count` | Token count of unique evidence texts joined with newlines; alias-compatible with `evidence_token_count`. |
| `joined_retrieved_text_token_count` | Token count of returned chunk texts joined in rank order; alias-compatible with `retrieved_joined_token_count`. |
| `set_level_precision`, `set_level_recall`, `set_level_f1` | Multiset token metrics for joined retrieved text against joined unique evidence. `set_level_f1` equals legacy `f1_joined_topk`. |
| `best_query_similarity_topk`, `mean_query_similarity_topk` | Maximum and arithmetic mean of Qdrant question-to-chunk scores. They equal the legacy best/average score fields. |
| `best_evidence_similarity_topk`, `mean_evidence_similarity_topk` | Maximum and arithmetic mean over every returned-chunk × unique-evidence cosine pair. |
| `retrieved_chunks` | Rank-ordered per-chunk metric objects described below. |

Each `retrieved_chunks` object contains:

```text
chunk_id, chunk_idx, granularity_level, granularity_tokens,
chunk_token_count, rank, query_similarity,
evidence_cosine_similarities[], max_evidence_similarity,
mean_evidence_similarity, evidence_token_f1_scores[], max_chunk_f1
```

`evidence_cosine_similarities` and `evidence_token_f1_scores` contain one object per unique evidence passage and include its `evidence_id`. Query similarity and evidence similarity are separate fields: the former is Qdrant's question-to-chunk retrieval score; the latter is recomputed from the stored chunk and evidence vectors.

Evidence behavior is deterministic:

- empty/whitespace-only evidence is ignored;
- duplicate evidence after lowercase/punctuation/whitespace normalization is evaluated once;
- if duplicate text exists under multiple point IDs, the lexicographically smallest point ID supplies the retained stored vector;
- a question with no remaining evidence yields no evaluation records, preserving prior behavior;
- zero returned chunks still yield a record with `returned_k = 0` and zero aggregate similarities and token metrics;
- zero-length normalized prediction or evidence text produces zero precision, recall, and F1;
- missing dense vectors or mismatched vector dimensions raise an explicit error instead of mixing incomparable values.

With `--store-text`, each chunk object additionally contains `text`, and the top-level record retains `retrieved_text` and `evidence_text`.

Existing output findings:

- `RetrievalEvalFixedSeparate_20260227_172957.jsonl` is empty.
- `RetrievalEvalFixedSeparate_20260227_173250.jsonl` contains 15 valid records for three questions across five granularities.
- These historical files use the legacy schema; newly generated output uses schema version 2.
- No invalid JSON, missing required fields, or duplicate `eval_id` values were found.

## Validation commands

All commands below are read-only with respect to Qdrant and existing dataset exports.

Quick smoke check with limited payload and vector coverage:

```powershell
.\.venv\Scripts\python.exe validate_data.py --max-points 1000 --vector-sample 100 --no-fail
```

Full payload scan with sampled vector checks:

```powershell
.\.venv\Scripts\python.exe validate_data.py --vector-sample 1000
```

Compare every expected question/evidence deterministic ID against source QASPER. This can require Hugging Face network access:

```powershell
.\.venv\Scripts\python.exe validate_data.py --qasper-check --vector-sample 1000
```

Full payload and full vector scan:

```powershell
.\.venv\Scripts\python.exe validate_data.py --qasper-check --full-vector-scan
```

Write a new report file without changing Qdrant:

```powershell
.\.venv\Scripts\python.exe validate_data.py --qasper-check --json-output tmp\data_integrity_report.json --no-fail
```

The command exits with status 1 when it finds integrity problems. `--no-fail` is useful for exploratory audits that should still produce a successful shell status. A limited `--max-points` scan labels its coverage incomplete and does not make full-dataset relationship claims.

## Automated tests

Run the focused suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite covers:

- stable and seed-sensitive ingestion/evaluation UUIDs;
- checkpoint save, reload, duplicate-stage suppression, and resume decisions;
- fixed non-overlapping token chunks, spans, final short chunk, and reconstruction;
- filtering of empty or missing highlighted evidence;
- normalized multiset token precision, recall, and F1, including normalized-empty text;
- manually verifiable cosine similarity and zero-vector behavior;
- multiple and duplicate evidence passages, stored evidence vectors, and per-chunk metrics;
- empty evidence, fewer-than-K results, and zero returned chunks;
- grouped, lock-protected JSONL writes under concurrent workers.

Current result: twenty-five focused tests pass, including mixed-variant registration, mixed-level eligibility, overlap suppression/backfill, collection-schema, and persistence idempotency coverage. The complete schemas, overlap policy, oracle rules, configuration, and validation commands are documented in `docs/EVALUATION_PIPELINE.md`.

## Remaining data problems

Before a full evaluation, the remaining actionable data issues are:

1. Resume ingestion for the four incomplete checkpoint entries without `--recreate`.
2. Verify that the resumed run adds six question points and sixteen evidence points while leaving the 46 zero-answerable papers without points.
3. Decide whether evidence offsets are required by later experiments. If so, replace or supplement exact substring matching and repair the 1,345 unresolved records through an explicit migration rather than silently rewriting them.
4. Align Qdrant client and server versions.
5. Run a full-vector scan if exhaustive detection of zero/non-finite vectors is required; current evidence is schema validation plus a 1,000-vector sample per collection.

No collection recreation is needed for validation or ordinary checkpoint resume.
