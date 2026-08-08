# Qwen Phase 2 supervised fine-tuning results

Phase 2 completed full-parameter supervised fine-tuning of the supervisor-confirmed
`Qwen/Qwen3.5-0.8B` router. It used the same preserved 2,245/924
official-QASPER train/validation questions, evidence-length Oracle, five labels,
fixed instruction, parser, and downstream retrieval method as Phase 1. The
authoritative run is `qwen-phase2-full-parameter-20260802-seed42-v2`.

All 852,985,920 parameters were trainable (100%). No LoRA, QLoRA, adapter,
prompt tuning, classification head, quantization, data resampling, or
hyperparameter search was used. The model received only the fixed routing
instruction and original question text. Training loss was restricted to the
assistant class string and required assistant-ending tokens; evidence, evidence
length, answers, paper text, retrieval data, embeddings, metadata, and
handcrafted features were not model inputs.

## Data and Oracle

The frozen Oracle counts GPT-2 tokens in the complete deduplicated ground-truth
evidence and chooses the nearest of 10, 20, 40, 80, and 160. Exact midpoint
ties choose the smaller candidate; values below 10 map to 10 and values above
160 map to 160. It is independent of retrieval F1, embeddings, retrieved
chunks, and router performance.

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

The validation Oracle is strongly imbalanced: class 160 is 420/924 (45.45%),
while class 10 is 13/924 (1.41%). The frozen train/validation Oracle SHA-256
values are `64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`
and `ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`.

## Model, environment, and training

- Base model revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Environment: `.venv-qwen`; Python 3.10.7; Transformers `5.15.0.dev0`
  at commit `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`.
- PyTorch: `2.8.0+cu128`; CUDA 12.8; `torch.bfloat16`; no quantization.
- Hardware: one NVIDIA A100-SXM4-40GB; driver 570.133.20.
- Sequence length: maximum 128; the formatted training set ranges from
  89--115 tokens (mean 95.40979955456571); targets use 4 or 5 tokens.
- Optimizer: AdamW; learning rate `2e-5`; weight decay `0.01`; cosine
  scheduler; 5% warmup; gradient clipping `1.0`.
- Per-device batch 4; gradient accumulation 8; effective batch 32. The final
  five-example accumulation group in each epoch was weighted per example.
- Three fixed epochs, 71 optimizer steps per epoch, 213 parameter-update
  steps; seed 42; strict deterministic CUDA algorithms; no early stopping.
- Full validation and a resumable checkpoint occurred after every epoch. All
  three epoch checkpoints were retained during training. Selection used
  validation macro-F1, then accuracy, weighted F1, balanced accuracy, lower
  validation loss, and earlier step.

| Epoch | Step | Validation loss | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Predicted 10/20/40/80/160 | Validation wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 71 | 0.304930741717289 | 0.45454545454545453 | 0.125 | 0.2840909090909091 | 0.2 | 0/0/0/0/924 | 340.64150045998394 |
| 2 | 142 | 0.30652774386591725 | 0.4556277056277056 | 0.12698176971774003 | 0.28686332944508014 | 0.20086206896551725 | 0/0/0/3/921 | 327.1569862253964 |
| 3 | 213 | 0.3081164950932259 | 0.4318181818181818 | **0.16502267760462996** | 0.32805741427623947 | 0.20697865353037764 | 0/0/0/149/775 | 321.21750357560813 |

Epoch 3 (`step-000213`) was selected because it has the highest validation
macro-F1. The initial/final recorded training losses were
0.4739701375365257/0.4026994347572327. The higher selected validation loss and
lower accuracy do not override the predeclared primary metric.

## Final classification results

The selected checkpoint was reloaded and deterministically evaluated on all
924 validation questions. Its outputs exactly matched the 924 outputs saved at
the epoch-3 validation event.

| Metric | Phase 2 | Phase 1 zero-shot | Evidence-length majority baseline |
|---|---:|---:|---:|
| Accuracy | 0.4318181818181818 | 0.04004329004329004 | 0.45454545454545453 |
| Macro-F1 | 0.16502267760462996 | 0.049045932422555796 | 0.125 |
| Weighted F1 | 0.32805741427623947 | 0.032612933907418644 | 0.2840909090909091 |
| Balanced accuracy | 0.20697865353037764 | 0.23369399361908724 | 0.2 |
| Valid outputs | 924/924 | 924/924 | 924/924 |
| Invalid outputs | 0 (0.0%) | 0 (0.0%) | 0 |

Top-2 accuracy is unavailable: deterministic generated text does not provide
five directly comparable class scores.

| Class | Precision | Recall | F1 | Oracle support | Phase 2 predictions |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 | 0 |
| 20 | 0.0 | 0.0 | 0.0 | 81 | 0 |
| 40 | 0.0 | 0.0 | 0.0 | 178 | 0 |
| 80 | 0.2953020134228188 | 0.1896551724137931 | 0.23097112860892388 | 232 | 149 |
| 160 | 0.45806451612903226 | 0.8452380952380952 | 0.5941422594142259 | 420 | 775 |

The confusion matrix below has Oracle rows and prediction columns ordered
10, 20, 40, 80, 160.

| Oracle ↓ / predicted → | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 0 | 1 | 12 |
| 20 | 0 | 0 | 0 | 16 | 65 |
| 40 | 0 | 0 | 0 | 23 | 155 |
| 80 | 0 | 0 | 0 | 44 | 188 |
| 160 | 0 | 0 | 0 | 65 | 355 |

Fine-tuning substantially improved accuracy, macro-F1, and weighted F1 over
the same-new-Oracle Phase 1 baseline, but not balanced accuracy. Phase 2
remains below the majority baseline in accuracy and predicts only classes 80
and 160, with 775/924 predictions assigned to 160. This is a limited
three-epoch, one-configuration result, not evidence that the task is solved.

## Unchanged retrieval evaluation

The local existing Qdrant service and collections were reused without writes,
re-indexing, schema changes, or port changes. Retrieval remained restricted to
the source paper, used the predicted granularity, `top-k=5`, existing
`text-embedding-3-small` question/chunk embeddings, cosine ranking, unchanged
chunk ordering and concatenation, and GPT-2 joined token-level retrieval F1.

| Metric | Phase 2 | Phase 1 zero-shot |
|---|---:|---:|
| Retrieval coverage | 924/924 (1.0) | 924/924 (1.0) |
| Valid-only mean joined retrieval F1 | 0.22658488852813854 | 0.23910868506493507 |
| Valid-only median joined retrieval F1 | 0.19615549999999998 | 0.2210845 |
| Coverage-adjusted full-set mean | 0.22658488852813866 | 0.23910868506493507 |
| Retrieval wall time (s) | 178.12831589998677 | 367.7590293000012 |

Because all predictions were valid, retrieval coverage is 100% and the
valid-only and coverage-adjusted means differ only by floating-point
aggregation. Classification metrics measure evidence-length-Oracle label
prediction; joined retrieval F1 measures downstream token overlap. They are
different metrics, and the classification improvement did not produce a
retrieval improvement in this run.

## Runtime and integrity

- Training, including three epoch validations and checkpoint writes:
  2,107.3131887838244 seconds (about 35 min 7 s).
- Selected-checkpoint load and isolated final generation:
  2.2354275435209274 + 296.8330853600055 seconds.
- Mean/median final inference: 0.32010594106710705 /
  0.3152373321354389 seconds per question.
- Known training + final validation + retrieval wall time:
  2,584.5100175873376 seconds (about 43 min 5 s).
- Training peak GPU allocated/reserved: 10.660949230194092 /
  11.943359375 GiB; maximum sampled process RSS during training:
  1.9669723510742188 GiB. The RSS value is a recorded-sample maximum, not a
  claim about an unobserved instantaneous peak.

TensorBoard recorded 38 scalar tags, 213 train steps, three validation events,
and 2,424 required scalar values. Its audit found zero loss, value, or count
mismatches and independently selected step 213. Five reloaded-checkpoint
generation probes repeated exactly. The final integrity audit passed every
check, including frozen-order predictions, exact metric recomputation,
retrieval coverage, checkpoint selection, TensorBoard agreement, deterministic
generation, and unchanged Phase 1 source hashes. The focused Phase 2 test suite
reports 19 passed tests; the combined Phase 1 and Phase 2 focused suites report
54 passed tests.

An initial full-run attempt, `qwen-phase2-full-parameter-20260802-seed42-v1`,
was intentionally interrupted after 21 steps and before any validation or
checkpoint. Review found that its partial epoch-tail accumulation would weight
the singleton microbatch incorrectly. The attempt's config and 21 structured
steps remain preserved; it contributes no reported metric. Exact per-example
tail weighting passed a dedicated preflight before the fresh, authoritative
`v2` run.

The selected checkpoint is archived locally at
`outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoints/step-000213`.
Its 11 files total 4,735,895,186 bytes (4.411 GiB) and match the recorded
SHA-256 manifest; `model.safetensors` hashes to
`7d23db1fde0c621623a7d4030073e8858854eba9a4b2d3d7bccda8ca730e2c45`.
Checkpoint binaries and TensorBoard event directories are intentionally
Git-ignored and must be archived separately from the repository.

## Comparability and artifacts

Phase 1 and Phase 2 Qwen results are directly comparable because they use the
same evidence-length Oracle, prompt, parser, preserved split, and evaluation
pipeline. Earlier Logistic Regression and MLP classification results use the
old retrieval-F1 Oracle and are **not directly comparable**; a fair router
comparison requires retraining and evaluating every router on the same new
Oracle labels.

The structured source of truth is
`outputs/qwen_finetuned_router_evidence_length_oracle/final_summary.json`.
Run configuration, histories, checkpoint manifest, best-checkpoint metadata,
TensorBoard inventory, checkpoint verification, environment locks, complete
predictions/raw outputs, classification artifacts, retrieval records, runtime,
hashes, and final integrity audit are under
`outputs/qwen_finetuned_router_evidence_length_oracle/`. The detailed report
and exact reproduction commands are in
`reports/qwen_finetuned_router_evidence_length_oracle/experiment_report.md`.
