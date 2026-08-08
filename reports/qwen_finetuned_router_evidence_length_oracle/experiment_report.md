# Fine-tuned Qwen3.5-0.8B router with evidence-length Oracle

## Status and objective

Phase 2 is complete. It performs the supervisor-confirmed full-parameter
supervised fine-tuning of `Qwen/Qwen3.5-0.8B` for five-class adaptive
chunk-granularity routing. The authoritative run ID is
`qwen-phase2-full-parameter-20260802-seed42-v2`.

The experiment uses exactly the 2,245 preserved official-QASPER training
questions and evaluates the selected checkpoint on all 924 preserved
official-QASPER validation questions. It remains isolated under
`outputs/qwen_finetuned_router_evidence_length_oracle/`; no frozen Phase 1
artifact was overwritten or modified. The final integrity audit re-hashed the
Phase 1 summary, fixed prompt, and Oracle JSONL files and confirmed all four
were unchanged.

Repository commit recorded at training launch:
`55af1bcbc4d7a089adaafd4da539581b2dbbed67`. The exact launch training script
is archived as `environment/qwen_phase2_training_launch.py` with SHA-256
`bb1a1a591ef60b933cd394d12b7087da1345dc03ded48d8c7739348b35392fd3`.
The post-training script snapshot hashes to
`35361c5314a288761f70d9152e391e77a66931766bb900545333f5a0ab608e54`.

## Experimental boundary

This is causal-language-model supervised fine-tuning, not an embedding
classifier and not parameter-efficient fine-tuning.

- Total parameters: 852,985,920.
- Trainable parameters: 852,985,920 (100%).
- Optimizer steps/parameter-update steps: 213.
- No frozen backbone, separate classification head, LoRA, QLoRA, adapter,
  prompt tuning, soft prompt, partial-layer tuning, or quantization.
- No class over/undersampling, synthetic data, random resplit, prompt search,
  hyperparameter search, or validation example in gradient updates.
- One predeclared configuration, three fixed epochs, and no early stopping.

The optimizer was AdamW over all model parameters. BF16 did not use a gradient
scaler. Checkpoints contain the model, optimizer, scheduler, random states,
global step/epoch/training state, configuration, and validation metadata.

## Frozen data and Oracle

The source Oracle files are:

- `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/train_oracle.jsonl`
- `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/validation_oracle.jsonl`

The previous project Oracle chose the granularity with the highest joined
top-five retrieval F1. Phase 2 instead uses the supervisor-confirmed
evidence-length Oracle frozen in Phase 1. For each question it strips evidence
spans, drops empty spans, deduplicates exact stripped evidence across
annotators, lexicographically sorts the unique spans, joins them with newline
separators, and counts GPT-2 tokens without special tokens. It then chooses the
numerically nearest of 10, 20, 40, 80, and 160. Exact midpoint ties choose the
smaller candidate; evidence shorter than 10 maps to 10 and evidence longer
than 160 maps to 160.

This label is independent of retrieval F1, embedding quality, cosine
similarity, retrieved chunks, and router performance.

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

Train Oracle SHA-256:
`64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`.
Validation Oracle SHA-256:
`ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`.
There is no train/validation question or document overlap. Validation is
strongly imbalanced: class 160 accounts for 420/924 (45.45%) and class 10 for
13/924 (1.41%).

## Input, target, chat template, and leakage boundary

The only semantic input is the frozen Phase 1 instruction:

> You are a router for a retrieval-augmented generation system. Based only on
> the question, select the chunk size most suitable for retrieving the evidence
> required to answer it. Choose exactly one value from: 10, 20, 40, 80, 160.
> Return only the number.

It is followed by `Question: {original_question_text}` and formatted with the
official Qwen chat template as the user turn. The assistant target is exactly
one of `10`, `20`, `40`, `80`, or `160`. Prompt/instruction/question and
padding tokens are masked with `-100`; loss is computed only on the assistant
class and required assistant-ending tokens. All five labels tokenize
unambiguously into 4 or 5 supervised tokens. Across the preserved training
set, formatted sequences range from 89 to 115 tokens (mean
95.40979955456571), below the declared maximum of 128.

The model did not receive evidence, evidence length, Oracle construction
details, answers, paper title/abstract/content, retrieved chunks, retrieval
scores, cosine similarities, OpenAI or other embeddings, document metadata,
question-length or handcrafted features, demonstrations, or explanatory and
chain-of-thought targets. Evidence length and Oracle label appear in saved
evaluation records only as post-inference evaluation metadata; they were not
model inputs.

The focused Phase 2 suite reports 19 passed tests, and the combined Phase 1
and Phase 2 focused suites report 54 passed tests. The Phase 2 suite covers preserved counts
and split separation, all label targets, target/prompt/padding masking,
collation, parser compatibility and invalid handling, determinism,
optimizer-state relocation, epoch-tail accumulation, checkpoint selection,
resume-log truncation, final prediction materialization, and retrieval-summary
validation. Tiny-overfit, smoke, full-loop, and corrected tail-weighting
preflights additionally exercised checkpoint save/reload, resume, TensorBoard,
and CUDA execution.

## Environment and hardware

- Environment: `.venv-qwen`; the legacy `.venv` remained unchanged.
- Python: `3.10.7 (main, Oct 3 2022, 02:19:58) [Clang 14.0.3]`.
- Executable: `/workspace/thesis-granularity-router/.venv-qwen/bin/python`.
- Transformers: `5.15.0.dev0`, source commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`.
- PyTorch: `2.8.0+cu128`; toolkit/runtime CUDA 12.8.
- TensorBoard: `2.20.0`.
- Device/dtype: CUDA on one NVIDIA A100-SXM4-40GB, `torch.bfloat16`.
- NVIDIA driver: 570.133.20; GPU compute capability 8.0.
- Host: AMD EPYC 7713, approximately 128.5 GB RAM, 60 GB instance disk.
- Quantization: none.

The pinned base model is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. The resolved package inventories
before and after training are preserved in
`environment/phase2_package_lock.txt` and
`environment/phase2_package_lock_after.txt`. Only the CUDA PyTorch build and
TensorBoard support needed by Phase 2 were added to the dedicated Qwen
environment; the original Python 3.9 `.venv` remains the reproducibility
environment for the earlier Logistic Regression, MLP, old-Oracle, and
retrieval experiments.

## Preflight evidence

The full-parameter path was feasible on the A100. A balanced five-example
tiny-overfit run reached training loss `1.7583345197635936e-06` at step 100,
with 7.453740119934082 GiB peak allocated GPU memory. A ten-example smoke run
(two examples per class) completed four optimizer steps; training loss changed
from 0.529105007648468 to 0.32945019006729126, elapsed time was
14.748280551284552 seconds, and peak allocated/reserved GPU memory was
8.12428092956543/9.544921875 GiB. Its checkpoint reloaded and deterministic
parser-compatible generation passed. Its four TensorBoard loss events exactly
matched its four structured steps.

These are technical preflights, not reported validation results and not
hyperparameter-selection trials.

## Full training configuration

| Item | Value |
|---|---|
| Training method | Full-parameter causal-LM SFT |
| Epochs | 3 fixed |
| Optimizer steps | 71/epoch; 213 total |
| Per-device batch | 4 |
| Gradient accumulation | 8 |
| Effective batch | 32 |
| Epoch tail | 5 examples, weighted exactly per example |
| Maximum sequence length | 128 |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Scheduler | Cosine |
| Warmup | 5% (11 optimizer steps) |
| Gradient clipping | `1.0` |
| Seed | 42 |
| Determinism | strict deterministic algorithms; `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Logging | every optimizer step |
| Validation/checkpoint | end of every epoch |
| Retention during training | all three epoch checkpoints |
| Early stopping | none |
| Primary selection metric | validation macro-F1 |
| Tie-break | accuracy, weighted F1, balanced accuracy, lower validation loss, earlier step |

The first/last structured training losses are
0.4739701375365257/0.4026994347572327. Mean optimizer-step duration was
5.17159837138065 seconds; mean throughput was 6.1007536701855 examples/s and
582.059189458218 tokens/s. The final step is the correctly weighted
five-example epoch tail and is therefore slower per example than a full
32-example step.

## Validation history and checkpoint selection

The 924 validation questions were used for evaluation and the predeclared
checkpoint rule only, never for gradients.

| Epoch | Checkpoint | Validation loss | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Predicted 10/20/40/80/160 | Invalid | Wall (s) |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | `step-000071` | 0.304930741717289 | 0.45454545454545453 | 0.125 | 0.2840909090909091 | 0.2 | 0/0/0/0/924 | 0 | 340.64150045998394 |
| 2 | `step-000142` | 0.30652774386591725 | 0.4556277056277056 | 0.12698176971774003 | 0.28686332944508014 | 0.20086206896551725 | 0/0/0/3/921 | 0 | 327.1569862253964 |
| 3 | `step-000213` | 0.3081164950932259 | 0.4318181818181818 | **0.16502267760462996** | 0.32805741427623947 | 0.20697865353037764 | 0/0/0/149/775 | 0 | 321.21750357560813 |

`step-000213` is selected because 0.16502267760462996 is the highest
validation macro-F1. Selection did not use downstream retrieval F1 or
qualitative inspection. The selected checkpoint's higher validation loss and
lower accuracy do not displace the predeclared primary metric. All three
events and checkpoint metadata are preserved in `validation_history.jsonl`,
`checkpoint_manifest.json`, and `best_checkpoint.json`.

## TensorBoard monitoring and audit

The authoritative run's TensorBoard directory is:

`outputs/qwen_finetuned_router_evidence_length_oracle/tensorboard/qwen-phase2-full-parameter-20260802-seed42-v2`

It contains 38 scalar tags covering loss, learning rate, epoch, global step,
gradient norm, duration/throughput, CPU/GPU memory, validation loss and four
aggregate metrics, invalid outputs, every class's precision/recall/F1, and
predicted class counts. There are 213 events for each training/system tag and
three for each validation tag, totaling 2,424 required scalar values.

The structured-versus-TensorBoard audit reports:

- 213 structured and TensorBoard training-loss steps;
- three structured validation events;
- zero loss mismatches;
- zero required scalar-value mismatches;
- zero required scalar-count mismatches;
- TensorBoard-derived and structured selected checkpoint both at step 213.

Final metrics come from JSON/JSONL/CSV artifacts, not visual estimates or
smoothed TensorBoard curves. TensorBoard did not alter training or checkpoint
selection.

Inspect the saved run without starting a server automatically:

```bash
.venv-qwen/bin/tensorboard --logdir outputs/qwen_finetuned_router_evidence_length_oracle/tensorboard/qwen-phase2-full-parameter-20260802-seed42-v2
```

## Selected-checkpoint deterministic validation

The selected checkpoint is:

`outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoints/step-000213`

After training it was loaded afresh and generated outputs for all 924 preserved
validation questions with `model.eval()`, `torch.inference_mode()`, and
`do_sample=False`. The post-training outputs matched all 924 epoch-3 outputs
exactly. All 924 parser results are valid; no invalid or ambiguous output was
mapped to a default class. Five additional checkpoint-reload probes repeated
exactly.

The selected checkpoint's `model.safetensors` SHA-256 is
`7d23db1fde0c621623a7d4030073e8858854eba9a4b2d3d7bccda8ca730e2c45`.
Optimizer, scheduler, and random-state files are present. The complete local
archive contains 11 files totaling 4,735,895,186 bytes (4.411 GiB), all matching
`selected_checkpoint_sha256.txt`. Checkpoint and TensorBoard trees are
intentionally ignored by Git; they must be retained in a separate experiment
archive. Versionable file inventories and hashes remain in the run directory.

## Final classification results

| Metric | Fine-tuned Phase 2 | Zero-shot Phase 1 | Majority baseline |
|---|---:|---:|---:|
| Accuracy | 0.4318181818181818 | 0.04004329004329004 | 0.45454545454545453 |
| Macro-F1 | 0.16502267760462996 | 0.049045932422555796 | 0.125 |
| Weighted F1 | 0.32805741427623947 | 0.032612933907418644 | 0.2840909090909091 |
| Balanced accuracy | 0.20697865353037764 | 0.23369399361908724 | 0.2 |
| Top-2 accuracy | unavailable | unavailable | unavailable |
| Valid outputs | 924/924 | 924/924 | 924/924 |
| Invalid outputs | 0 (0.0%) | 0 (0.0%) | 0 |

Top-2 accuracy is unavailable because deterministic generated text provides no
directly comparable five-class scores; none was invented or approximated.

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 |
| 20 | 0.0 | 0.0 | 0.0 | 81 |
| 40 | 0.0 | 0.0 | 0.0 | 178 |
| 80 | 0.2953020134228188 | 0.1896551724137931 | 0.23097112860892388 | 232 |
| 160 | 0.45806451612903226 | 0.8452380952380952 | 0.5941422594142259 | 420 |

### Distributions

| Source | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Validation Oracle | 13 | 81 | 178 | 232 | 420 |
| Phase 1 zero-shot predictions | 767 | 40 | 116 | 0 | 1 |
| Phase 2 fine-tuned predictions | 0 | 0 | 0 | 149 | 775 |

The Phase 2 model selects class 160 for 775/924 questions (83.87%) and never
selects 10, 20, or 40. Fine-tuning moved the zero-shot collapse away from class
10, but produced a new two-class concentration rather than balanced five-class
routing.

### Confusion matrix

Rows are Oracle labels and columns are predictions, ordered 10, 20, 40, 80,
160.

| Oracle ↓ / predicted → | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 0 | 1 | 12 |
| 20 | 0 | 0 | 0 | 16 | 65 |
| 40 | 0 | 0 | 0 | 23 | 155 |
| 80 | 0 | 0 | 0 | 44 | 188 |
| 160 | 0 | 0 | 0 | 65 | 355 |

## Phase 1 and majority-baseline interpretation

Phase 1 and Phase 2 use the same preserved split, evidence-length Oracle,
prompt semantics, chat template, parser, and metrics, so this comparison is
methodologically aligned. Phase 2 raises accuracy from
0.04004329004329004 to 0.4318181818181818, macro-F1 from
0.049045932422555796 to 0.16502267760462996, and weighted F1 from
0.032612933907418644 to 0.32805741427623947. Balanced accuracy falls from
0.23369399361908724 to 0.20697865353037764 because Phase 2 no longer recovers
the three smallest classes.

The evidence-length majority baseline always predicts 160. Its accuracy
0.45454545454545453 exceeds Phase 2 accuracy, while its macro-F1 0.125 is below
Phase 2 macro-F1. Thus Phase 2 is not a general five-class solution: it gains
some class-80 discrimination, but its accuracy does not beat the trivial
majority predictor.

Earlier Logistic Regression and MLP classification results used the old
retrieval-F1 Oracle. They are **not directly comparable** with either
evidence-length-Oracle Qwen result. A fair multi-router comparison requires
training and evaluating every router on the same new labels and preserved
splits.

## Unchanged end-to-end retrieval

Retrieval was run locally after copying the selected predictions back from the
GPU trainer. The existing Qdrant service at `127.0.0.1:6334` and its existing
collections were reused read-only; no remote Qdrant instance or collection was
created. The evaluation preserved source-paper filtering, predicted
granularity filtering, `top-k=5`, existing `text-embedding-3-small`
question/chunk embeddings, cosine similarity, ranking and chunk order,
newline concatenation, evidence cleaning/deduplication, and GPT-2 joined
token-level F1.

| Retrieval metric | Phase 2 | Phase 1 |
|---|---:|---:|
| Coverage | 924/924 = 1.0 | 924/924 = 1.0 |
| Valid-only mean joined F1 | 0.22658488852813854 | 0.23910868506493507 |
| Valid-only median joined F1 | 0.19615549999999998 | 0.2210845 |
| Coverage-adjusted full-set mean | 0.22658488852813866 | 0.23910868506493507 |
| Retrieval wall time (s) | 178.12831589998677 | 367.7590293000012 |

Invalid predictions receive no retrieval and no default granularity. There are
zero invalid predictions here, so all 924 have retrieval records and
valid-only versus coverage-adjusted means differ only by floating-point
aggregation. Had invalids existed, valid-only F1 would summarize retrieved
valid predictions while the transparent full-set mean assigned invalids zero
contribution.

Classification accuracy, macro-F1, and weighted F1 measure prediction of the
evidence-length Oracle class. Joined retrieval F1 measures downstream token
overlap after retrieval. They are distinct. In this run, substantially better
classification than Phase 1 did not improve retrieval: mean joined F1 fell
from 0.23910868506493507 to 0.22658488852813854.

## Runtime and resources

| Stage/resource | Recorded value |
|---|---:|
| Full training wall, including three validations/checkpoints | 2,107.3131887838244 s |
| Selected-checkpoint model load | 2.2354275435209274 s |
| Isolated final generation | 296.8330853600055 s |
| Mean final inference/question | 0.32010594106710705 s |
| Median final inference/question | 0.3152373321354389 s |
| Selected epoch validation wall | 321.21750357560813 s |
| Retrieval wall | 178.12831589998677 s |
| Known training + final validation + retrieval | 2,584.5100175873376 s |
| Training peak GPU allocated | 10.660949230194092 GiB |
| Training peak GPU reserved | 11.943359375 GiB |
| Maximum sampled process RSS during training | 1.9669723510742188 GiB |
| Reloaded final-validation peak GPU allocated/reserved | 1.6835732460021973 / 1.765625 GiB |
| Reloaded final-validation RSS | 1.6948738098144531 GiB |

The full known pipeline took about 43 min 5 s. Training wall includes epoch
validation and checkpoint writing, while the separate 296.833-second final
generation is the required fresh selected-checkpoint inference.

## Aborted v1 attempt

The first full-run ID, `qwen-phase2-full-parameter-20260802-seed42-v1`, is
preserved rather than hidden. It was manually interrupted after 21 training
steps, before the first validation or checkpoint. Review identified a weighting
problem for the partial accumulation group at each epoch: with 2,245 examples,
batch 4, and accumulation 8, the final group contains one four-example
microbatch and one singleton, which must be averaged by example rather than by
microbatch. The v1 run used training-script SHA-256
`054cd3e2ea256ab929052c06919996d9d0c544d8c6f4e734936591990c5fa1fe`.

The implementation was corrected to weight every example equally, including
the five-example tail. A dedicated tail-weighting preflight succeeded, and the
authoritative v2 run started from the original pinned model rather than from
v1. V1 has no validation result, is excluded from every reported metric, and
remains available through its config, 21-step structured history, TensorBoard
run, and `logs/full-training.log`.

## Integrity audit

`integrity_audit.json` has status `passed`. It independently confirms:

- 924 unique predictions in frozen validation order;
- exact classification recomputation and artifact reconciliation;
- zero invalid outputs and no default mapping;
- exactly 924 retrieval records covering the 924 valid predictions;
- exact retrieval-summary recomputation;
- 213 complete training steps and three validation events;
- reproducible checkpoint selection by the declared rule;
- zero TensorBoard/structured-log mismatches;
- deterministic selected-checkpoint generation;
- exact SHA-256 agreement for all 11 locally archived selected-checkpoint files;
- unchanged Phase 1 source hashes.

The process-RSS figure is the maximum recorded sample, not a claim about an
unobserved instantaneous peak. The canonical final summary SHA-256 recorded by
the audit is
`73f9ffb773aedcc47ba7ebe3850d28e372038ae795e3f6cb69f888bfcfb87d04`.
Prediction, classification, confusion-matrix, retrieval, configuration,
manifest, TensorBoard-audit, and checkpoint-verification hashes are retained in
the same audit file.

## Artifacts

Main results:

- `outputs/qwen_finetuned_router_evidence_length_oracle/final_summary.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/integrity_audit.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/dataset_manifest.json`

Authoritative run metadata:

- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/training_config.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/training_history.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/validation_history.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoint_manifest.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/best_checkpoint.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/formatted_example_inspection.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/validation/predictions_step-000071.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/validation/predictions_step-000142.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/validation/predictions_step-000213.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/tensorboard_scalar_inventory.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoint_verification.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoint_archive_verification.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/selected_checkpoint_files.txt`
- `outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/selected_checkpoint_sha256.txt`

Final selected-checkpoint validation:

- `outputs/qwen_finetuned_router_evidence_length_oracle/validation/predictions.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/validation/raw_outputs.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/validation/parsed_predictions.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/validation/invalid_outputs.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/validation/runtime_summary.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/classification/metrics.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/classification/confusion_matrix.csv`
- `outputs/qwen_finetuned_router_evidence_length_oracle/classification/predicted_vs_oracle.svg`

Retrieval:

- `outputs/qwen_finetuned_router_evidence_length_oracle/retrieval/results.jsonl`
- `outputs/qwen_finetuned_router_evidence_length_oracle/retrieval/summary.json`
- `outputs/qwen_finetuned_router_evidence_length_oracle/retrieval/runtime_segments.jsonl`

Environment, logs, and monitoring:

- `outputs/qwen_finetuned_router_evidence_length_oracle/environment/`
- `outputs/qwen_finetuned_router_evidence_length_oracle/logs/`
- `outputs/qwen_finetuned_router_evidence_length_oracle/tensorboard/qwen-phase2-full-parameter-20260802-seed42-v2/`
- `requirements-qwen-phase2.txt`
- `qwen_phase2.py`
- `tests/test_qwen_phase2.py`

Frozen Phase 1 source and aligned baseline:

- `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/`
- `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/configuration/fixed_prompt.json`
- `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/final_summary.json`
- `docs/QWEN_PHASE1_RESULTS.md`
- `reports/qwen_pretrained_zero_shot_router_evidence_length_oracle/experiment_report.md`

## Reproduction commands

Run training on a clean CUDA Linux host. These commands create only the
dedicated Qwen environment and must never target the legacy `.venv`:

```bash
uv venv --python 3.10.7 .venv-qwen
uv pip install --python .venv-qwen/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-qwen/bin/python -r requirements-qwen-phase2.txt

.venv-qwen/bin/python --version
.venv-qwen/bin/python -c "import torch, transformers, tensorboard; print(torch.__version__, torch.version.cuda, transformers.__version__, tensorboard.__version__)"
uv pip freeze --python .venv-qwen/bin/python
```

Verify data, formatting/masking tests, and preflights:

```bash
.venv-qwen/bin/python qwen_phase2.py inspect-data
.venv-qwen/bin/python -m pytest tests/test_qwen_phase2.py -q

.venv-qwen/bin/python qwen_phase2.py tiny-overfit --run-id qwen-phase2-tiny-overfit-seed42-v1 --max-steps 20 --per-class 1
.venv-qwen/bin/python qwen_phase2.py tiny-overfit --run-id qwen-phase2-tiny-overfit-seed42-v1 --max-steps 100 --per-class 1 --resume outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-tiny-overfit-seed42-v1/checkpoints/step-000020
.venv-qwen/bin/python qwen_phase2.py smoke --run-id qwen-phase2-smoke-full-parameter-seed42-v1 --max-steps 4 --per-class 2
```

Run the exact full configuration, or resume it from a saved epoch checkpoint:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

.venv-qwen/bin/python qwen_phase2.py train --run-id qwen-phase2-full-parameter-20260802-seed42-v2

# Example deterministic resume after epoch 2
.venv-qwen/bin/python qwen_phase2.py train --run-id qwen-phase2-full-parameter-20260802-seed42-v2 --resume outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoints/step-000142
```

Audit monitoring, reload the selected checkpoint, and run final validation.
`final-validation` also materializes the classification metrics, confusion
matrix, distributions, and histogram; there is no separate classification
scoring command.

```bash
.venv-qwen/bin/python qwen_phase2.py audit-tensorboard --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.venv-qwen/bin/python qwen_phase2.py verify-checkpoint --run-id qwen-phase2-full-parameter-20260802-seed42-v2 --checkpoint outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoints/step-000213
.venv-qwen/bin/python qwen_phase2.py final-validation --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.venv-qwen/bin/tensorboard --logdir outputs/qwen_finetuned_router_evidence_length_oracle/tensorboard/qwen-phase2-full-parameter-20260802-seed42-v2
```

With the preserved local `.env` pointing to the already verified Qdrant
service at `127.0.0.1:6334`, reproduce retrieval and the final cross-artifact
audit on Windows:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2.py evaluate-retrieval --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.\.venv-qwen\Scripts\python.exe qwen_phase2.py audit-final --run-id qwen-phase2-full-parameter-20260802-seed42-v2
```

The two Markdown reports are maintained from `final_summary.json`, the run
histories/manifests, retrieval summary, TensorBoard audit, checkpoint
verification, and `integrity_audit.json`. No separate report-generator command
exists, so none is fabricated here.

## Limitations and next steps

- Only one predeclared training configuration and one seed were evaluated.
- Checkpoint selection uses the same 924 validation examples reported here;
  no test split was loaded or evaluated.
- The evidence-length Oracle is strongly imbalanced, and classes 10 and 20
  have only 55 and 267 training examples.
- The selected model predicts only 80 and 160, so its improved macro-F1 remains
  low in absolute terms and its accuracy remains below the majority baseline.
- The downstream mean/median joined retrieval F1 are below Phase 1, showing
  that better Oracle-label classification does not guarantee better retrieval.
- Binary checkpoint and TensorBoard event trees are deliberately outside Git;
  reproducibility requires preserving the separate archive and hash manifests.

Any next training experiment should be a separately named Phase 2 extension,
retain this frozen baseline, predeclare its treatment of imbalance and
checkpoint selection, and evaluate on the same preserved split and unchanged
retrieval pipeline. It must not reinterpret old-Oracle LR/MLP numbers as
directly comparable classification baselines.
