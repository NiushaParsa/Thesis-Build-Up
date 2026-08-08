# Qwen Phase 2D token-count prompt sequence-classifier results

Phase 2D is complete. It repeats the Phase 2C Base-model sequence-classifier
experiment with one intended semantic change: the fixed instruction replaces
qualitative context descriptions with the exact candidate token counts. The
model, revision, classifier, data, Oracle, initialization seed, optimization,
checkpoint-selection rule, and retrieval evaluator remain frozen.

On the 924 preserved validation questions, Phase 2D achieves accuracy
0.36904761904761907, macro-F1 0.22994524079282935, weighted F1
0.3644656337102369, and balanced accuracy 0.2391812745015638. These four
classification metrics are higher than Phase 2C in this single-seed paired
run. Top-2 accuracy decreases from 0.6428571428571429 to
0.6341991341991342, and mean joined retrieval F1 decreases from
0.27914719588744585 to 0.2767166677489178. The result is therefore a modest
classification improvement, not a uniform improvement across all outcomes.

## Research question and controlled change

Phase 2D asks whether making the five prompt options numerically explicit
helps the same classifier predict the evidence-length Oracle. The exact fixed
instruction is:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, 4 = 80 tokens, 5 = 160 tokens. Return only the number

Instruction SHA-256:
`b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5`.
The complete model input is:

```text
{instruction}

Question: {original_question_text}
```

The prompt-only protocol audit in the six-way comparison artifact passed. The
only semantic prompt replacement is:

| Phase 2C | Phase 2D |
|---|---|
| `1 = very short context` | `1 = 10 tokens` |
| `2 = short context` | `2 = 20 tokens` |
| `3 = medium context` | `3 = 40 tokens` |
| `4 = long context` | `4 = 80 tokens` |
| `5 = very long context` | `5 = 160 tokens` |

Phase 2C's instruction SHA-256 is
`9e879535647c2bfcd3627d0d65f84c36a1bf442ed95bb5b07029c878ca990de7`.
The changed prompt necessarily changes its tokenization: every recorded
minimum, maximum, and mean input length is nine tokens higher than in Phase
2C. This is a direct consequence of the prompt intervention, not an additional
input feature or configuration change.

| Split | Phase 2C length | Phase 2D length | Phase 2D mean | Over 128 |
|---|---:|---:|---:|---:|
| Train | 86--112 | 95--121 | 101.13363028953229 | 0 |
| Validation | 87--115 | 96--124 | 100.59740259740259 | 0 |

No example is truncated. The tokenizer uses right padding, the model's
end-of-text token as padding (ID 248044), special tokens enabled, and an
explicit nested and top-level padding configuration.

## Input, classifier, and exclusions

`AutoModelForSequenceClassification` produces five directly comparable
logits. The bias-free classifier parameter is a 5 x 1024 `score.weight` head.
The class-head mapping remains:

| Class-head ID | Canonical chunk size |
|---:|---:|
| 0 | 10 |
| 1 | 20 |
| 2 | 40 |
| 3 | 80 |
| 4 | 160 |

Training uses uniform, unweighted, single-label five-class cross-entropy.
Inference uses deterministic argmax over the five logits. Top-2 accuracy is
available from the same directly comparable logits.

The model receives only the fixed instruction and original question text. It
does not receive evidence, evidence length, answers, paper text, embeddings,
retrieved chunks, retrieval scores, metadata, or handcrafted features. There
is no chat-template input, assistant target token, text generation, decoding,
output parser, parser fallback, or default granularity.

## Preserved data and evidence-length Oracle

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

Train Oracle SHA-256:
`64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`.
Validation Oracle SHA-256:
`ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`.

The Oracle counts GPT-2 tokens in the complete stripped,
exact-deduplicated, deterministically ordered ground-truth evidence. It chooses
the closest candidate from 10, 20, 40, 80, and 160, resolves exact midpoint
ties toward the smaller candidate, maps lengths below 10 to 10, and maps
lengths above 160 to 160. It is independent of retrieval F1, embeddings,
cosine similarity, retrieved chunks, and router performance.

The validation Oracle is strongly imbalanced. Class 160 contributes 420/924
= 45.45454545454545%, whereas class 10 contributes only 13/924 =
1.406926406926407%. No QASPER test example was loaded or evaluated.

## Environment and training configuration

| Item | Recorded value |
|---|---|
| Run ID | `qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1` |
| Formulation | `qwen-phase2d-base-sequence-classifier-token-count-prompt-v1` |
| Experiment fingerprint | `dad60bd9a0530865110c2310f62a896c73350fa383c7812d5c6733e376bc377d` |
| Model | `Qwen/Qwen3.5-0.8B-Base` |
| Revision | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Architecture | `AutoModelForSequenceClassification`, five logits |
| Parameters marked trainable/total | 852,991,040 / 852,991,040 |
| Environment | `.venv-qwen` |
| Python | 3.10.7 |
| Python executable | `/workspace/thesis-granularity-router/.venv-qwen/bin/python` |
| Transformers | `5.15.0.dev0` |
| Transformers commit | `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7` |
| PyTorch/CUDA | `2.8.0+cu128` / 12.8 |
| GPU/device | one `NVIDIA A100-SXM4-40GB` / CUDA |
| Dtype/quantization | `torch.bfloat16` / none |
| Objective/class weighting | uniform five-class cross-entropy / uniform |
| Epochs/optimizer updates | 3 / 213, 71 per epoch |
| Per-device/accumulation/effective batch | 4 / 8 / 32 |
| Optimizer | AdamW |
| Learning rate/weight decay | 2e-5 / 0.01 |
| Scheduler/warmup | cosine / 5% = 11 steps |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Evaluation/checkpointing | complete validation after each epoch |
| Selection | macro-F1, then accuracy, weighted F1, balanced accuracy, lower validation CE, earlier step |
| Early stopping | none; fixed three epochs |
| Training-script SHA-256 | `99ba4f9a57b5547e776d81d2c7c94aece2670a9b5ccaf348c8a1fbeb0baa5efc` |
| Resume-contract SHA-256 | `de92814b2a4b25eedc6a1e0c6383ddf8313449869cd80a54fb6a35eca5f534fa` |

The initial randomly added classifier head has float32 SHA-256
`09826669f451891218742ea86926e0b484d1696e57999276889d97b5ccdcbda5`,
exactly matching Phase 2C under seed 42. The selected Phase 2D checkpoint head
has SHA-256
`cbc44bdd71b91be4a7f97c19f87651003eac05b292eb61ea5634dee0cb351025`.
This supports the intended same-initialization paired comparison.

The run configuration records repository value `40f79e1`. This is the
abbreviated base repository HEAD; it resolves to full commit
`40f79e12dbe27ab42934d10732188c1d76087b17`. That base commit did not yet
contain the uncommitted Phase 2D files that were copied to the CUDA host.
Therefore, it must not be described as a commit containing Phase 2D. The
executed training script is anchored by its exact SHA-256 above, and the later
commit that adds Phase 2D should be recorded separately.

The legacy `.venv`, system Python, previous Qwen environments, and completed
Phase 1, 2, 2B, and 2C artifact trees remained separate and unchanged.

## Gradient coverage

The gradient audit passed. The classifier head and language backbone received
gradients:

- parameters with gradients: 752,398,144 across 321 tensors;
- parameters without gradients: 100,592,896 across 153 tensors.

All 852,991,040 parameters were marked trainable, but the no-gradient tensors
belong to the composite checkpoint's vision tower. Because this is a text-only
path with no image input, the vision tower did not participate in the graph
and was not updated. The run is therefore a full-parameter configuration with
observed language-backbone and classifier-head gradients, not a claim that the
vision tower was trained.

## Complete epoch validation history

| Epoch/checkpoint | Validation CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 | Predicted 10/20/40/80/160 | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 / `step-000071` | 1.330027622577948 | 0.44047619047619047 | 0.18072924091134496 | 0.34316665343597114 | 0.2169532849947418 | 0.698051948051948 | 0/0/122/51/751 | 32.7092076446861 |
| 2 / `step-000142` | 1.4269125938931584 | 0.2803030303030303 | 0.1678216855285752 | 0.2325661147417672 | 0.21978505931624878 | 0.5844155844155844 | 0/5/209/637/73 | 31.841020613908768 |
| 3 / `step-000213` | 1.3543887827303502 | 0.36904761904761907 | **0.22994524079282935** | 0.3644656337102369 | 0.2391812745015638 | 0.6341991341991342 | 0/16/219/332/357 | 31.920586789026856 |

`step-000213` is selected because epoch 3 has the highest validation
macro-F1. Epoch 1 has the lowest validation CE and the highest accuracy and
top-2 accuracy, but those are not the predeclared primary selection metric.
The selected-checkpoint reload reproduced the selected-epoch scores exactly
for all 924 examples.

## Final classification result

| Metric | Value |
|---|---:|
| Accuracy | 0.36904761904761907 |
| Macro-F1 | 0.22994524079282935 |
| Weighted F1 | 0.3644656337102369 |
| Balanced accuracy | 0.2391812745015638 |
| Top-2 accuracy | 0.6341991341991342 |
| Top-2 status | available from comparable five-class head logits |
| Valid outputs | 924/924 = 100% |
| Invalid outputs | 0 |
| Invalid-output percentage | 0.0% |
| Majority class | 160 |
| Majority accuracy baseline | 0.45454545454545453 |
| Majority macro-F1 baseline | 0.125 |

Phase 2D correctly predicts 341/924 labels. It is below the class-160
majority baseline on accuracy, although its macro-F1 is above the majority
baseline macro-F1.

| Source | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Oracle count | 13 | 81 | 178 | 232 | 420 |
| Oracle percentage | 1.4069% | 8.7662% | 19.2641% | 25.1082% | 45.4545% |
| Phase 2D prediction count | 0 | 16 | 219 | 332 | 357 |
| Prediction percentage | 0.0000% | 1.7316% | 23.7013% | 35.9307% | 38.6364% |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 |
| 20 | 0.125 | 0.024691358024691357 | 0.041237113402061855 | 81 |
| 40 | 0.2876712328767123 | 0.3539325842696629 | 0.31738035264483627 | 178 |
| 80 | 0.25 | 0.3577586206896552 | 0.29432624113475175 | 232 |
| 160 | 0.5406162464985994 | 0.4595238095238095 | 0.49678249678249675 | 420 |

Class 10 has zero recall and is never predicted. The explicit token-count
mapping does not solve the rarest-class failure. Class 20 also remains weak:
only 2 of 81 true class-20 examples are correct, while 2 of 16 class-20
predictions are correct.

The final confusion matrix uses Oracle rows and predicted columns ordered 10,
20, 40, 80, 160:

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 3 | 7 | 3 |
| 20 | 0 | 2 | 21 | 33 | 25 |
| 40 | 0 | 4 | 63 | 60 | 51 |
| 80 | 0 | 4 | 60 | 83 | 85 |
| 160 | 0 | 6 | 72 | 149 | 193 |

## Clean Phase 2C--2D delta

The saved comparison audit verifies that Phase 2C and 2D use the same Base
checkpoint and revision, initial classifier head, train and validation
records, evidence-length Oracle, label mapping, architecture, objective,
optimizer schedule, seed, selection rule, and retrieval identity. Run IDs,
timestamps, output roots, formulation and prompt hashes, script hashes, and
experiment fingerprints differ as required provenance.

| Metric | Phase 2C | Phase 2D | Phase 2D - Phase 2C |
|---|---:|---:|---:|
| Validation CE | 1.367579244690024 | 1.3543887827303502 | -0.0131904619596738 |
| Accuracy | 0.34523809523809523 | 0.36904761904761907 | +0.023809523809523836 |
| Macro-F1 | 0.21763191244497584 | 0.22994524079282935 | +0.012313328347853508 |
| Weighted F1 | 0.3435657773957275 | 0.3644656337102369 | +0.020899856314509357 |
| Balanced accuracy | 0.22993634120458348 | 0.2391812745015638 | +0.009244933296980312 |
| Top-2 accuracy | 0.6428571428571429 | 0.6341991341991342 | -0.008658008658008698 |
| Mean joined retrieval F1 | 0.27914719588744585 | 0.2767166677489178 | -0.0024305281385280653 |
| Median joined retrieval F1 | 0.2607245 | 0.2558975 | -0.004827000000000026 |

Predicted-distribution change:

| Class | Phase 2C | Phase 2D | Delta |
|---:|---:|---:|---:|
| 10 | 0 | 0 | 0 |
| 20 | 20 | 16 | -4 |
| 40 | 224 | 219 | -5 |
| 80 | 374 | 332 | -42 |
| 160 | 306 | 357 | +51 |

Per-class F1 change:

| Class | Phase 2C F1 | Phase 2D F1 | Delta |
|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 |
| 20 | 0.019801980198019802 | 0.041237113402061855 | +0.021435133204042053 |
| 40 | 0.3233830845771144 | 0.31738035264483627 | -0.006002731932278127 |
| 80 | 0.29042904290429045 | 0.29432624113475175 | +0.0038971982304613073 |
| 160 | 0.45454545454545453 | 0.49678249678249675 | +0.04223704223704222 |

The Phase 2D prompt is associated with a shift from class 80 toward class 160
and higher single-seed accuracy, macro-F1, weighted F1, and balanced accuracy.
It does not recover class 10, and the lower top-2 and retrieval scores show
that the observed benefit is metric-specific.

## Unchanged downstream retrieval

The classifier's class ID is mapped to its canonical chunk size before the
unchanged Qdrant evaluator runs. Retrieval retains source-paper filtering,
predicted granularity, `top_k=5`, existing `text-embedding-3-small`
1,536-dimensional vectors, cosine ranking, unchanged chunk ordering and
concatenation, GPT-2 evidence tokenization, and joined token-level F1 version
`qasper-token-prf-v2`.

| Retrieval metric | Value |
|---|---:|
| Status | complete |
| Coverage | 924/924 = 100% |
| Invalid predictions without retrieval | 0 |
| Valid-only mean joined retrieval F1 | 0.2767166677489178 |
| Valid-only median joined retrieval F1 | 0.2558975 |
| Coverage-adjusted full-set mean | 0.27671666774891795 |
| Top-k | 5 |
| Paper restricted | true |
| Evaluation configuration SHA-256 | `9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8` |
| Retrieval wall time | 151.0063940999098 seconds |
| Durable per-question processing sum | 143.97760520025622 seconds |
| Runtime segments | 1 |
| Current/completed/uninterrupted invocation wall | 151.0063940999098 seconds |
| Reported runtime basis | `complete_uninterrupted_invocation` |

Every prediction is an explicit five-logit argmax. No invalid output is mapped
to a default class. Classification metrics measure prediction of the
evidence-length Oracle label; joined retrieval F1 measures evidence-token
overlap after downstream retrieval. These are different outcomes and must not
be treated as interchangeable.

## Runtime and resources

| Measurement | Value |
|---|---:|
| Training wall time | 1224.5802961867303 seconds |
| Initial/final recorded optimizer-window loss | 1.5487420707941055 / 1.5045942068099976 |
| Training peak allocated/reserved GPU | 9.0316162109375 / 9.6015625 GiB |
| Training RSS | 1.9677543640136719 GiB |
| Selected-checkpoint load | 2.7541816290467978 seconds |
| Isolated final inference | 34.72815803065896 seconds |
| Mean inference/question | 0.0374373855065248 seconds |
| Median inference/question | 0.03656412195414305 seconds |
| Synchronized batch-forward sum | 34.59214420802891 seconds |
| Final peak allocated/reserved GPU | 1.7161517143249512 / 1.77734375 GiB |
| Final RSS | 1.7001190185546875 GiB |
| Training + selected-checkpoint load/inference | 1262.062635846436 seconds |
| Retrieval wall time | 151.0063940999098 seconds |
| Known training + validation + retrieval | 1413.0690299463458 seconds |

Training took approximately 20 minutes 24.58 seconds; the known combined
training, selected-checkpoint validation, and retrieval time was approximately
23 minutes 33.07 seconds. The initial and final recorded losses are individual
optimizer-window losses, not epoch means. The final update contains a partial
five-example accumulation window, so those two values alone do not establish a
training-loss trend.

## Interpretation and limitations

Within this controlled seed-42 run, replacing qualitative descriptions with
the exact token counts improves the selected-checkpoint classification
accuracy, macro-F1, weighted F1, and balanced accuracy relative to Phase 2C.
Phase 2D has the highest saved Qwen macro-F1, weighted F1, and balanced
accuracy in the six-way evidence-length-Oracle comparison. Phase 2 numeric SFT
still has the highest accuracy, and Phase 2B-A still has the highest mean
joined retrieval F1.

The result remains below the majority baseline on accuracy and completely
misses class 10. It also uses only one seed. The same validation split is used
for checkpoint selection and final reporting, and no untouched QASPER test
result is available. Consequently, the observed Phase 2C--2D difference is a
valid single-seed prompt ablation but is not an estimate of run-to-run
variance, statistical significance, or held-out generalization. It should not
be generalized into a claim that explicit token counts always improve the
classifier.

Earlier Logistic Regression and MLP classification experiments used the old
retrieval-F1 Oracle. Their classification results are not directly comparable
with Phase 2D's evidence-length-Oracle labels.

## Artifacts and integrity

Authoritative artifacts are under
`outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/`:

- `configuration/experiment.json`
- `configuration/preflight_manifest.json`
- `final_summary.json`
- `integrity/selected_checkpoint_transfer_verification.json`
- `integrity/final_integrity_audit.json`
- `classification/metrics.json`
- `classification/confusion_matrix.csv`
- `classification/predicted_vs_oracle.svg`
- `validation/predictions.jsonl`
- `validation/raw_outputs.jsonl`
- `validation/parsed_predictions.jsonl`
- `validation/invalid_outputs.jsonl`
- `validation/runtime_summary.json`
- `retrieval/results.jsonl`
- `retrieval/runtime_segments.jsonl`
- `retrieval/summary.json`

The complete run records are under
`runs/qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1/`, including `training_config.json`,
`dataset_manifest.json`, `formatted_example_inspection.json`,
`gradient_coverage_audit.json`, `training_history.jsonl`,
`validation_history.jsonl`, all three epoch prediction files,
`checkpoint_manifest.json`, `best_checkpoint.json`, `summary.json`, and the
retained `checkpoints/step-000213/`.

The checkpoint transfer verification passed. The remote/local archive
SHA-256 is
`2dd4d23ff77179e1b33e522829cb2fdd6dd12684500a2158cc95f5f79a242a56`,
and all nine extracted checkpoint files match their remote hashes. The
independent final audit passed 73/73 artifact assertions, and the focused
Phase 2--2D regression suite passed 102/102 tests. Its machine-readable record
is `integrity/final_integrity_audit.json`. The comparison artifact is
`outputs/qwen_phase2d_comparison_evidence_length_oracle/six_way_comparison.json`.
The detailed report is
`reports/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/experiment_report.md`.

Absolute `/dev/shm/...` paths inside the saved run metadata preserve the
original ephemeral CUDA-host provenance. The corresponding retained local
files live under the repository-relative output tree above.

## Exact reproduction commands

Original CUDA-host execution form, from
`/workspace/thesis-granularity-router`:

```bash
cd /workspace/thesis-granularity-router
export HF_HOME=/dev/shm/qwen_phase2d_hf
export MPLCONFIGDIR=/dev/shm/phase2d_mpl
export PYTHONPATH=/dev/shm/phase2d_code
export PHASE2D_REPOSITORY_COMMIT=40f79e1

PY=.venv-qwen/bin/python
SCRIPT=/dev/shm/phase2d_code/qwen_phase2d_sequence_classifier.py
ROOT=/dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle
RUN=qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1

$PY $SCRIPT --output-root $ROOT inspect
$PY $SCRIPT --output-root $ROOT train --mode full --run-id $RUN
$PY $SCRIPT --output-root $ROOT final-validation --run-id $RUN
```

Before execution, the deployed `$SCRIPT` must hash to
`99ba4f9a57b5547e776d81d2c7c94aece2670a9b5ccaf348c8a1fbeb0baa5efc`.
The `40f79e1` value reproduces the recorded base-HEAD field; the script hash,
not that uncommitted base commit, identifies the executed Phase 2D code.

After transferring the output tree and selected checkpoint back to the local
project, with the unchanged Qdrant service available at `127.0.0.1:6334`:

```powershell
$root = "outputs\qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle"
$run = "qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1"

.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py evaluate-retrieval --output-root $root --run-id $run
.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py compare --phase2d-summary "$root\final_summary.json" --output outputs\qwen_phase2d_comparison_evidence_length_oracle\six_way_comparison.json
```

These commands must use the preserved Oracle files and existing Qdrant
collections. They must not recreate, re-index, or mutate a collection.
