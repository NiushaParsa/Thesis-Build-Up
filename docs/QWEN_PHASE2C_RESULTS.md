# Qwen Phase 2C Base sequence-classifier results

Phase 2C completed a supervisor-motivated direct five-class classifier using
the exact `Qwen/Qwen3.5-0.8B-Base` checkpoint, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. It is an isolated experiment and
does not overwrite the frozen Phase 1, Phase 2, or Phase 2B baselines.

Its validation macro-F1 is 0.21763191244497584, the best saved Qwen macro-F1
so far under the evidence-length Oracle. Numeric-target Phase 2 still has the
best accuracy, 0.4318181818181818. Phase 2B-A still has the best downstream
mean joined retrieval F1, 0.28646775432900434, compared with Phase 2C at
0.27914719588744585.

## Motivation and exact formulation

The supervisor-motivated change is to evaluate a Base checkpoint as a
conventional five-logit sequence classifier and to use semantic context-length
options in a revised instruction. This removes generated text, assistant
target tokens, and parsing from the classification interface.

The exact fixed instruction is:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = very short context, 2 = short context, 3 = medium context, 4 = long context, 5 = very long context. Return only the number

Instruction SHA-256:
`9e879535647c2bfcd3627d0d65f84c36a1bf442ed95bb5b07029c878ca990de7`.
The plain input template is
`{instruction}\n\nQuestion: {original_question_text}`.

`AutoModelForSequenceClassification` produces five directly comparable logits.
Class IDs 0/1/2/3/4 map to canonical chunk sizes 10/20/40/80/160. The
classifier parameter is a bias-free 5×1024 `score.weight` head. Training uses
uniform, unweighted, single-label five-class cross-entropy; inference uses
deterministic argmax. Top-2 accuracy is available from the same five logits.

There is no generation, chat template, assistant-token target, decoding,
output parser, parser fallback, or default class. The model receives only the
fixed instruction and original question text. It receives no evidence,
evidence length, answer, paper text, retrieved chunk, retrieval score,
question embedding, metadata, or handcrafted feature.

Train input lengths are 86--112 tokens, mean 92.13363028953229. Validation
lengths are 87--115, mean 91.59740259740259. No input exceeds the maximum
sequence length of 128, and no example is silently truncated.

## Preserved data and evidence-length Oracle

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

Train Oracle SHA-256:
`64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`.
Validation Oracle SHA-256:
`ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`.

The Oracle counts GPT-2 tokens in complete stripped, exact-deduplicated,
deterministically ordered ground-truth evidence; chooses the closest candidate
from 10, 20, 40, 80, and 160; uses the smaller candidate for exact midpoint
ties; and clips outside the 10--160 range. It is independent of retrieval F1,
embeddings, cosine similarity, retrieved chunks, and router performance.

Class 160 is 420/924 = 45.45% of validation, while class 10 has only 13
examples. No QASPER test example was loaded or evaluated.

## Environment and training configuration

| Item | Recorded value |
|---|---|
| Run ID | `qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1` |
| Formulation | `qwen-phase2c-base-sequence-classifier-v1` |
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
| GPU/device | one `NVIDIA A100-SXM4-40GB` / `cuda` |
| Dtype/quantization | `torch.bfloat16` / none |
| Objective | uniform unweighted five-class cross-entropy |
| Epochs/optimizer updates | 3 / 213 |
| Per-device/accumulation/effective batch | 4 / 8 / 32 |
| Optimizer | AdamW |
| Learning rate/weight decay | 2e-5 / 0.01 |
| Scheduler/warmup | cosine / 5% = 11 steps |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Evaluation/checkpointing | full validation after each epoch |
| Checkpoint selection | macro-F1, then accuracy, weighted F1, balanced accuracy, lower validation CE, earlier step |
| Experiment fingerprint | `6508ded1f9c25b451207f891f90fa5c9a7a4c09da7ea7125555bfa0ec7faca90` |
| Training-script SHA-256 | `6eeb155296b239463d5ba7c8c75dfed8dd59f8c5285cab1edf7cb6a553f9aefb` |
| Repository commit at launch | `55af1bcbc4d7a089adaafd4da539581b2dbbed67` |

The protected legacy `.venv`, system Python, and earlier Qwen environments and
artifacts remained separate and unchanged.

## Gradient coverage

All 852,991,040 parameters were marked trainable and placed in the
full-parameter optimization configuration. The saved gradient audit passed and
found gradients in the language backbone and classifier head:

- parameters with gradients: 752,398,144 across 321 tensors;
- parameters without gradients: 100,592,896 across 153 tensors.

The no-gradient parameters belong to the composite checkpoint's vision tower.
No images enter the text-only path, so the vision tower did not participate in
the graph and was not updated. The run must therefore be described as a
full-parameter configuration with language-backbone/head gradients, not as a
run that updated vision parameters.

## Epoch validation and checkpoint selection

| Epoch/checkpoint | Validation CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 | Predicted 10/20/40/80/160 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 / `step-000071` | 1.3593969378636512 | 0.43614718614718617 | 0.18808510033373074 | 0.3464346842238434 | 0.22002135398622552 | 0.6753246753246753 | 0/72/18/100/734 |
| 2 / `step-000142` | 1.4560290551804878 | 0.2727272727272727 | 0.17146113530710744 | 0.22858705955012748 | 0.2204381769640713 | 0.551948051948052 | 0/4/264/578/78 |
| 3 / `step-000213` | 1.367579244690024 | 0.34523809523809523 | **0.21763191244497584** | 0.3435657773957275 | 0.22993634120458348 | 0.6428571428571429 | 0/20/224/374/306 |

`step-000213` is selected because epoch 3 has the highest validation macro-F1.
Epoch 1 has higher accuracy and top-2, but those are not the primary selection
metric. The selected-checkpoint reload reproduced exact five-logit scores for
all 924 selected-epoch outputs.

## Final classification result

| Metric | Value |
|---|---:|
| Accuracy | 0.34523809523809523 |
| Macro-F1 | 0.21763191244497584 |
| Weighted F1 | 0.3435657773957275 |
| Balanced accuracy | 0.22993634120458348 |
| Top-2 accuracy | 0.6428571428571429 |
| Valid outputs | 924/924 = 100% |
| Invalid outputs | 0 |
| Invalid-output percentage | 0.0% |
| Majority class | 160 |
| Majority accuracy baseline | 0.45454545454545453 |
| Majority macro-F1 baseline | 0.125 |

Phase 2C is below the majority baseline on accuracy but above it on macro-F1.

| Source | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Oracle | 13 | 81 | 178 | 232 | 420 |
| Phase 2C prediction | 0 | 20 | 224 | 374 | 306 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 |
| 20 | 0.05 | 0.012345679012345678 | 0.019801980198019802 | 81 |
| 40 | 0.29017857142857145 | 0.3651685393258427 | 0.3233830845771144 | 178 |
| 80 | 0.23529411764705882 | 0.3793103448275862 | 0.29042904290429045 | 232 |
| 160 | 0.5392156862745098 | 0.39285714285714285 | 0.45454545454545453 | 420 |

Class 10 has zero recall and is never predicted. Class 20 is also very weak:
only 1 of 81 true class-20 examples is correct, while only 1 of 20 class-20
predictions is correct. This is recall 0.012345679012345678 and F1
0.019801980198019802.

The confusion matrix uses Oracle rows and predicted columns ordered 10, 20,
40, 80, 160:

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 5 | 7 | 1 |
| 20 | 0 | 1 | 19 | 39 | 22 |
| 40 | 0 | 4 | 65 | 66 | 43 |
| 80 | 0 | 5 | 64 | 88 | 75 |
| 160 | 0 | 10 | 71 | 174 | 165 |

## Unchanged downstream retrieval

The classifier class ID is mapped to its canonical chunk size before the
unchanged local Qdrant evaluator runs. Retrieval retains source-paper
filtering, predicted granularity, `top_k=5`, existing
`text-embedding-3-small` 1,536-dimensional vectors, cosine ranking, unchanged
chunk order and concatenation, GPT-2 evidence tokenization, and joined
token-level F1 version `qasper-token-prf-v2`.

| Retrieval metric | Value |
|---|---:|
| Coverage | 924/924 = 100% |
| Invalid predictions without retrieval | 0 |
| Valid-only mean joined retrieval F1 | 0.27914719588744585 |
| Valid-only median joined retrieval F1 | 0.2607245 |
| Coverage-adjusted full-set mean | 0.2791471958874462 |
| Retrieval wall time | 134.9306207000045 seconds |

Classification metrics measure prediction of the evidence-length Oracle
label. Joined retrieval F1 measures evidence-token overlap after downstream
retrieval. They are different outcomes and need not move together.

## Runtime and resources

| Measurement | Value |
|---|---:|
| Training wall time | 1276.56244828552 seconds |
| Initial/final recorded step loss | 1.6514464020729065 / 1.5576722860336303 |
| Selected epoch validation wall | 33.45399766601622 seconds |
| Training peak allocated/reserved GPU | 8.96875286102295 / 9.517578125 GiB |
| Training RSS | 1.96734619140625 GiB |
| Selected-checkpoint load | 2.5492455568164587 seconds |
| Isolated final inference | 33.99719780869782 seconds |
| Mean inference/question | 0.036642024783393394 seconds |
| Median inference/question | 0.03524067858234048 seconds |
| Final peak allocated/reserved GPU | 1.715855598449707 / 1.77734375 GiB |
| Final RSS | 1.6998367309570312 GiB |
| Training + final validation | 1313.1088916510344 seconds |
| Training + final validation + retrieval | 1448.0395123510389 seconds |

The initial and final loss values are individual optimizer-window losses, not
epoch averages. The last epoch ends with a partial five-example window, so the
two values must not be used by themselves to infer a training-loss trend.

## Five-way Qwen comparison

All rows use the same 924 validation questions, evidence-length Oracle, five
canonical chunk classes, and unchanged downstream retrieval protocol.

| Metric | Phase 1 | Phase 2 numeric | Phase 2B-A | Phase 2B-B | Phase 2C classifier |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.04004329004329004 | **0.4318181818181818** | 0.35064935064935066 | 0.37012987012987014 | 0.34523809523809523 |
| Macro-F1 | 0.049045932422555796 | 0.16502267760462996 | 0.20922603632601472 | 0.16836616836616836 | **0.21763191244497584** |
| Weighted F1 | 0.032612933907418644 | 0.32805741427623947 | 0.3406050804511769 | 0.3142183142183142 | **0.3435657773957275** |
| Balanced accuracy | 0.23369399361908724 | 0.20697865353037764 | **0.2383201416948027** | 0.20607553366174058 | 0.22993634120458348 |
| Top-2 | unavailable | unavailable | 0.6071428571428571 | **0.7056277056277056** | 0.6428571428571429 |
| Mean joined retrieval F1 | 0.23910868506493507 | 0.22658488852813854 | **0.28646775432900434** | 0.24962774025974027 | 0.27914719588744585 |

Phase 2C has the highest saved Qwen macro-F1 and weighted F1. Numeric-target
Phase 2 has the highest accuracy. Phase 2B-A has the highest balanced accuracy
and downstream mean joined retrieval F1. Phase 2B-B has the highest available
top-2; Phase 1 and Phase 2 do not expose comparable five-class scores.

## Interpretation and comparability limits

Phase 2C is benchmark-comparable because it preserves the validation
questions, evidence-length Oracle, canonical classes, and retrieval evaluator.
It is not a clean causal architecture ablation. It simultaneously changes:

- checkpoint family, from the chat/instruct model to the Base model;
- classifier formulation, from generated or next-token targets to a
  five-logit sequence-classification head;
- fixed prompt and plain-sequence formatting.

The macro-F1 increase therefore cannot be attributed to the head, Base
checkpoint, or prompt in isolation. The run uses one seed, and the same
validation split is used for checkpoint selection and reported comparison. It
does not measure run-to-run variance or held-out test generalization.

Previous Logistic Regression and MLP classification results use the old
retrieval-F1 Oracle and are not directly comparable. A scientifically stronger
follow-up should change one factor at a time, use multiple seeds, and reserve
an untouched test split under a predeclared protocol. The present result does
not support stronger causal or generalization claims.

## Artifacts and reproduction

Authoritative Phase 2C artifacts are under
`outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/`. They
include experiment/preflight configuration; run configuration and dataset
manifest; formatted-example and gradient-coverage audits; complete histories
and selected-checkpoint metadata; canonical predictions, raw/parsed/invalid
records and runtime; classification metrics, confusion matrix, and histogram;
retrieval records, runtime segments, and summary; and `final_summary.json`.

The original run metadata intentionally preserves `/dev/shm/...` paths from
the ephemeral CUDA host as provenance. The locally retained selected
checkpoint is under the corresponding repository-relative output tree at
`runs/.../checkpoints/step-000213/`. Its archive and nine extracted files were
verified against the remote SHA-256 values; the verification record is
`outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/integrity/selected_checkpoint_transfer_verification.json`.

The five-way comparison is
`outputs/qwen_phase2c_comparison_evidence_length_oracle/five_way_comparison.json`.
The standalone report is
`reports/qwen_phase2c_sequence_classifier_evidence_length_oracle/experiment_report.md`.

CUDA-host commands:

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
