# Phase 2C experiment report: Qwen Base sequence classifier

## Experiment status and identity

- Status: `complete`.
- Phase: `Phase 2C Base sequence-classification fine-tuning`.
- Run ID:
  `qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1`.
- Formulation: `qwen-phase2c-base-sequence-classifier-v1`.
- Experiment fingerprint:
  `6508ded1f9c25b451207f891f90fa5c9a7a4c09da7ea7125555bfa0ec7faca90`.
- Model: `Qwen/Qwen3.5-0.8B-Base`.
- Model revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Evaluated validation examples: 924.
- Valid outputs: 924/924; invalid outputs: 0.
- Retrieval: `complete`, 924/924, coverage 1.0.

This output tree is isolated from and does not overwrite completed Qwen
Phases 1, 2, 2B-A, or 2B-B.

## Supervisor motivation

Phase 2C evaluates the supervisor-motivated alternative of using a Base
checkpoint as a conventional five-class sequence classifier. The revised
instruction describes semantic context-length options, and the model returns
five directly comparable class-head logits rather than generated text. The
purpose is to establish a direct-classification benchmark, not to claim in
advance that any one architectural or prompt change is responsible for an
outcome.

The exact fixed instruction is:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = very short context, 2 = short context, 3 = medium context, 4 = long context, 5 = very long context. Return only the number

Instruction SHA-256:
`9e879535647c2bfcd3627d0d65f84c36a1bf442ed95bb5b07029c878ca990de7`.

## Input, labels, and decision rule

The plain sequence input is:

```text
{instruction}

Question: {original_question_text}
```

The model receives only the fixed instruction and original question text. It
does not receive evidence, evidence length, answers, paper text, embeddings,
retrieved chunks, retrieval scores, metadata, or handcrafted features.

The architecture is `AutoModelForSequenceClassification` with
`problem_type=single_label_classification` and five logits. The bias-free
classifier head is `score.weight` with shape 5×1024. Class mappings are:

| Class-head ID | Canonical chunk size |
|---:|---:|
| 0 | 10 |
| 1 | 20 |
| 2 | 40 |
| 3 | 80 |
| 4 | 160 |

Training uses uniform unweighted five-class cross-entropy. Inference is
deterministic argmax over the same five logits, and top-2 accuracy comes from
their directly comparable scores. There is no chat-template formatting,
assistant target token, generation, decoding, parser, fallback, or default
granularity. The saved tokenizer may carry model metadata, but no chat template
participates in the input path.

The tokenizer is `Qwen2Tokenizer`, with right padding, the model's end-of-text
token as padding (ID 248044), and special tokens enabled. No example is
truncated:

| Split | Minimum tokens | Maximum tokens | Mean tokens | Over 128 |
|---|---:|---:|---:|---:|
| Train | 86 | 112 | 92.13363028953229 | 0 |
| Validation | 87 | 115 | 91.59740259740259 | 0 |

## Preserved data and Oracle

Phase 2C uses the exact evidence-length Oracle and preserved split records used
by the preceding Qwen experiments.

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 | Oracle SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 | `64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88` |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 | `ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d` |

The Oracle counts GPT-2 tokens in complete stripped, exact-deduplicated,
deterministically ordered ground-truth evidence; chooses the closest size from
10, 20, 40, 80, and 160; resolves exact midpoint ties toward the smaller size;
and clips values below 10 or above 160 to the endpoints. It is independent of
retrieval F1, embedding quality, cosine similarity, retrieved chunks, and
router performance.

Class 160 supplies 420/924 = 45.45% of validation, while class 10 supplies 13
examples. No QASPER test example was loaded or evaluated.

## Model loading and environment

The initial Base load deliberately adds a new classifier head. The loading
audit reports only `score.weight` as an expected missing key, no unexpected or
mismatched keys, and no error. Seed 42 was set before model loading. The
initial head's float32 SHA-256 was
`09826669f451891218742ea86926e0b484d1696e57999276889d97b5ccdcbda5`.
The selected checkpoint reload had no missing, unexpected, or mismatched keys;
its head float32 SHA-256 was
`19368b0656a72304eb41a0f2fa9fca72d569d514ed960572cdbf8cebd65601bf`.

| Item | Recorded value |
|---|---|
| Environment | `.venv-qwen` |
| Python | `3.10.7 (main, Oct  3 2022, 02:19:58) [Clang 14.0.3 ]` |
| Executable | `/workspace/thesis-granularity-router/.venv-qwen/bin/python` |
| Transformers | `5.15.0.dev0` |
| Transformers commit | `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7` |
| PyTorch | `2.8.0+cu128` |
| CUDA build | 12.8 |
| GPU | `NVIDIA A100-SXM4-40GB` |
| Device/dtype | CUDA / `torch.bfloat16` |
| Quantization | none |
| TensorBoard | `2.20.0` |
| Repository commit | `55af1bcbc4d7a089adaafd4da539581b2dbbed67` |
| Training-script SHA-256 | `6eeb155296b239463d5ba7c8c75dfed8dd59f8c5285cab1edf7cb6a553f9aefb` |
| Resume-contract SHA-256 | `09fdebe2660fbed57d93ea712ab9f3918e139011bf637d6652231a3d36a11473` |

The protected legacy `.venv`, system Python, local Phase 1 environment, and
all earlier experiment artifact trees remained separate and unchanged.

## Training configuration

- Full-parameter sequence-classification configuration.
- Parameters marked trainable/total: 852,991,040 / 852,991,040.
- Objective: uniform, unweighted, five-class cross-entropy.
- Optimizer: AdamW over the marked-trainable parameters.
- Epochs: 3 fixed, with no early stopping.
- Parameter-update/global steps: 213, with 71 per epoch.
- Per-device batch: 4.
- Gradient accumulation: 8.
- Effective batch: 32.
- Maximum sequence length: 128.
- Learning rate: `2e-5`.
- Weight decay: `0.01`.
- Scheduler: cosine.
- Warmup: 5%, 11 steps.
- Gradient clipping: `1.0`.
- Seed: 42.
- Evaluation/checkpointing: end of each epoch.
- Checkpoint retention: current and best during training, selected only at
  completion.
- Selection: highest validation macro-F1, then accuracy, weighted F1,
  balanced accuracy, lower validation loss, and earlier step.

## Gradient coverage audit

The saved gradient audit status is `passed`:

- classifier head received gradients: true;
- language backbone received gradients: true;
- parameters with gradients: 752,398,144 across 321 tensors;
- parameters without gradients: 100,592,896 across 153 tensors.

The without-gradient samples are all from `model.visual.*`. Because the run is
text-only and supplies no image input, the composite model's vision tower does
not participate in the graph and does not receive updates. Thus every
parameter is configured trainable, but only the language path and classifier
head have observed gradients. This is not evidence of a training failure and
must not be reported as vision-tower fine-tuning.

## Complete epoch validation history

| Epoch/checkpoint | Uniform CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 | Predicted 10/20/40/80/160 | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 / `step-000071` | 1.3593969378636512 | 0.43614718614718617 | 0.18808510033373074 | 0.3464346842238434 | 0.22002135398622552 | 0.6753246753246753 | 0/72/18/100/734 | 34.56756311096251 |
| 2 / `step-000142` | 1.4560290551804878 | 0.2727272727272727 | 0.17146113530710744 | 0.22858705955012748 | 0.2204381769640713 | 0.551948051948052 | 0/4/264/578/78 | 33.42555764876306 |
| 3 / `step-000213` | 1.367579244690024 | 0.34523809523809523 | **0.21763191244497584** | 0.3435657773957275 | 0.22993634120458348 | 0.6428571428571429 | 0/20/224/374/306 | 33.45399766601622 |

Epoch 1 confusion matrix, with Oracle rows and predicted columns ordered 10,
20, 40, 80, 160:

```text
[0,0,0,3,10]
[0,7,1,17,56]
[0,22,4,16,136]
[0,25,7,30,170]
[0,18,6,34,362]
```

Epoch 2:

```text
[0,0,4,9,0]
[0,1,20,53,7]
[0,1,71,99,7]
[0,1,75,136,20]
[0,1,94,281,44]
```

Epoch 3:

```text
[0,0,5,7,1]
[0,1,19,39,22]
[0,4,65,66,43]
[0,5,64,88,75]
[0,10,71,174,165]
```

## Selected checkpoint

`step-000213` from epoch 3 is selected by the highest validation macro-F1,
0.21763191244497584. No tie-break was required. Epoch 1 has higher accuracy
and top-2 accuracy, but neither is the primary selection metric. The selected
checkpoint reload reproduced exact five-logit scores for all 924 selected-epoch
outputs.

The locally retained selected checkpoint is under:

`outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/runs/qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1/checkpoints/step-000213/`

## Final classification metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.34523809523809523 |
| Macro-F1 | 0.21763191244497584 |
| Weighted F1 | 0.3435657773957275 |
| Balanced accuracy | 0.22993634120458348 |
| Top-2 accuracy | 0.6428571428571429 |
| Top-2 status | available from comparable five-class head logits |
| Evaluated examples | 924 |
| Valid predictions | 924 |
| Invalid predictions | 0 |
| Majority class | 160 |
| Majority accuracy | 0.45454545454545453 |
| Majority macro-F1 | 0.125 |

The classifier is below the class-160 majority baseline on accuracy and above
it on macro-F1.

| Source | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Oracle | 13 | 81 | 178 | 232 | 420 |
| Prediction | 0 | 20 | 224 | 374 | 306 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 |
| 20 | 0.05 | 0.012345679012345678 | 0.019801980198019802 | 81 |
| 40 | 0.29017857142857145 | 0.3651685393258427 | 0.3233830845771144 | 178 |
| 80 | 0.23529411764705882 | 0.3793103448275862 | 0.29042904290429045 | 232 |
| 160 | 0.5392156862745098 | 0.39285714285714285 | 0.45454545454545453 | 420 |

Class 10 has zero recall and is never predicted. Class 20 remains extremely
weak: only one of its 81 reference examples is correct, and only one of 20
class-20 predictions is correct. Its recall is 0.012345679012345678 and F1 is
0.019801980198019802.

Final confusion matrix:

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 5 | 7 | 1 |
| 20 | 0 | 1 | 19 | 39 | 22 |
| 40 | 0 | 4 | 65 | 66 | 43 |
| 80 | 0 | 5 | 64 | 88 | 75 |
| 160 | 0 | 10 | 71 | 174 | 165 |

## End-to-end retrieval

The five-logit class ID is mapped to its canonical chunk size before the
unchanged retrieval evaluator runs. The evaluation uses the existing local
Qdrant collections, source-paper filtering, predicted granularity, `top_k=5`,
`text-embedding-3-small` with dimension 1,536, cosine similarity, unchanged
chunk ordering and concatenation, GPT-2 evidence tokenization, and
`f1_joined_topk` version `qasper-token-prf-v2`.

| Retrieval metric | Value |
|---|---:|
| Status | complete |
| Evaluated/retrieved | 924/924 |
| Invalid predictions without retrieval | 0 |
| Coverage | 1.0 |
| Valid-only mean joined retrieval F1 | 0.27914719588744585 |
| Valid-only median joined retrieval F1 | 0.2607245 |
| Coverage-adjusted full-set mean | 0.2791471958874462 |
| Evaluation configuration SHA-256 | `9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8` |
| Retrieval wall time | 134.9306207000045 seconds |
| Durable per-question processing sum | 129.5303218982299 seconds |

Every prediction is an explicit class-head argmax. No invalid output is mapped
to a default class. Classification metrics and joined retrieval F1 measure
different experiment stages and are not interchangeable.

## Runtime and resources

Training:

- elapsed wall time: 1276.56244828552 seconds;
- initial/final recorded optimizer-step loss:
  1.6514464020729065 / 1.5576722860336303;
- peak GPU allocated/reserved:
  8.96875286102295 / 9.517578125 GiB;
- recorded process RSS: 1.96734619140625 GiB.

The two reported loss values are individual optimizer-window losses, not
epoch means. The final window contains five examples rather than the nominal
effective batch of 32, so their difference is not a standalone estimate of
the training-loss trend.

Selected-checkpoint final validation:

- model load: 2.5492455568164587 seconds;
- isolated inference wall: 33.99719780869782 seconds;
- mean inference/question: 0.036642024783393394 seconds;
- median inference/question: 0.03524067858234048 seconds;
- synchronized batch-forward sum: 33.857230899855494 seconds;
- peak GPU allocated/reserved:
  1.715855598449707 / 1.77734375 GiB;
- process RSS: 1.6998367309570312 GiB.

Known combined durations:

- training plus selected-checkpoint load/inference:
  1313.1088916510344 seconds;
- training, selected-checkpoint load/inference, and retrieval:
  1448.0395123510389 seconds.

## Five-way Qwen comparison

| Metric | Phase 1 zero-shot | Phase 2 numeric SFT | Phase 2B-A unweighted | Phase 2B-B balanced | Phase 2C classifier |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.04004329004329004 | **0.4318181818181818** | 0.35064935064935066 | 0.37012987012987014 | 0.34523809523809523 |
| Macro-F1 | 0.049045932422555796 | 0.16502267760462996 | 0.20922603632601472 | 0.16836616836616836 | **0.21763191244497584** |
| Weighted F1 | 0.032612933907418644 | 0.32805741427623947 | 0.3406050804511769 | 0.3142183142183142 | **0.3435657773957275** |
| Balanced accuracy | 0.23369399361908724 | 0.20697865353037764 | **0.2383201416948027** | 0.20607553366174058 | 0.22993634120458348 |
| Top-2 accuracy | unavailable | unavailable | 0.6071428571428571 | **0.7056277056277056** | 0.6428571428571429 |
| Mean joined retrieval F1 | 0.23910868506493507 | 0.22658488852813854 | **0.28646775432900434** | 0.24962774025974027 | 0.27914719588744585 |

Phase 2C has the best saved Qwen validation macro-F1 so far at
0.21763191244497584, and also the highest weighted F1. Numeric-target Phase 2
has the best accuracy, 0.4318181818181818. Phase 2B-A has the best balanced
accuracy and downstream mean joined retrieval F1, 0.28646775432900434, while
Phase 2C retrieval is 0.27914719588744585.

## Interpretation and limitations

The five Qwen runs use the same 924 validation questions, evidence-length
Oracle, five chunk classes, and downstream retrieval protocol. They are
benchmark-comparable with those shared conditions stated.

Phase 2C is not a clean causal architecture ablation. It changes the following
simultaneously:

- checkpoint family: Base rather than the chat/instruct checkpoint;
- classifier formulation: a five-logit sequence head rather than generated
  numeric text or restricted next-token aliases;
- fixed prompt and plain-sequence formatting.

The observed macro-F1 gain cannot be attributed to one change in isolation.
Furthermore, this is a single seed; the validation split is used for both
checkpoint selection and final reporting; and no QASPER test result is
available. The result does not establish run-to-run stability, causal benefit
from the classifier head, or held-out generalization.

Earlier Logistic Regression and MLP classification results use the old
retrieval-F1 Oracle and are not directly comparable. A stronger follow-up
should use controlled one-factor changes, multiple seeds, and an untouched
test split under a predeclared protocol.

## Artifact inventory

Configuration:

- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/configuration/experiment.json`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/configuration/preflight_manifest.json`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/final_summary.json`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/integrity/selected_checkpoint_transfer_verification.json`

Run records under
`outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/runs/qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1/`:

- `training_config.json`
- `dataset_manifest.json`
- `formatted_example_inspection.json`
- `gradient_coverage_audit.json`
- `training_history.jsonl`
- `validation_history.jsonl`
- `checkpoint_manifest.json`
- `best_checkpoint.json`
- `summary.json`
- `validation/predictions_step-000071.jsonl`
- `validation/predictions_step-000142.jsonl`
- `validation/predictions_step-000213.jsonl`
- retained `checkpoints/step-000213/`

Absolute `/dev/shm/...` paths inside the immutable run metadata record the
original ephemeral CUDA-host locations. For local use, resolve them to the
corresponding repository-relative output-tree paths above. The integrity
record confirms that the remote and local selected-checkpoint archive hash and
all nine extracted file hashes match.

Canonical validation:

- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/validation/predictions.jsonl`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/validation/raw_outputs.jsonl`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/validation/parsed_predictions.jsonl`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/validation/invalid_outputs.jsonl`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/validation/runtime_summary.json`

Classification and retrieval:

- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/classification/metrics.json`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/classification/confusion_matrix.csv`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/classification/predicted_vs_oracle.svg`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/retrieval/results.jsonl`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/retrieval/runtime_segments.jsonl`
- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/retrieval/summary.json`

TensorBoard and comparison:

- `outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/tensorboard/qwen-phase2c-base-sequence-classifier-full-parameter-20260804-seed42-v1/`
- `outputs/qwen_phase2c_comparison_evidence_length_oracle/five_way_comparison.json`

## Exact reproduction commands

On the compatible CUDA host, from the project root:

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
