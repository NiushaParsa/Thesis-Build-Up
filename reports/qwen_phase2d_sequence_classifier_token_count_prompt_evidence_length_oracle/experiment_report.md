# Phase 2D experiment report: exact-token-count prompt ablation

## Experiment status and identity

- Status: `complete`.
- Phase: `Phase 2D Base sequence-classification fine-tuning`.
- Run ID:
  `qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1`.
- Formulation:
  `qwen-phase2d-base-sequence-classifier-token-count-prompt-v1`.
- Experiment fingerprint:
  `dad60bd9a0530865110c2310f62a896c73350fa383c7812d5c6733e376bc377d`.
- Model: `Qwen/Qwen3.5-0.8B-Base`.
- Model revision:
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Evaluated validation examples: 924.
- Valid predictions: 924/924; invalid predictions: 0.
- Retrieval: complete, 924/924, coverage 1.0.
- Selected checkpoint: `step-000213`, epoch 3.

Phase 2D is isolated from the completed Phase 1, 2, 2B, and 2C artifact
trees. It is designed as a one-factor, single-seed prompt ablation against
Phase 2C.

## Objective and exact prompt intervention

The experiment tests whether stating the exact candidate chunk-token counts
inside the routing instruction changes the performance of the otherwise
unchanged Phase 2C sequence classifier.

The exact Phase 2D instruction is:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, 4 = 80 tokens, 5 = 160 tokens. Return only the number

Instruction SHA-256:
`b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5`.

Phase 2C used the same surrounding text with this mapping:

```text
1 = very short context, 2 = short context, 3 = medium context,
4 = long context, 5 = very long context
```

Phase 2D replaces only that mapping with:

```text
1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens,
4 = 80 tokens, 5 = 160 tokens
```

Phase 2C instruction SHA-256:
`9e879535647c2bfcd3627d0d65f84c36a1bf442ed95bb5b07029c878ca990de7`.
The saved comparison artifact's `prompt_only_protocol_audit` has status
`passed` and records the relationship as
`prompt_only_single_seed_ablation`.

## Frozen Phase 2C--2D protocol

The following values are the same in both runs:

- Base model ID and pinned revision;
- five-logit `AutoModelForSequenceClassification` architecture;
- class IDs and canonical chunk-size mappings;
- initial classifier-head weights under seed 42;
- complete preserved train and validation records and their order;
- evidence-length Oracle labels and hashes;
- uniform five-class cross-entropy;
- full-parameter configuration;
- optimizer, learning rate, weight decay, schedule, warmup, batch sizes,
  gradient accumulation, clipping, and fixed three epochs;
- deterministic seed and checkpoint-selection rule;
- tokenizer, maximum sequence length, padding, and no-truncation contract;
- downstream retrieval identity, paper restriction, top-k, embeddings,
  similarity/ranking, concatenation, tokenizer, and joined F1.

The allowed provenance differences are timestamps, formulation and prompt
identity, run ID, output root, repository field, script hash, experiment
fingerprint, and resume-contract hash. Token sequence lengths also differ as a
necessary consequence of changing the prompt.

## Input, labels, and decision rule

The exact plain sequence input is:

```text
{instruction}

Question: {original_question_text}
```

Only the fixed instruction and original question text enter the model. The
model receives no evidence, evidence length, answer, paper text, retrieved
chunk, retrieval score, question embedding, metadata, or handcrafted feature.
There is no chat-template input, assistant target token, text generation,
decoding, output parser, parser fallback, or default class.

`AutoModelForSequenceClassification` returns five directly comparable logits.
The model configuration uses `problem_type=single_label_classification` and
these mappings:

| Class-head ID | Canonical chunk size |
|---:|---:|
| 0 | 10 |
| 1 | 20 |
| 2 | 40 |
| 3 | 80 |
| 4 | 160 |

The bias-free classifier head is `score.weight` with shape 5 x 1024. Training
uses uniform unweighted cross-entropy. Final prediction is deterministic
argmax over the five logits, and top-2 accuracy is computed from those same
comparable scores.

The tokenizer is `Qwen2Tokenizer`, uses right padding, uses the end-of-text
token as padding (ID 248044), enables special tokens, and does not truncate.
Both the top-level and nested text configuration explicitly carry the padding
ID and disable cache use.

| Split | Minimum tokens | Maximum tokens | Mean tokens | Over 128 |
|---|---:|---:|---:|---:|
| Train | 95 | 121 | 101.13363028953229 | 0 |
| Validation | 96 | 124 | 100.59740259740259 | 0 |

Relative to Phase 2C, each split's minimum, maximum, and mean is nine tokens
higher. All 3,169 records remain within the fixed 128-token limit, so the
prompt change does not introduce truncation.

## Preserved data and evidence-length Oracle

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 | Oracle SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 | `64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88` |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 | `ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d` |

The Oracle counts GPT-2 tokens in the complete stripped,
exact-deduplicated, deterministically ordered reference evidence. It selects
the closest candidate from 10, 20, 40, 80, and 160, resolves exact midpoint
ties toward the smaller candidate, and clips outside the 10--160 range. It is
independent of retrieval F1, embedding quality, cosine similarity, retrieved
chunks, and router performance.

Validation is strongly imbalanced:

| Class | Count | Percentage |
|---:|---:|---:|
| 10 | 13 | 1.406926406926407% |
| 20 | 81 | 8.766233766233766% |
| 40 | 178 | 19.264069264069263% |
| 80 | 232 | 25.10822510822511% |
| 160 | 420 | 45.45454545454545% |

No QASPER test example was loaded or evaluated.

## Model loading and environment

The initial Base-model load expected only `score.weight` to be absent. The
loading audit found no unexpected or mismatched keys and no loading error.
Seed 42 was set before model loading. The initial head float32 SHA-256 is:

`09826669f451891218742ea86926e0b484d1696e57999276889d97b5ccdcbda5`

This exactly matches Phase 2C's initial head hash. The selected checkpoint
reload had no missing, unexpected, or mismatched keys. Its trained head hash
is:

`cbc44bdd71b91be4a7f97c19f87651003eac05b292eb61ea5634dee0cb351025`

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
| Recorded repository field | `40f79e1` |
| Full base repository HEAD | `40f79e12dbe27ab42934d10732188c1d76087b17` |
| Executed training-script SHA-256 | `99ba4f9a57b5547e776d81d2c7c94aece2670a9b5ccaf348c8a1fbeb0baa5efc` |
| Resume-contract SHA-256 | `de92814b2a4b25eedc6a1e0c6383ddf8313449869cd80a54fb6a35eca5f534fa` |

Repository provenance requires a qualification. The saved `40f79e1` value is
an abbreviation of the base HEAD
`40f79e12dbe27ab42934d10732188c1d76087b17`. The new Phase 2D files were
uncommitted at launch and were copied separately to the CUDA host, so the base
commit does not contain Phase 2D. The authoritative identity of the executed
training code is its SHA-256
`99ba4f9a57b5547e776d81d2c7c94aece2670a9b5ccaf348c8a1fbeb0baa5efc`.
The eventual repository commit containing Phase 2D must be recorded as a
separate, later provenance event.

The legacy `.venv`, system Python, and all earlier experiment environments and
artifacts remained unchanged.

## Training configuration

- Method: full-parameter sequence-classification configuration.
- Parameters marked trainable/total: 852,991,040 / 852,991,040.
- Objective: uniform, unweighted, five-class cross-entropy.
- Optimizer: AdamW.
- Fixed epochs: 3, with no early stopping.
- Parameter updates: 213, 71 per epoch.
- Per-device batch: 4.
- Gradient accumulation: 8.
- Effective batch: 32.
- Maximum sequence length: 128.
- Learning rate: `2e-5`.
- Weight decay: `0.01`.
- Scheduler: cosine.
- Warmup: 5%, 11 optimizer steps.
- Gradient clipping: `1.0`.
- Seed: 42.
- Logging: every optimizer step.
- Evaluation and checkpointing: end of each epoch.
- Retention: current and best during training; selected checkpoint only at
  completion.
- Selection: validation macro-F1, then accuracy, weighted F1, balanced
  accuracy, lower validation CE, and earlier step.

The first recorded optimizer-window loss is 1.5487420707941055. The final
recorded optimizer-window loss is 1.5045942068099976. These are not epoch
means. The last optimizer update contains a partial five-example accumulation
window rather than the nominal effective batch of 32, so this pair alone must
not be interpreted as a learning curve.

## Gradient coverage audit

The audit status is `passed`:

- classifier head received gradients: true;
- language backbone received gradients: true;
- parameters with gradients: 752,398,144 across 321 tensors;
- parameters without gradients: 100,592,896 across 153 tensors.

The no-gradient samples are from `model.visual.*`. Because the experiment
supplies only text, the composite checkpoint's vision tower is not in the
active graph and receives no update. All parameters are marked trainable, but
only the active language path and classifier head have observed gradients.

## Complete epoch validation history

| Epoch/checkpoint | Uniform CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 | Predicted 10/20/40/80/160 | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 1 / `step-000071` | 1.330027622577948 | 0.44047619047619047 | 0.18072924091134496 | 0.34316665343597114 | 0.2169532849947418 | 0.698051948051948 | 0/0/122/51/751 | 32.7092076446861 |
| 2 / `step-000142` | 1.4269125938931584 | 0.2803030303030303 | 0.1678216855285752 | 0.2325661147417672 | 0.21978505931624878 | 0.5844155844155844 | 0/5/209/637/73 | 31.841020613908768 |
| 3 / `step-000213` | 1.3543887827303502 | 0.36904761904761907 | **0.22994524079282935** | 0.3644656337102369 | 0.2391812745015638 | 0.6341991341991342 | 0/16/219/332/357 | 31.920586789026856 |

Per-class epoch metrics are shown as precision / recall / F1. Support is fixed
at 13, 81, 178, 232, and 420 for classes 10, 20, 40, 80, and 160.

| Class | Epoch 1 P / R / F1 | Epoch 2 P / R / F1 | Epoch 3 P / R / F1 |
|---:|---|---|---|
| 10 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| 20 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.125 / 0.024691358024691357 / 0.041237113402061855 |
| 40 | 0.22950819672131148 / 0.15730337078651685 / 0.18666666666666668 | 0.2679425837320574 / 0.3146067415730337 / 0.289405684754522 | 0.2876712328767123 / 0.3539325842696629 / 0.31738035264483627 |
| 80 | 0.2549019607843137 / 0.05603448275862069 / 0.09187279151943464 | 0.24489795918367346 / 0.6724137931034483 / 0.35903337169159955 | 0.25 / 0.3577586206896552 / 0.29432624113475175 |
| 160 | 0.4873501997336884 / 0.8714285714285714 / 0.6251067463706235 | 0.6438356164383562 / 0.11190476190476191 / 0.19066937119675453 | 0.5406162464985994 / 0.4595238095238095 / 0.49678249678249675 |

Epoch 1 confusion matrix, with Oracle rows and predicted columns ordered 10,
20, 40, 80, 160:

```text
[0, 0, 2, 1, 10]
[0, 0, 14, 7, 60]
[0, 0, 28, 10, 140]
[0, 0, 44, 13, 175]
[0, 0, 34, 20, 366]
```

Epoch 2:

```text
[0, 0, 3, 10, 0]
[0, 0, 17, 61, 3]
[0, 0, 56, 114, 8]
[0, 1, 60, 156, 15]
[0, 4, 73, 296, 47]
```

Epoch 3:

```text
[0, 0, 3, 7, 3]
[0, 2, 21, 33, 25]
[0, 4, 63, 60, 51]
[0, 4, 60, 83, 85]
[0, 6, 72, 149, 193]
```

The prediction distribution changes substantially across epochs. Epoch 1
selects class 160 for 751 questions; epoch 2 selects class 80 for 637; epoch 3
is less concentrated but still never selects class 10. This is factual
evidence of checkpoint-to-checkpoint variability within the fixed run.

## Selected checkpoint

Epoch 3, `step-000213`, is selected because its macro-F1
0.22994524079282935 is greater than epoch 1's 0.18072924091134496 and epoch
2's 0.1678216855285752. No tie-break is required. Epoch 1 instead has the
lowest validation CE, highest accuracy, highest top-2 accuracy, and highest
class-160 F1. The difference follows the predeclared macro-F1 selection rule.

The final selected-checkpoint reload exactly reproduces all 924 selected-epoch
score rankings and predictions. The locally retained checkpoint is:

`outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/runs/qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1/checkpoints/step-000213/`

## Final classification metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.36904761904761907 |
| Macro-F1 | 0.22994524079282935 |
| Weighted F1 | 0.3644656337102369 |
| Balanced accuracy | 0.2391812745015638 |
| Top-2 accuracy | 0.6341991341991342 |
| Top-2 status | available from comparable five-class head logits |
| Evaluated examples | 924 |
| Valid predictions | 924 |
| Invalid predictions | 0 |
| Invalid-output percentage | 0.0% |
| Majority class | 160 |
| Majority accuracy | 0.45454545454545453 |
| Majority macro-F1 | 0.125 |

The model correctly classifies 341/924 examples. A constant class-160
classifier would correctly classify 420/924. Phase 2D is therefore below the
majority baseline in accuracy while above it in macro-F1.

Oracle and predicted distributions:

| Source | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Oracle count | 13 | 81 | 178 | 232 | 420 |
| Oracle percentage | 1.4069% | 8.7662% | 19.2641% | 25.1082% | 45.4545% |
| Prediction count | 0 | 16 | 219 | 332 | 357 |
| Prediction percentage | 0.0000% | 1.7316% | 23.7013% | 35.9307% | 38.6364% |

Final per-class metrics:

| Class | Precision | Recall | F1 | Support | Correct |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 | 0 |
| 20 | 0.125 | 0.024691358024691357 | 0.041237113402061855 | 81 | 2 |
| 40 | 0.2876712328767123 | 0.3539325842696629 | 0.31738035264483627 | 178 | 63 |
| 80 | 0.25 | 0.3577586206896552 | 0.29432624113475175 | 232 | 83 |
| 160 | 0.5406162464985994 | 0.4595238095238095 | 0.49678249678249675 | 420 | 193 |

Class 10 has zero recall at every epoch and is never predicted. The exact
token-count prompt therefore does not solve the rarest-class failure. Class 20
also remains poorly represented: 2 of 81 reference class-20 examples are
correct, and 2 of the 16 class-20 predictions are correct.

Final confusion matrix:

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 3 | 7 | 3 |
| 20 | 0 | 2 | 21 | 33 | 25 |
| 40 | 0 | 4 | 63 | 60 | 51 |
| 80 | 0 | 4 | 60 | 83 | 85 |
| 160 | 0 | 6 | 72 | 149 | 193 |

## Phase 2C--2D paired delta

The saved audit verifies the intended controlled relationship. The following
table reports Phase 2D minus Phase 2C without rounding the stored values.

| Metric | Phase 2C | Phase 2D | Delta |
|---|---:|---:|---:|
| Selected validation CE | 1.367579244690024 | 1.3543887827303502 | -0.0131904619596738 |
| Accuracy | 0.34523809523809523 | 0.36904761904761907 | +0.023809523809523836 |
| Macro-F1 | 0.21763191244497584 | 0.22994524079282935 | +0.012313328347853508 |
| Weighted F1 | 0.3435657773957275 | 0.3644656337102369 | +0.020899856314509357 |
| Balanced accuracy | 0.22993634120458348 | 0.2391812745015638 | +0.009244933296980312 |
| Top-2 accuracy | 0.6428571428571429 | 0.6341991341991342 | -0.008658008658008698 |
| Mean joined retrieval F1 | 0.27914719588744585 | 0.2767166677489178 | -0.0024305281385280653 |
| Median joined retrieval F1 | 0.2607245 | 0.2558975 | -0.004827000000000026 |

Prediction-distribution delta:

| Class | Phase 2C | Phase 2D | Delta |
|---:|---:|---:|---:|
| 10 | 0 | 0 | 0 |
| 20 | 20 | 16 | -4 |
| 40 | 224 | 219 | -5 |
| 80 | 374 | 332 | -42 |
| 160 | 306 | 357 | +51 |

Per-class Phase 2D minus Phase 2C:

| Class | Precision delta | Recall delta | F1 delta |
|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 |
| 20 | +0.075 | +0.012345679012345678 | +0.021435133204042053 |
| 40 | -0.0025073385518591396 | -0.011235955056179803 | -0.006002731932278127 |
| 80 | +0.01470588235294118 | -0.021551724137930994 | +0.0038971982304613073 |
| 160 | +0.0014005602240896309 | +0.06666666666666665 | +0.04223704223704222 |

The explicit-count prompt is associated with more class-160 and fewer
class-80 predictions. The largest per-class F1 improvement is class 160;
class 20 also improves from a very low starting point. Class 40 declines
slightly and class 10 remains absent. The overall classification gains coexist
with lower top-2 and retrieval scores.

## End-to-end retrieval

The predicted class ID is mapped to the corresponding canonical chunk size
before the unchanged retrieval evaluator runs. The evaluator uses the existing
Qdrant collections, source-paper filtering, predicted granularity, `top_k=5`,
`text-embedding-3-small` vectors with dimension 1,536, cosine ranking,
unchanged chunk order and concatenation, GPT-2 evidence tokenization, and
joined token-level F1.

| Retrieval item | Value |
|---|---:|
| Status | complete |
| Evaluated predictions | 924 |
| Valid-prediction retrievals | 924 |
| Invalid predictions without retrieval | 0 |
| Coverage | 1.0 |
| Valid-only mean joined retrieval F1 | 0.2767166677489178 |
| Valid-only median joined retrieval F1 | 0.2558975 |
| Coverage-adjusted full-set mean | 0.27671666774891795 |
| Top-k | 5 |
| Paper restricted | true |
| Embedding model/dimension | `text-embedding-3-small` / 1,536 |
| Tokenizer | GPT-2 |
| Metric | `f1_joined_topk` |
| Metric version | `qasper-token-prf-v2` |
| Schema version | 2 |
| Normalization | `lowercase-remove-punctuation-collapse-whitespace-v1` |
| Configuration SHA-256 | `9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8` |
| Retrieval wall | 151.0063940999098 seconds |
| Durable per-question sum | 143.97760520025622 seconds |
| Runtime segments | 1 |
| Current segment wall | 151.0063940999098 seconds |
| Completed invocation wall | 151.0063940999098 seconds |
| Complete uninterrupted run wall | 151.0063940999098 seconds |
| Reported runtime basis | `complete_uninterrupted_invocation` |

Every canonical classifier output is an explicit argmax. No parser or default
class is involved. If an invalid prediction had existed, it would have been
excluded rather than mapped; in this run there are none, so valid-only and
full-set coverage are both 100%.

Classification accuracy, classification macro-F1, weighted F1, and joined
retrieval F1 are separate measurements. The first three score Oracle-label
prediction; joined retrieval F1 measures token overlap after downstream
same-paper retrieval. Phase 2D improves the classification metrics over Phase
2C but slightly lowers retrieval F1, illustrating why the two stages must be
reported separately.

## Runtime and resources

Training:

- wall time: 1224.5802961867303 seconds, approximately 20 minutes 24.58
  seconds;
- initial/final optimizer-window loss:
  1.5487420707941055 / 1.5045942068099976;
- peak allocated/reserved GPU memory:
  9.0316162109375 / 9.6015625 GiB;
- process RSS: 1.9677543640136719 GiB.

Selected-checkpoint final validation:

- model load: 2.7541816290467978 seconds;
- isolated inference wall: 34.72815803065896 seconds;
- selected-epoch validation wall: 31.920586789026856 seconds;
- mean inference/question: 0.0374373855065248 seconds;
- median inference/question: 0.03656412195414305 seconds;
- synchronized batch-forward sum: 34.59214420802891 seconds;
- peak allocated/reserved GPU memory:
  1.7161517143249512 / 1.77734375 GiB;
- process RSS: 1.7001190185546875 GiB;
- selected-epoch exact score match: true for 924/924 outputs.

Retrieval and combined durations:

- retrieval wall: 151.0063940999098 seconds;
- training plus selected-checkpoint load/inference:
  1262.062635846436 seconds;
- known training, selected-checkpoint load/inference, and retrieval:
  1413.0690299463458 seconds, approximately 23 minutes 33.07 seconds.

## Six-way comparison context

The saved six-way comparison uses the same 924 validation examples and
evidence-length Oracle for all Qwen rows. Under those shared benchmark
conditions:

- Phase 2D has the highest macro-F1, 0.22994524079282935;
- Phase 2D has the highest weighted F1, 0.3644656337102369;
- Phase 2D has the highest balanced accuracy, 0.2391812745015638;
- Phase 2 numeric SFT has the highest accuracy, 0.4318181818181818;
- Phase 2B-A has the highest mean joined retrieval F1,
  0.28646775432900434;
- Phase 2B-B has the highest available top-2 accuracy,
  0.7056277056277056.

Phase 2C--2D is the clean prompt-only pair. Earlier cross-phase comparisons
remain multiply confounded by checkpoint family, model interface, target
formulation, loss, or prompt changes. Phase 1 and Phase 2 also lack comparable
five-class scores for top-2.

## Interpretation and limitations

The exact-count prompt produces a modest single-seed classification gain over
the qualitative prompt at the macro-F1-selected checkpoint. The improvement is
not universal: top-2 accuracy and downstream retrieval F1 are lower, class 40
F1 is slightly lower, and class 10 remains completely unrecognized. Accuracy
also remains 0.08549783549783546 below the class-160 majority baseline.

The data imbalance is a central limitation: class 160 supplies 45.45% of
validation and class 10 only 1.41%. The shift of 51 additional predictions to
class 160 improves class-160 recall and weighted metrics but does not establish
better behavior for rare classes.

This is one deterministic seed. It does not estimate variance across initial
conditions. The same validation split is used for checkpoint selection and
reported performance, and no untouched QASPER test result exists. The
Phase 2C--2D comparison is a valid controlled observation for seed 42, but it
does not establish statistical significance, general benefit of numeric
prompt wording, or held-out generalization.

Previous Logistic Regression and MLP router results used the old
retrieval-F1 Oracle. Their classification metrics are not directly comparable
with these evidence-length-Oracle results.

## Artifact inventory and integrity

Configuration and summary:

- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/configuration/experiment.json`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/configuration/preflight_manifest.json`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/final_summary.json`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/integrity/selected_checkpoint_transfer_verification.json`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/integrity/final_integrity_audit.json`

Run records under
`outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/runs/qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1/`:

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

Canonical validation artifacts:

- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/validation/predictions.jsonl`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/validation/raw_outputs.jsonl`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/validation/parsed_predictions.jsonl`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/validation/invalid_outputs.jsonl`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/validation/runtime_summary.json`

Classification and retrieval:

- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/classification/metrics.json`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/classification/confusion_matrix.csv`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/classification/predicted_vs_oracle.svg`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/retrieval/results.jsonl`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/retrieval/runtime_segments.jsonl`
- `outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/retrieval/summary.json`

Comparison and code:

- `outputs/qwen_phase2d_comparison_evidence_length_oracle/six_way_comparison.json`
- `qwen_phase2d_sequence_classifier.py`
- `qwen_phase2d_posttraining.py`
- `tests/test_qwen_phase2d_sequence_classifier.py`
- `tests/test_qwen_phase2d_posttraining.py`
- `docs/QWEN_PHASE2D_RESULTS.md`

Absolute `/dev/shm/...` paths in the immutable training records preserve the
original ephemeral-host locations. Their local equivalents are under the
repository-relative tree above.

The checkpoint transfer integrity status is `passed`. Source Vast instance:
46617164. The reconstructed remote/local checkpoint archive has SHA-256
`2dd4d23ff77179e1b33e522829cb2fdd6dd12684500a2158cc95f5f79a242a56`,
size 2,886,773,596 bytes, and 28 transfer chunks. The independently transferred
metadata archive has SHA-256
`178a5ed25f6fd62270e1ba1814811d352f9d6d2073303d7cb2cb0a0e036fa4f8`.
All nine extracted checkpoint files match their recorded remote SHA-256
values. The independent final audit passed 73/73 artifact assertions,
including TensorBoard, Qdrant, prediction, metric, retrieval, runtime, and
Phase 2C non-mutation checks. The focused Phase 2--2D regression suite passed
102/102 tests.

## Exact reproduction commands

The original CUDA-host execution form, from
`/workspace/thesis-granularity-router`, is:

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

Verify the deployed script before training:

```bash
sha256sum /dev/shm/phase2d_code/qwen_phase2d_sequence_classifier.py
```

The required result is
`99ba4f9a57b5547e776d81d2c7c94aece2670a9b5ccaf348c8a1fbeb0baa5efc`.
The recorded `40f79e1` environment value identifies only the base HEAD, not a
commit containing the uncommitted Phase 2D file.

After transferring the full output tree and selected checkpoint to the local
project, run unchanged retrieval against the existing Qdrant service at
`127.0.0.1:6334`:

```powershell
$root = "outputs\qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle"
$run = "qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1"

.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py evaluate-retrieval --output-root $root --run-id $run
.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py compare --phase2d-summary "$root\final_summary.json" --output outputs\qwen_phase2d_comparison_evidence_length_oracle\six_way_comparison.json
```

Protocol regression tests can be repeated with:

```powershell
.\.venv-qwen\Scripts\python.exe -m pytest -q tests\test_qwen_phase2c_sequence_classifier.py tests\test_qwen_phase2c_posttraining.py tests\test_qwen_phase2d_sequence_classifier.py tests\test_qwen_phase2d_posttraining.py
```

Reproduction must use the preserved Oracle records and existing Qdrant
collections. It must not recreate, re-index, or mutate any collection.
