# Granularity Routers

## Scope and leakage boundary

The legacy embedding-router path in `granularity_router.py` trains a classifier
from persisted `RouterDataset` records. Its only inference feature is the
existing question embedding. Evidence text, evidence vectors, per-granularity
retrieval metrics, Oracle scores, and label tie-break metadata are never
included in the feature matrix. The Qwen path is documented separately below:
it receives the fixed instruction and original question text, not the
1,536-dimensional embedding.

The stored `router_target_granularity` level is mapped to the fixed token classes:

```text
1 -> 10, 2 -> 20, 3 -> 40, 4 -> 80, 5 -> 160
```

Training loads only QASPER `train` and `validation`. Preprocessing is fitted on train only. Validation selects hyperparameters. The `test` oracle split is not read unless `--evaluate-test` is explicitly supplied for a final evaluation.

The loader rejects:

- question IDs appearing in more than one split;
- document IDs appearing in more than one split;
- duplicate questions for one evaluation configuration;
- mixed oracle configuration hashes unless one is selected explicitly;
- inconsistent embedding dimensions, embedding-model identities, or label versions;
- missing or non-finite question vectors.

## Models and selection

The trainer always reports:

1. a deterministic majority-class baseline fitted from train labels;
2. a multinomial logistic-regression baseline, implemented as a five-output linear softmax model.

Logistic learning rate and weight decay are selected by validation macro F1, then validation accuracy. A configurable one-hidden-layer MLP is evaluated only with `--enable-mlp`. It becomes the primary model only when its validation macro-F1 improvement over logistic regression is at least `--mlp-min-improvement` (default `0.01`). Otherwise logistic regression remains primary.

All neural optimization uses a recorded seed and deterministic PyTorch operations. Standardization is enabled by default and stores the train-derived mean and scale in the model artifact.

## Reported metrics

Every evaluated model reports:

- accuracy;
- macro and weighted F1 over the five fixed classes;
- balanced accuracy, defined as mean recall over reference classes with support;
- top-2 accuracy;
- per-class precision, recall, F1, and support;
- a 5×5 confusion matrix ordered as `10,20,40,80,160`;
- class distribution;
- deltas relative to the majority baseline.

Validation metrics are always reported. Test metrics are absent unless final evaluation is explicitly requested.

## Persisted artifacts

The output directory contains:

- `router_model.pt`: model states, train-fitted preprocessing arrays, majority class, label mapping, embedding identity/dimension, oracle configuration hash and label version, random seed, training configuration, and Git revision;
- `metadata.json`: human-readable artifact metadata without model tensors;
- `training_report.json`: dataset distributions, validation metrics, candidate configurations, majority comparisons, MLP justification, and optional final test metrics.

## PowerShell commands

The oracle dataset must contain the same configuration hash for both train and validation. When several hashes exist, select one explicitly:

```powershell
.\.venv\Scripts\python.exe granularity_router.py train `
  --evaluation-config-hash <hash> `
  --output-dir models\granularity_router `
  --seed 42
```

Enable the optional MLP comparison:

```powershell
.\.venv\Scripts\python.exe granularity_router.py train `
  --evaluation-config-hash <hash> `
  --enable-mlp `
  --mlp-hidden-sizes 64,128 `
  --mlp-dropouts 0.1,0.2 `
  --mlp-min-improvement 0.01
```

Final test evaluation must be a deliberate separate run:

```powershell
.\.venv\Scripts\python.exe granularity_router.py train `
  --evaluation-config-hash <frozen-hash> `
  --evaluate-test
```

Predict from `PaperQuestion` embeddings without accessing oracle/evidence payloads:

```powershell
.\.venv\Scripts\python.exe granularity_router.py predict `
  --model models\granularity_router\router_model.pt `
  --split validation `
  --limit 100 `
  --output-jsonl outputs\router_predictions.jsonl
```

## Current results and data readiness

### Pretrained Qwen3.5-0.8B Phase 1 result

The Phase 1 Qwen router is separate from the embedding routers. Logistic
Regression and MLP consume 1,536-dimensional question embeddings;
`Qwen/Qwen3.5-0.8B` consumes only the fixed instruction and original question
text. No evidence, answer, paper content, retrieval output, embedding,
metadata, or handcrafted feature is supplied.

The evidence-length Oracle strips, exact-deduplicates, sorts, and newline-joins
all evidence spans, counts GPT-2 tokens, and selects the nearest class with
smaller-candidate midpoint ties. Validation support for 10/20/40/80/160 is
13/81/178/232/420, making class 160 the 45.45% majority. Qwen predictions were
767/40/116/0/1. Its accuracy/macro-F1/weighted F1 was
0.040043/0.049046/0.032613; all 924 outputs were valid. Unchanged same-paper
top-five retrieval achieved mean joined retrieval F1 0.239109 with 100%
coverage. Top-2 accuracy is unavailable.

These classification results are not directly comparable with earlier
Logistic/MLP results because those use the old retrieval-F1 Oracle. See
`docs/QWEN_PHASE1_RESULTS.md` and the standalone experiment report.

### Fine-tuned Qwen3.5-0.8B Phase 2 result

Phase 2 is complete for the exact same `Qwen/Qwen3.5-0.8B` model and revision
`2fc06364715b967f1860aea9cf38778875588b17` used in Phase 1. It is
full-parameter supervised fine-tuning, not a new
classification head: all 852,985,920 parameters were trainable, and training
executed 213 optimizer updates. No LoRA, QLoRA, adapter, prompt-tuning, quantization,
or added input feature was used. Cross-entropy loss was applied only to the
assistant target tokens representing one of `10`, `20`, `40`, `80`, or `160`;
the fixed instruction and question tokens were masked from the loss.

The frozen evidence-length Oracle contains 2,245 train questions from 845
papers and 924 validation questions from 277 papers. Its validation
distribution is 10: 13, 20: 81, 40: 178, 80: 232, and 160: 420. Thus 160 is
45.45% of validation. The selected Phase 2 checkpoint predicts only 80 and
160: 10: 0, 20: 0, 40: 0, 80: 149, and 160: 775. This is a different collapse
from Phase 1's strong preference for 10, but remains a severe class-collapse
failure: recall and F1 are zero for the three smallest classes.

| Validation metric | Phase 1 pretrained | Phase 2 fine-tuned |
|---|---:|---:|
| Accuracy | 0.04004329004329004 | 0.4318181818181818 |
| Macro-F1 | 0.049045932422555796 | 0.16502267760462996 |
| Weighted F1 | 0.032612933907418644 | 0.32805741427623947 |
| Balanced accuracy | 0.23369399361908724 | 0.20697865353037764 |
| Mean joined retrieval F1 | 0.23910868506493507 | 0.22658488852813854 |

This Phase 1/Phase 2 comparison is valid because both Qwen runs use the same
preserved split and same evidence-length Oracle. It does not make the old
Logistic/MLP classification results comparable: those models were evaluated
against the retrieval-F1 Oracle. Phase 2 improves accuracy, macro-F1, and
weighted F1 over Phase 1, but remains below the 160-majority baseline on
accuracy (0.431818 versus 0.454545), has lower balanced accuracy than Phase 1,
and has lower downstream mean joined retrieval F1. Classification metrics
measure Oracle-label prediction; joined retrieval F1 measures evidence-token
overlap after retrieval and is a different outcome.

All 924 generated outputs were valid. Top-2 accuracy is unavailable because
deterministic generated text does not provide comparable scores for all five
classes. Unchanged source-paper-restricted retrieval used `top_k=5`,
`text-embedding-3-small`, cosine similarity, existing Qdrant collections, and
the existing joined token-level F1. Coverage was 924/924 (100%); mean and
median joined retrieval F1 were 0.22658488852813854 and
0.19615549999999998.

Training used Python 3.10.7 in the separate `.venv-qwen`, PyTorch
`2.8.0+cu128`, Transformers `5.15.0.dev0` at commit
`2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`, one
`NVIDIA A100-SXM4-40GB`, CUDA, `torch.bfloat16`, no quantization, seed 42,
three epochs, per-device batch size 4, gradient accumulation 8, effective
batch size 32, learning rate 2e-5, weight decay 0.01, cosine scheduling with
5% warmup, and gradient clipping at 1.0. The original `.venv` and all Phase 1
artifacts remained unchanged.

The selected checkpoint is `step-000213`, chosen by validation macro-F1 after
the third epoch. Training, including epoch validation and checkpointing, took
2,107.3131887838244 seconds. Reloaded final validation took 299.0685129035264
seconds including model loading; isolated generation took 296.8330853600055
seconds, with mean/median generation latency 0.32010594106710705/
0.3152373321354389 seconds per question. Retrieval took 178.12831589998677
seconds. Peak allocated training GPU memory was 10.660949230194092 GiB.

The authoritative summary is
`outputs/qwen_finetuned_router_evidence_length_oracle/final_summary.json`.
The standalone reports are `docs/QWEN_PHASE2_RESULTS.md` and
`reports/qwen_finetuned_router_evidence_length_oracle/experiment_report.md`.
The run configuration, checkpoint manifest, structured histories, TensorBoard
audit, checkpoint verification, predictions, classification artifacts,
retrieval records, and integrity audit remain under
`outputs/qwen_finetuned_router_evidence_length_oracle/`. The final integrity
audit passed, including frozen-order prediction checks, exact metric
recomputation, retrieval coverage, checkpoint selection, TensorBoard agreement,
deterministic checkpoint generation, exact selected-checkpoint archive hashes,
and unchanged Phase 1 source hashes.

Exact commands for the recorded run, from the project root, are:

```bash
.venv-qwen/bin/python qwen_phase2.py inspect-data
.venv-qwen/bin/python qwen_phase2.py train --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.venv-qwen/bin/python qwen_phase2.py audit-tensorboard --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.venv-qwen/bin/python qwen_phase2.py verify-checkpoint --run-id qwen-phase2-full-parameter-20260802-seed42-v2 --checkpoint outputs/qwen_finetuned_router_evidence_length_oracle/runs/qwen-phase2-full-parameter-20260802-seed42-v2/checkpoints/step-000213
.venv-qwen/bin/python qwen_phase2.py final-validation --run-id qwen-phase2-full-parameter-20260802-seed42-v2
```

The unchanged retrieval evaluation was run against the existing local Qdrant
service from Windows:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2.py evaluate-retrieval --run-id qwen-phase2-full-parameter-20260802-seed42-v2
.\.venv-qwen\Scripts\python.exe qwen_phase2.py audit-final --run-id qwen-phase2-full-parameter-20260802-seed42-v2
```

### Qwen3.5-0.8B Phase 2B restricted-alias results

Phase 2B completed two separately named full-parameter runs without changing
the frozen Phase 1 or Phase 2 artifacts. Both use the same
`Qwen/Qwen3.5-0.8B` revision, preserved 2,245/924 train/validation questions,
evidence-length Oracle, seed, optimizer schedule, and question-only leakage
boundary. The fixed Phase 2B instruction maps one-token aliases `1` through
`5` to chunk sizes 10, 20, 40, 80, and 160. Training applies cross-entropy to
the five alias logits at the first assistant answer position, and inference is
deterministic argmax over those same directly comparable logits. It is not
unrestricted generation and does not use the Phase 1/2 free-text parser.

Phase 2B-A is unweighted. Phase 2B-B changes only the loss weights within the
Phase 2B pair, using effective-number weights with `beta=0.999`, calculated
from the preserved training labels. Both train all 852,985,920 parameters for
three epochs and 213 optimizer steps on one A100 in BF16 without quantization.

| Validation metric | Phase 1 | Phase 2 | Phase 2B-A unweighted | Phase 2B-B class-balanced |
|---|---:|---:|---:|---:|
| Accuracy | 0.04004329004329004 | **0.4318181818181818** | 0.35064935064935066 | 0.37012987012987014 |
| Macro-F1 | 0.049045932422555796 | 0.16502267760462996 | **0.20922603632601472** | 0.16836616836616836 |
| Weighted F1 | 0.032612933907418644 | 0.32805741427623947 | **0.3406050804511769** | 0.3142183142183142 |
| Balanced accuracy | 0.23369399361908724 | 0.20697865353037764 | **0.2383201416948027** | 0.20607553366174058 |
| Top-2 accuracy | unavailable | unavailable | 0.6071428571428571 | 0.7056277056277056 |
| Mean joined retrieval F1 | 0.23910868506493507 | 0.22658488852813854 | **0.28646775432900434** | 0.24962774025974027 |

All four runs have 924/924 valid predictions and 100% retrieval coverage.
The Oracle distribution is 13/81/178/232/420 for classes
10/20/40/80/160. Phase 2B-A predicts 0/0/427/189/308, while Phase 2B-B
predicts 0/0/0/434/490. Thus neither variant learns to select 10 or 20. The
class-balanced variant is worse than the unweighted variant on the primary
macro-F1, weighted F1, balanced accuracy, and mean joined retrieval F1; this
is a negative result for this exact weighting configuration, not evidence
against every possible imbalance treatment. Neither Phase 2B variant exceeds
the class-160 majority accuracy baseline of 0.45454545454545453, although both
exceed its macro-F1 of 0.125.

Unchanged source-paper-restricted retrieval uses the predicted canonical
chunk size, `top_k=5`, existing `text-embedding-3-small` vectors, cosine
ranking, and joined token-level retrieval F1. Phase 2B-A mean/median joined
retrieval F1 is 0.28646775432900434/0.2748425; Phase 2B-B is
0.24962774025974027/0.223194. These downstream retrieval values are not
classification metrics.

Phase 2B-A selected epoch 3 (`step-000213`); Phase 2B-B selected epoch 2
(`step-000142`), in both cases by validation macro-F1. Training wall times
were 1308.664808139205 and 1306.7509042322636 seconds. Saved final inference
wall times were 35.81447528861463 and 39.01004763878882 seconds. Saved local
retrieval wall times were 178.27286399999866 and 377.0999227000284 seconds,
respectively; the latter overlapped a checkpoint archive transfer, so their
difference must not be interpreted as a method-speed effect.

The four Qwen rows share the new Oracle and validation questions, but Phase
2B deliberately changes the prompt output schema, target encoding, and
decision rule. Therefore the Phase 2-to-2B differences cannot be attributed
only to alias symbols. The Phase 2B-A/B weighting comparison is the closest
controlled pair, but it is a single-seed result selected and reported on the
same validation split; no test split was evaluated. Old-Oracle Logistic
Regression and MLP classification results remain not directly comparable.

See `docs/QWEN_PHASE2B_RESULTS.md`, the two standalone reports under
`reports/qwen_phase2b_alias_*_evidence_length_oracle/`, and the authoritative
summaries under `outputs/qwen_phase2b_alias_*_evidence_length_oracle/`. The
four-way machine-readable comparison is
`outputs/qwen_phase2b_comparison_evidence_length_oracle/four_way_comparison.json`.
Manual post-transfer checksum dry-runs verified the copied trees; the two stale
Phase 2B-B preflight timestamp copies were replaced and the targeted rerun had
no differences. The selected A/B checkpoints contain 11 files totaling
4,735,895,574/4,735,895,530 bytes, retrieval replay revalidated 924/924 rows
for each run, and all Qdrant collection counts were unchanged. No standalone
Phase 2B hash-inventory artifact was saved.

### Qwen3.5-0.8B-Base Phase 2C sequence-classifier result

Phase 2C implements the supervisor-motivated direct-classification path: use
the exact Base checkpoint with a conventional five-logit classifier, and use
the revised semantic context-length prompt instead of training the chat model
to generate literal chunk-size or alias tokens. The exact checkpoint is
`Qwen/Qwen3.5-0.8B-Base`, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

The fixed instruction is:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = very short context, 2 = short context, 3 = medium context, 4 = long context, 5 = very long context. Return only the number

Its SHA-256 is
`9e879535647c2bfcd3627d0d65f84c36a1bf442ed95bb5b07029c878ca990de7`.
The plain input is `{instruction}\n\nQuestion: {original_question_text}`. There
is no chat template, answer-generation target, unrestricted or restricted
generation, assistant-token loss, or output parser. No evidence, evidence
length, answer, paper text, retrieved chunk, retrieval score, metadata,
embedding, or handcrafted feature is supplied.

`AutoModelForSequenceClassification` adds a bias-free 5×1024 `score.weight`
head. Class IDs 0/1/2/3/4 map to chunk sizes 10/20/40/80/160. Training uses
uniform five-class cross-entropy and deterministic five-logit argmax at
inference. All 852,991,040 parameters were marked trainable for three epochs
and 213 optimizer updates. The text-only gradient audit found gradients in the
language backbone and classifier head (752,398,144 parameters); the composite
checkpoint's vision tower received no gradient (100,592,896 parameters), as
expected because no image input entered this text-only path.

The frozen evidence-length Oracle and preserved questions are unchanged:
2,245 train questions from 845 papers and 924 validation questions from 277
papers. Validation support is 13/81/178/232/420 for classes
10/20/40/80/160. The selected epoch-3 checkpoint predicts
0/20/224/374/306.

| Metric | Phase 2C value |
|---|---:|
| Accuracy | 0.34523809523809523 |
| Macro-F1 | 0.21763191244497584 |
| Weighted F1 | 0.3435657773957275 |
| Balanced accuracy | 0.22993634120458348 |
| Top-2 accuracy | 0.6428571428571429 |
| Valid outputs | 924/924 |
| Mean joined retrieval F1 | 0.27914719588744585 |
| Median joined retrieval F1 | 0.2607245 |

Per-class precision/recall/F1 is 0/0/0 for class 10;
0.05/0.012345679012345678/0.019801980198019802 for class 20;
0.29017857142857145/0.3651685393258427/0.3233830845771144 for class 40;
0.23529411764705882/0.3793103448275862/0.29042904290429045 for class 80;
and 0.5392156862745098/0.39285714285714285/0.45454545454545453 for class
160. Class 10 has zero recall. Class 20 is also very weak: only 1 of 81 true
examples is correctly classified, and only 1 of 20 predictions of class 20 is
correct.

The final confusion matrix has Oracle rows and predicted columns ordered 10,
20, 40, 80, 160:

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 5 | 7 | 1 |
| 20 | 0 | 1 | 19 | 39 | 22 |
| 40 | 0 | 4 | 65 | 66 | 43 |
| 80 | 0 | 5 | 64 | 88 | 75 |
| 160 | 0 | 10 | 71 | 174 | 165 |

The selected checkpoint is epoch 3, `step-000213`, because its validation
macro-F1 is the highest of the three epoch checkpoints. Training took
1276.56244828552 seconds. Reloaded final inference took 33.99719780869782
seconds after 2.5492455568164587 seconds of model loading; mean/median
inference time was 0.036642024783393394/0.03524067858234048 seconds per
question. Training peak allocated/reserved GPU memory was
8.96875286102295/9.517578125 GiB. Retrieval took 134.9306207000045 seconds;
known training, final-validation, and retrieval time was
1448.0395123510389 seconds.

Unchanged retrieval covers 924/924 questions and retains source-paper
filtering, `top_k=5`, `text-embedding-3-small`, cosine ranking, canonical chunk
ordering and concatenation, GPT-2 tokenization, and joined token-level F1.
Classification metrics and joined retrieval F1 remain different outcomes.

At the completion of Phase 2C, across the five saved evidence-length-Oracle
Qwen runs, Phase 2C had the best macro-F1 at 0.21763191244497584.
Numeric-target Phase 2 retained the best
accuracy at 0.4318181818181818, and Phase 2B-A retains the best mean joined
retrieval F1 at 0.28646775432900434 versus Phase 2C's
0.27914719588744585. Phase 2C simultaneously changes the checkpoint family,
classifier formulation, and prompt, so it is benchmark-comparable under the
same data and Oracle but is not a clean causal architecture ablation. It is a
single seed selected and reported on validation; controlled one-factor
follow-ups and multiple seeds are required before stronger claims. Old-Oracle
Logistic Regression and MLP classification results remain not directly
comparable.

See `docs/QWEN_PHASE2C_RESULTS.md`,
`reports/qwen_phase2c_sequence_classifier_evidence_length_oracle/experiment_report.md`,
`outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle/final_summary.json`,
and
`outputs/qwen_phase2c_comparison_evidence_length_oracle/five_way_comparison.json`.

### Legacy live-collection readiness snapshot

As checked on 2026-06-22, the local legacy `RouterDataset` collection contains
two validation records, both targeting the 20-token class, under two different
evaluation-configuration hashes, and no train records. This snapshot concerns
the live old-Oracle embedding-router command only. It does not describe the
preserved 2,245/924 Qwen evidence-length-Oracle files used by completed Phases
1, 2, 2B, 2C, and 2D.

### Qwen3.5-0.8B-Base Phase 2D token-count-prompt result

Phase 2D repeats the Phase 2C five-logit sequence-classifier experiment as a
controlled prompt-only, single-seed ablation. The qualitative mapping in the
Phase 2C instruction is replaced by the exact candidate token counts:

> You are a router for a retrieval-augmented generation system. Based only on the question, select the option representing the context size most suitable for retrieving the evidence required to answer it. Choose exactly one value from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, 4 = 80 tokens, 5 = 160 tokens. Return only the number

The instruction SHA-256 is
`b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5`.
The saved six-way protocol audit passed and confirms that the Phase 2C to
Phase 2D semantic change is only the mapping from qualitative descriptions to
10/20/40/80/160 token counts. The Base model and revision, sequence-classifier
architecture, fresh seed-42 head initialization, frozen questions and Oracle
hashes, input template, uniform loss, hyperparameters, checkpoint selection,
and downstream retrieval configuration are unchanged. Run identifiers,
timestamps, script hashes, fingerprints, output paths, and prompt-caused
sequence lengths necessarily differ.

The Phase 2D experiment fingerprint is
`dad60bd9a0530865110c2310f62a896c73350fa383c7812d5c6733e376bc377d`.

The model is `Qwen/Qwen3.5-0.8B-Base` at revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, loaded with
`AutoModelForSequenceClassification`. Five logits map class IDs 0--4 directly
to 10/20/40/80/160. Input remains only the fixed instruction, two newlines,
`Question: `, and the original question. There is no chat template,
generation, parser, evidence, answer, paper text, retrieved chunk, retrieval
score, embedding, metadata, or handcrafted feature. The new prompt produces
95--121-token train inputs and 96--124-token validation inputs; all remain
below the frozen maximum of 128 and none is truncated.

Training uses the same 2,245 examples from 845 papers and validation uses the
same 924 examples from 277 papers. The validation Oracle distribution is
13/81/178/232/420 for 10/20/40/80/160; class 160 therefore remains the
420/924 = 45.45% majority and class 10 has only 13 examples. The training
distribution is 55/267/586/687/650. All 852,991,040 parameters were marked
trainable under uniform cross-entropy for three epochs and 213 updates. The
text backbone and classifier head received gradients; the 100,592,896 vision
parameters received none on this text-only path. Epoch 3, `step-000213`, was
selected by validation macro-F1.

| Metric | Phase 2C | Phase 2D | Phase 2D minus Phase 2C |
|---|---:|---:|---:|
| Accuracy | 0.34523809523809523 | 0.36904761904761907 | +0.023809523809523836 |
| Macro-F1 | 0.21763191244497584 | 0.22994524079282935 | +0.012313328347853508 |
| Weighted F1 | 0.3435657773957275 | 0.3644656337102369 | +0.020899856314509357 |
| Balanced accuracy | 0.22993634120458348 | 0.2391812745015638 | +0.009244933296980312 |
| Top-2 accuracy | 0.6428571428571429 | 0.6341991341991342 | -0.008658008658008698 |
| Mean joined retrieval F1 | 0.27914719588744585 | 0.2767166677489178 | -0.0024305281385280653 |
| Median joined retrieval F1 | 0.2607245 | 0.2558975 | -0.004827000000000026 |

All 924 Phase 2D classifier outputs are valid. The prediction distribution is
0/16/219/332/357. The prompt change improves the four reported top-1
classification aggregates relative to Phase 2C, but it does not solve the
minority-class problem: class 10 is never predicted and has zero precision,
recall, and F1. Class 20 has precision 0.125, recall
0.024691358024691357, and F1 0.041237113402061855, corresponding to only 2
correct examples among its 81 Oracle examples. Per-class precision/recall/F1
for 40, 80, and 160 is respectively
0.2876712328767123/0.3539325842696629/0.31738035264483627,
0.25/0.3577586206896552/0.29432624113475175, and
0.5406162464985994/0.4595238095238095/0.49678249678249675.

The final confusion matrix has Oracle rows and predicted columns ordered 10,
20, 40, 80, 160:

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 3 | 7 | 3 |
| 20 | 0 | 2 | 21 | 33 | 25 |
| 40 | 0 | 4 | 63 | 60 | 51 |
| 80 | 0 | 4 | 60 | 83 | 85 |
| 160 | 0 | 6 | 72 | 149 | 193 |

Phase 2D accuracy remains below the class-160 majority baseline
0.45454545454545453; macro-F1 is above the majority baseline 0.125. Across
the six saved new-Oracle Qwen runs, Phase 2D has the highest macro-F1,
weighted F1, and balanced accuracy. Numeric-target Phase 2 retains the highest
accuracy, 0.4318181818181818, and Phase 2B-A retains the highest mean joined
retrieval F1, 0.28646775432900434.

The unchanged source-paper-restricted `top_k=5` retrieval covers 924/924
questions. Mean/median joined retrieval F1 is
0.2767166677489178/0.2558975. Classification metrics measure prediction of the
evidence-length Oracle label; joined retrieval F1 measures GPT-2-token overlap
after the predicted class controls downstream retrieval. They are distinct
outcomes, which is why improved Phase 2D classification does not imply improved
retrieval relative to Phase 2C.

Training took 1224.5802961867303 seconds. Selected-checkpoint loading and
isolated final inference took 2.7541816290467978 and 34.72815803065896
seconds; mean/median inference time was
0.0374373855065248/0.03656412195414305 seconds per question. Retrieval took
151.0063940999098 seconds, and known training plus final validation and
retrieval time is 1413.0690299463458 seconds. Peak allocated/reserved training
GPU memory was 9.0316162109375/9.6015625 GiB.

This is a clean Phase 2C-to-Phase 2D prompt comparison, but it remains one
seed. The same validation set selects the checkpoint and supplies the reported
metrics, no QASPER test result is claimed, and no run-to-run variance is
available. Earlier cross-phase comparisons remain multiply confounded, while
old-Oracle Logistic Regression and MLP classification results remain not
directly comparable.

Authoritative artifacts are under
`outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle/`:
`final_summary.json`; experiment/preflight configuration; the run
configuration, dataset manifest, histories, checkpoint manifest and selected
checkpoint; canonical/raw/parsed/invalid validation records and runtime;
classification metrics, confusion matrix and histogram; retrieval records,
runtime segments and summary; and
`integrity/selected_checkpoint_transfer_verification.json`. The independent
73-assertion final audit and recorded 102-test focused regression result are in
`integrity/final_integrity_audit.json`. The machine-readable prompt-only audit
and six-run comparison are in
`outputs/qwen_phase2d_comparison_evidence_length_oracle/six_way_comparison.json`.
The transfer audit verifies the 2,886,773,596-byte checkpoint archive at
SHA-256
`2dd4d23ff77179e1b33e522829cb2fdd6dd12684500a2158cc95f5f79a242a56`
and all nine extracted files against the remote source.

Recorded CUDA-host commands:

```bash
.venv-qwen/bin/python qwen_phase2d_sequence_classifier.py --output-root /dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle inspect
.venv-qwen/bin/python qwen_phase2d_sequence_classifier.py --output-root /dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle train --mode full --run-id qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1
.venv-qwen/bin/python qwen_phase2d_sequence_classifier.py --output-root /dev/shm/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle final-validation --run-id qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1
```

Recorded local read-only Qdrant and comparison commands:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py evaluate-retrieval --run-id qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-20260808-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2d_posttraining.py compare --output outputs\qwen_phase2d_comparison_evidence_length_oracle\six_way_comparison.json
```
