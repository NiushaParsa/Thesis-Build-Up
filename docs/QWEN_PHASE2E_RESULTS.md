# Qwen Phase 2E Results: Learning-Rate Grid with Five Epochs

## Status and scope

Phase 2E completed the classification-training grid, locked its winning
checkpoint, and then completed unchanged downstream retrieval for that winner.
It is a controlled hyperparameter study built on the Phase 2D Base-model
sequence classifier. The three fresh runs use learning rates `5e-6`, `1e-5`,
and `2e-5`, run for five epochs with seed 42, and retain the exact Phase 2D
token-count prompt and all other training inputs.

The global classification winner is `lr5e-6`, epoch 4,
`step-000284`. Its validation macro-F1 is
`0.22777929657889012`. This does not exceed the Phase 2D macro-F1 of
`0.22994524079282935`.

The 924 official validation examples have been observed repeatedly and are
used here to select one of 15 learning-rate-by-epoch candidates. They must be
described as a development/model-selection set, not as an unbiased final test
of generalization.

## Research question

Phase 2E asks whether a lower learning rate and a longer, five-epoch schedule
improve the Phase 2D sequence-classification router. It changes only:

- experiment and artifact identity;
- learning rate, selected from the predeclared grid `5e-6`, `1e-5`, `2e-5`;
- epochs, from three to five;
- the derived cosine-schedule horizon and warmup count.

It does not change the model, prompt, tokenizer, examples, Oracle labels,
five-class mapping, loss weighting, input features, batch configuration,
seed, checkpoint metric, or downstream retrieval protocol.

Phase 2E is not simply Phase 2D with two additional epochs. Phase 2D used a
213-step cosine schedule with 11 warmup steps, whereas each Phase 2E trial
uses a 355-step cosine schedule with 18 warmup steps. Consequently, even the
Phase 2E `2e-5` trajectory differs from Phase 2D from the first optimizer
step.

## Model, input, and labels

| Item | Frozen value |
|---|---|
| Model | `Qwen/Qwen3.5-0.8B-Base` |
| Revision | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Architecture | `AutoModelForSequenceClassification` |
| Problem | Five-way single-label classification |
| Canonical classes | 10, 20, 40, 80, 160 |
| Class-ID mapping | 0→10, 1→20, 2→40, 3→80, 4→160 |
| Objective | Uniform five-class cross-entropy |
| Prompt SHA-256 | `b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5` |
| Input template | `{instruction}\n\nQuestion: {original_question_text}` |

The fixed instruction is:

> You are a router for a retrieval-augmented generation system. Based only on
> the question, select the option representing the context size most suitable
> for retrieving the evidence required to answer it. Choose exactly one value
> from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, 4 = 80 tokens, 5 = 160
> tokens. Return only the number

The model receives only this instruction and the original question text. It
does not receive evidence, evidence length, answers, paper text, retrieved
chunks, retrieval scores, metadata, handcrafted features, a chat template, or
assistant target tokens. Although the instruction uses the aliases 1–5, the
model is trained as a sequence classifier and produces five comparable head
logits rather than generated conversational text.

## Oracle and preserved data

The evidence-length Oracle measures the GPT-2 token length of the complete
deduplicated ground-truth evidence and selects the closest value among 10, 20,
40, 80, and 160. Exact midpoint ties use the smaller class; values below 10
map to 10 and values above 160 map to 160. This Oracle is independent of
retrieval F1, embeddings, cosine similarity, retrieved chunks, and router
performance.

| Split | Examples | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation/development | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

The validation class percentages are approximately 1.41%, 8.77%, 19.26%,
25.11%, and 45.45%, respectively. The strong imbalance, especially the rare
class 10 and frequent class 160, is important when interpreting accuracy and
macro-F1.

Data-identity hashes:

- train Oracle SHA-256: `64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`;
- validation Oracle SHA-256: `ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`.

## Training configuration

| Setting | Value |
|---|---|
| Learning-rate grid | `5e-6`, `1e-5`, `2e-5` |
| Epochs per trial | 5 |
| Optimizer steps per epoch | 71 |
| Total optimizer steps per trial | 355 |
| Validation/checkpoint steps | 71, 142, 213, 284, 355 |
| Per-device batch size | 4 |
| Gradient accumulation | 8 |
| Effective batch size | 32 |
| Maximum sequence length | 128 |
| Weight decay | 0.01 |
| Scheduler | Cosine |
| Warmup ratio / steps | 0.05 / 18 |
| Gradient clipping | 1.0 |
| Class weights | Uniform |
| Seed | 42 |
| Dtype | `torch.bfloat16` |
| Device | CUDA, NVIDIA A100-SXM4-40GB |
| Quantization | None |
| Validation/checkpoint frequency | End of each epoch |
| Early stopping | None; fixed five epochs |

Run identities are fixed per variant:

| Variant | Run ID |
|---|---|
| `lr5e-6` | `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr5e-6-5epochs-full-parameter-20260808-seed42-v1` |
| `lr1e-5` | `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr1e-5-5epochs-full-parameter-20260808-seed42-v1` |
| `lr2e-5` | `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr2e-5-5epochs-full-parameter-20260808-seed42-v1` |

Tokenizer preflight identified `Qwen2Tokenizer`, right padding,
`<|endoftext|>` as padding token with ID 248044, special tokens enabled, and
no truncation. Train sequence lengths range from 95 to 121 tokens with mean
`101.13363028953229`; validation lengths range from 96 to 124 with mean
`100.59740259740259`. No sequence exceeds the configured maximum.

## Predeclared selection rule

The experiment evaluates all 15 epoch checkpoints and selects exactly once by
the following lexicographic order:

1. higher validation macro-F1;
2. higher validation accuracy;
3. higher validation weighted F1;
4. higher validation balanced accuracy;
5. lower validation cross-entropy;
6. earlier optimizer step;
7. lower numeric learning rate only for an otherwise exact tie.

The selection artifact was locked at
`2026-08-08T20:26:31.564119+00:00`. Retrieval was not used for selection and
cannot revise the winner.

## Complete 15-candidate results

The predicted-distribution column is ordered 10/20/40/80/160. Top-2 accuracy
is valid because every checkpoint produces comparable five-class logits.

| LR | Epoch | Step | Validation CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 | Predicted 10/20/40/80/160 | Validation wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 5e-6 | 1 | 71 | 1.4160810518058347 | 0.30844155844155846 | 0.19524927027126598 | 0.29896932960463235 | 0.2272939545327509 | 0.6017316017316018 | 0/14/456/221/233 | 33.8498853109777 |
| 5e-6 | 2 | 142 | 1.4814272373269646 | 0.2619047619047619 | 0.15051563951039187 | 0.1790667704247578 | 0.21846274532313537 | 0.538961038961039 | 0/14/162/723/25 | 33.0731706880033 |
| 5e-6 | 3 | 213 | 1.4225975060875797 | 0.2857142857142857 | 0.18413217050572556 | 0.2729541950135127 | 0.22010663326561558 | 0.5714285714285714 | 0/4/450/309/161 | 32.74500731192529 |
| **5e-6** | **4** | **284** | **1.3759860497016412** | **0.3484848484848485** | **0.22777929657889012** | **0.3473258648868964** | **0.24232226137689133** | **0.6190476190476191** | **0/15/275/366/268** | **33.22264311462641** |
| 5e-6 | 5 | 355 | 1.3720230444685204 | 0.3484848484848485 | 0.22559313887729796 | 0.3484691654624007 | 0.23795324604507093 | 0.6309523809523809 | 0/14/289/328/293 | 33.61626974865794 |
| 1e-5 | 1 | 71 | 1.3604521274050594 | 0.39285714285714285 | 0.17892937509863063 | 0.3207834319072489 | 0.22323678939912955 | 0.6298701298701299 | 0/208/37/36/643 | 32.11344109848142 |
| 1e-5 | 2 | 142 | 1.3957898890301264 | 0.29978354978354976 | 0.16337588291006747 | 0.24666410097710725 | 0.21723795242263538 | 0.6515151515151515 | 0/13/42/753/116 | 32.49877715110779 |
| 1e-5 | 3 | 213 | 1.3921086592075629 | 0.33874458874458874 | 0.2121913036477682 | 0.33449363556478245 | 0.23582110347834226 | 0.6341991341991342 | 0/4/370/280/270 | 32.61650368757546 |
| 1e-5 | 4 | 284 | 1.458046925016296 | 0.3235930735930736 | 0.21540884371375907 | 0.32864368966292395 | 0.22681571407388273 | 0.6038961038961039 | 3/47/286/321/267 | 32.64094417728484 |
| 1e-5 | 5 | 355 | 1.4509322921951096 | 0.32792207792207795 | 0.21263800582609677 | 0.33163386759668284 | 0.22428790776717938 | 0.6071428571428571 | 2/40/300/292/290 | 33.21086246334016 |
| 2e-5 | 1 | 71 | 1.3393364873799412 | 0.4090909090909091 | 0.1672499143468602 | 0.3096967677990664 | 0.22668312119429257 | 0.6536796536796536 | 0/196/9/17/702 | 33.15667689591646 |
| 2e-5 | 2 | 142 | 1.355628962124581 | 0.32575757575757575 | 0.1728867102330036 | 0.2910400848572852 | 0.21147432020866092 | 0.7023809523809523 | 0/4/23/656/241 | 34.77005876787007 |
| 2e-5 | 3 | 213 | 1.5105030201214216 | 0.3463203463203463 | 0.21680431699734548 | 0.3387164676151271 | 0.24251581390803914 | 0.5909090909090909 | 1/6/382/251/284 | 33.07019843161106 |
| 2e-5 | 4 | 284 | 2.136673576388008 | 0.31277056277056275 | 0.2232188383733722 | 0.3222002147927793 | 0.23209435113606625 | 0.5584415584415584 | 10/92/238/334/250 | 33.84760375879705 |
| 2e-5 | 5 | 355 | 2.2110366181377725 | 0.3170995670995671 | 0.2252323080025679 | 0.3281524813999655 | 0.23285430021449655 | 0.5562770562770563 | 12/88/254/309/261 | 33.07202775031328 |

The winner leads the runner-up, `5e-6` epoch 5, by
`0.0021861577015921674` macro-F1. The best checkpoint within `1e-5` is epoch
4 with macro-F1 `0.21540884371375907`; the best within `2e-5` is epoch 5 with
macro-F1 `0.2252323080025679`.

The selection metric matters. The grid's highest accuracy is
`0.4090909090909091` at `2e-5`, epoch 1; highest top-2 is
`0.7023809523809523` at `2e-5`, epoch 2; and lowest cross-entropy is
`1.3393364873799412` at `2e-5`, epoch 1. None of those rows has the highest
macro-F1.

## Locked-winner classification result

| Metric | Value |
|---|---:|
| Learning rate | 0.000005 |
| Selected epoch / checkpoint | 4 / `step-000284` |
| Validation CE | 1.3759860497016412 |
| Accuracy | 0.3484848484848485 = 322/924 |
| Macro-F1 | 0.22777929657889012 |
| Weighted F1 | 0.3473258648868964 |
| Balanced accuracy | 0.24232226137689133 |
| Top-2 accuracy | 0.6190476190476191 = 572/924 |
| Valid outputs | 924/924 = 100% |
| Invalid outputs | 0 |
| Majority-class accuracy baseline | 0.45454545454545453 |
| Majority-class macro-F1 baseline | 0.125 |

The classifier is `0.10606060606060602` below the class-160 majority baseline
on accuracy and `0.10277929657889012` above it on macro-F1. The majority
baseline values are classification baselines; no majority-router retrieval
score has been evaluated or inferred.

| Source | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Oracle | 13 | 81 | 178 | 232 | 420 |
| Phase 2E winner | 0 | 15 | 275 | 366 | 268 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 |
| 20 | 0.2 | 0.037037037037037035 | 0.0625 | 81 |
| 40 | 0.2581818181818182 | 0.398876404494382 | 0.31346578366445915 | 178 |
| 80 | 0.26229508196721313 | 0.41379310344827586 | 0.3210702341137124 | 232 |
| 160 | 0.5671641791044776 | 0.3619047619047619 | 0.4418604651162791 | 420 |

The confusion matrix uses Oracle rows and predicted columns ordered 10, 20,
40, 80, 160:

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 8 | 5 | 0 |
| 20 | 0 | 3 | 23 | 39 | 16 |
| 40 | 0 | 3 | 71 | 65 | 39 |
| 80 | 0 | 3 | 72 | 96 | 61 |
| 160 | 0 | 6 | 101 | 161 | 152 |

Class 10 is never predicted and has zero recall. Class 20 improves relative to
Phase 2D but remains weak, with only 3 of 81 true examples recovered. The
winner shifts predictions away from class 160 and toward classes 40 and 80.

## Contextual comparison with Phase 2D

Phase 2D is the closest prior baseline because it uses the same Base
checkpoint, exact-token prompt, architecture, data, Oracle, loss, seed, input
features, and label mapping. The comparison remains descriptive: Phase 2E
changes learning rate and the entire cosine horizon, selects across 15
candidates, and reuses the same development examples.

| Metric | Phase 2D | Phase 2E winner | Phase 2E − Phase 2D |
|---|---:|---:|---:|
| Validation CE | 1.3543887827303502 | 1.3759860497016412 | +0.02159726697129094 |
| Accuracy | 0.36904761904761907 | 0.3484848484848485 | -0.02056277056277056 |
| Macro-F1 | 0.22994524079282935 | 0.22777929657889012 | -0.0021659442139392304 |
| Weighted F1 | 0.3644656337102369 | 0.3473258648868964 | -0.017139768823340507 |
| Balanced accuracy | 0.2391812745015638 | 0.24232226137689133 | +0.003140986875327545 |
| Top-2 accuracy | 0.6341991341991342 | 0.6190476190476191 | -0.015151515151515138 |

Phase 2E correctly predicts 19 fewer labels and includes 14 fewer true labels
in its top two classes. Its per-class F1 changes relative to Phase 2D are:

| Class | Phase 2D F1 | Phase 2E F1 | Delta |
|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 |
| 20 | 0.041237113402061855 | 0.0625 | +0.021262886597938145 |
| 40 | 0.31738035264483627 | 0.31346578366445915 | -0.00391456898037712 |
| 80 | 0.29432624113475175 | 0.3210702341137124 | +0.02674399297896063 |
| 160 | 0.49678249678249675 | 0.4418604651162791 | -0.054922031666217674 |

Prediction-count changes are 0, -1, +56, +34, and -89 for classes 10, 20,
40, 80, and 160. Phase 2E improves balanced accuracy slightly but does not
improve the primary macro-F1 and performs worse on accuracy, weighted F1,
top-2, and validation CE. It should not be presented as an overall
improvement over Phase 2D.

## Downstream retrieval

Retrieval completed with status `complete` for the locked `lr5e-6` epoch-4
checkpoint. It evaluated all 924 valid predictions, with no invalid prediction
excluded and no default granularity substituted.

| Retrieval metric | Value |
|---|---:|
| Evaluated examples | 924 |
| Valid-prediction retrievals | 924/924 |
| Invalid predictions without retrieval | 0 |
| Retrieval coverage | 1.0 = 100% |
| Valid-only mean joined retrieval F1 | 0.2793735097402597 |
| Valid-only median joined retrieval F1 | 0.267412 |
| Coverage-adjusted full-set mean joined retrieval F1 | 0.27937350974026 |
| Retrieval wall time | 282.3799051999813 seconds |
| Durable question-processing sum | 271.5550483992556 seconds |
| Runtime segments | 1 |
| Top-k | 5 |
| Paper restricted | true |

The complete, uninterrupted invocation wall time, completed-invocation wall
time, current-segment wall time, and reported retrieval wall time are all
`282.3799051999813` seconds. The reported runtime basis is
`complete_uninterrupted_invocation`. An independent recomputation agrees with
the saved retrieval summary.

Retrieval used only the locked winner and the unchanged protocol: predicted
granularity, same-paper filtering, `top_k=5`, existing
`text-embedding-3-small` 1,536-dimensional embeddings, cosine similarity,
existing Qdrant collections and chunk order, unchanged concatenation, and
joined GPT-2 token-level retrieval F1 version
`qasper-token-prf-v2`. The frozen retrieval configuration hash is
`9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8`,
and the normalization version is
`lowercase-remove-punctuation-collapse-whitespace-v1`. Retrieval cannot select
or revise the classification winner.

Contextual comparison with Phase 2D:

| Retrieval metric | Phase 2D | Phase 2E | Phase 2E − Phase 2D |
|---|---:|---:|---:|
| Mean joined retrieval F1 | 0.2767166677489178 | 0.2793735097402597 | +0.0026568419913419 |
| Median joined retrieval F1 | 0.2558975 | 0.267412 | +0.0115145 |

These small positive retrieval deltas are descriptive only. The same reused
development examples underlie both runs, and Phase 2E was selected by
classification macro-F1 rather than retrieval F1. They are not evidence of a
statistically established retrieval improvement.

Classification macro-F1 measures agreement with the evidence-length Oracle.
Joined retrieval F1 measures downstream evidence-token overlap. They are
different metrics and must be reported separately.

## Runtime and resources

Per-trial run summaries:

| LR | Selected checkpoint | Initial recorded loss | Final recorded loss | Run elapsed (s) | Peak allocated/reserved GPU (GiB) | RSS (GiB) |
|---:|---|---:|---:|---:|---:|---:|
| 5e-6 | epoch 4 / step 284 | 1.5487420707941055 | 1.0789286613464355 | 2044.1943467836827 | 9.0316162109375 / 9.62109375 | 2.0423965454101562 |
| 1e-5 | epoch 4 / step 284 | 1.5487420707941055 | 0.9514243721961975 | 2022.7333836276084 | 9.0316162109375 / 9.62109375 | 1.9690093994140625 |
| 2e-5 | epoch 5 / step 355 | 1.5487420707941055 | 0.1819002278149128 | 2067.4948720689863 | 9.0316162109375 / 9.62109375 | 1.9653129577636719 |

The initial and final recorded losses are individual optimizer-window values,
not epoch averages. They must not be used alone to infer a loss trend or
degree of overfitting.

| Aggregate/runtime item | Value |
|---|---:|
| Sum of three recorded trial elapsed times | 6134.422602480277 seconds |
| All 15 validation-event wall times | 497.5040703564882 seconds |
| Grid-manifest creation to classification-final-summary timestamp span | 6241.472588 seconds |
| Winner model load | 2.47247052565217 seconds |
| Winner isolated final inference | 33.506004774942994 seconds |
| Mean inference/question | 0.03612477649043584 seconds |
| Median inference/question | 0.035192497074604034 seconds |
| Synchronized batch-forward sum | 33.37929347716272 seconds |
| Winner selected-epoch validation | 33.22264311462641 seconds |
| Winner final peak GPU allocated/reserved | 1.7161517143249512 / 1.77734375 GiB |
| Winner final RSS | 1.6993751525878906 GiB |
| Sum of three trial elapsed times plus winner reload | 6170.401077780873 seconds |
| Retrieval wall time | 282.3799051999813 seconds |
| Durable retrieval question-processing sum | 271.5550483992556 seconds |
| Selected trial + final validation + retrieval | 2362.552727284259 seconds |
| Three trial elapsed fields + winner reload + retrieval | 6452.780982980854 seconds |

The `6241.472588`-second value ends at the classification-final summary and is
a timestamp span, not a separately instrumented wall-time field. It includes
orchestration gaps and should be labelled accordingly.

All 852,991,040 parameters were marked trainable. Each trial's gradient audit
passed: the classifier head and language backbone received gradients;
752,398,144 parameters across 321 tensors had gradients. The 100,592,896
parameters across 153 tensors without gradients belong to the composite
checkpoint's vision tower, which is absent from the text-only computation
graph. Therefore, Phase 2E is a full-parameter configuration with observed
language-backbone and head updates, not a claim that the vision tower was
updated.

## Environment and provenance

| Item | Value |
|---|---|
| Python | 3.10.7 |
| Python executable | `/workspace/thesis-granularity-router/.venv-qwen/bin/python` |
| PyTorch | `2.8.0+cu128` |
| CUDA build | 12.8 |
| Transformers | `5.15.0.dev0` |
| Transformers commit | `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7` |
| TensorBoard | 2.20.0 |
| GPU | NVIDIA A100-SXM4-40GB |
| Grid fingerprint | `dc80671e8635cb2e479c7e231662eedb1be0920e28497d8f8e8b016703ff2b2b` |

Every trial used the same initialized classifier-head SHA-256:
`09826669f451891218742ea86926e0b484d1696e57999276889d97b5ccdcbda5`.
Per-trial experiment fingerprints are:

- `lr5e-6`: `d57c1356578b402ddbf7b77fc64de5c7167a70908c45a05235b8ccb9f7e38adf`;
- `lr1e-5`: `96c18d9b75034000be73514afa360d8c2594d697ded6f291e3ee32a4e1fad722`;
- `lr2e-5`: `b8c32c2f4501881db3ee7d04be5b57c455d5fcdd87cc46e17f422010074d9628`.

The repository base commit is
`12c7b1a22f552f83d54a752f87f6687c98b52944`. That commit does not by itself
contain the new Phase 2E files; execution is identified by the
content-addressed source snapshot. The Phase 2E orchestrator SHA-256 is
`fa686f3f363b6b54ec082baf4eb7ea7b21ffe01da8c50f10284d57c65042039e`.
The frozen Phase 2D implementation reused by the orchestrator has SHA-256
`99ba4f9a57b5547e776d81d2c7c94aece2670a9b5ccaf348c8a1fbeb0baa5efc`.
The remote runner SHA-256 is
`183f01df84a334f2bc013d6ae4f95e7d1e676b0deed4630220f5a53b53eda376`.
Accordingly, `execution_implementation_phase: Phase 2D` in run summaries
records deliberate implementation reuse and does not relabel the Phase 2E
study.

## Integrity

All three preflight manifests passed. Each completed-run audit passed all
15/15 checks, covering run identity, steps, five validation events, frozen
configuration, data identity, fingerprints, selected checkpoints, learning
rate, epoch count, warmup, and initial head hash.

At transfer time, all 64 extracted metadata files matched all 64 SHA-256
entries in the remote metadata manifest. Authorized retrieval then changed
exactly two summary files: `comparison/selected_final_summary.json` and
`trials/lr5e-6/final_summary.json`. The final audit verifies that the other
62/62 metadata entries remain unchanged and that both changed summaries have
the expected post-retrieval contents and hashes.

Selected-checkpoint transfer verification passed. All 13/13 files in the
manifest bundle and all 27/27 extracted checkpoint files match their
recorded hashes, with 9/9 files verified for each selected checkpoint. The
verified archive inventory is:

| Variant/checkpoint | Archive bytes | Archive SHA-256 | Chunks | Verified files |
|---|---:|---|---:|---:|
| `lr5e-6` / step 284 | 2,895,385,475 | `d9ad9a71a678ae646df1e1b1dac23d6317b21f8be3ed82834111c57cfb43aa02` | 28 | 9/9 |
| `lr1e-5` / step 284 | 2,884,152,550 | `b96825d2fe23908416bd7ea4ced4e7e89beba7ebe9a30c37f1d57fce2c3ee18f` | 28 | 9/9 |
| `lr2e-5` / step 355 | 2,860,205,438 | `f76fab941057ad6e10db20cff85ca9a86f7b67a28e67f32363d7b1d28108b349` | 28 | 9/9 |

The transferred metadata archive SHA-256 is
`26afbdc8a7f33454c2d27d1023b46daaa216fe36ced0173fe6ca09df946784e8`.
The final audit also verifies zero forbidden payloads, the immutable
15-candidate classification-winner lock, and retrieval recomputation from 924
records with 924 unique question IDs, coverage 1.0, mean
`0.2793735097402597`, and median `0.267412`. It records no experiment rerun,
no retrieval rerun, and no Qdrant mutation.

The authoritative final record is
`integrity/final_post_retrieval_audit.json`. It passed at
`2026-08-08T21:52:50.934743+00:00`. The underlying checkpoint-transfer record
is `integrity/selected_checkpoints_transfer_verification.json`, whose SHA-256
is `701d78a6f0ba1f3c871ea4afd5f19d0a48138eb2276931da4385a7f80f01e4f3`.

## Interpretation and limitations

The lower `5e-6` rate produced the best macro-F1 among the predeclared Phase
2E candidates, with epoch 4 marginally ahead of epoch 5. The primary metric
nevertheless remains slightly below Phase 2D. Phase 2E improves class-20 and
class-80 F1 and aggregate balanced accuracy relative to Phase 2D, while it
reduces class-160 F1, overall accuracy, weighted F1, and top-2 accuracy. It
does not recover class 10.

These values are descriptive development results. The same 924 examples have
been used for checkpoint selection and comparisons across several phases, and
Phase 2E performs an additional 15-way selection on them. This reuse creates
selection optimism. The single seed gives no run-to-run variance, confidence
interval, or significance estimate. A final claim about generalization
requires an untouched test set and a predeclared evaluation protocol.

Earlier Logistic Regression and MLP results used the old retrieval-F1 Oracle.
Their classification metrics are not directly comparable with the Phase 2E
evidence-length-Oracle results. Phase 2C also differs from Phase 2E in prompt
and optimization schedule; Phase 2D is the closest contextual baseline.

## Artifact map

The canonical study root is
`outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/`.
Important paths are:

- `configuration/grid_experiment.json`: predeclared grid and frozen protocol;
- `comparison/lr_grid_metrics.csv`: compact 15-candidate table;
- `comparison/selected_trial.json`: immutable classification-winner lock;
- `comparison/selected_final_summary.json`: complete study-level winner result;
- `trials/<variant>/configuration/`: experiment and preflight records;
- `trials/<variant>/runs/<run_id>/training_config.json`: complete run configuration;
- `trials/<variant>/runs/<run_id>/training_history.jsonl`: optimizer-step history;
- `trials/<variant>/runs/<run_id>/validation_history.jsonl`: all five validation events;
- `trials/<variant>/runs/<run_id>/validation/predictions_step-*.jsonl`: all epoch predictions;
- `trials/<variant>/runs/<run_id>/best_checkpoint.json`: per-trial checkpoint selection;
- `trials/<variant>/runs/<run_id>/phase2e_completed_run_audit.json`: completed-run integrity audit;
- `trials/lr5e-6/final_summary.json`: winning-trial classification summary;
- `trials/lr5e-6/classification/`: metrics, confusion matrix, and histogram;
- `trials/lr5e-6/validation/`: canonical predictions, raw outputs, parsed predictions, invalid records, and runtime;
- `trials/lr5e-6/retrieval/results.jsonl`: all 924 retrieval records;
- `trials/lr5e-6/retrieval/runtime_segments.jsonl`: durable runtime segment;
- `trials/lr5e-6/retrieval/summary.json`: complete retrieval summary;
- `trials/<variant>/runs/<run_id>/checkpoints/<selected-step>/`: retained selected checkpoint for each trial.

Absolute `/dev/shm/...` paths inside run metadata preserve the original CUDA
host provenance. The repository-relative tree above is the retained local
counterpart.

## Reproduction commands

On the CUDA host, using the preserved Phase 2E source snapshot and existing
`.venv-qwen`:

```bash
cd /workspace/thesis-granularity-router
export HF_HOME=/dev/shm/qwen_phase2e_hf
export MPLCONFIGDIR=/dev/shm/phase2e_mpl
export PYTHONPATH=/dev/shm/phase2e_code
export PHASE2E_REPOSITORY_BASE_COMMIT=12c7b1a22f552f83d54a752f87f6687c98b52944

(cd /dev/shm/phase2e_code && sha256sum --check phase2e_source_manifest.sha256)

PY=/workspace/thesis-granularity-router/.venv-qwen/bin/python
SCRIPT=/dev/shm/phase2e_code/qwen_phase2e_sequence_classifier_lr_grid.py
STUDY=/dev/shm/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle

$PY $SCRIPT --study-root $STUDY prepare
for variant in lr5e-6 lr1e-5 lr2e-5; do
  $PY $SCRIPT --study-root $STUDY inspect --variant $variant
  $PY $SCRIPT --study-root $STUDY train --variant $variant
  $PY $SCRIPT --study-root $STUDY audit-completed --variant $variant
done
$PY $SCRIPT --study-root $STUDY select
$PY $SCRIPT --study-root $STUDY final-selected
```

An interrupted trial with a valid checkpoint manifest can be resumed with:

```bash
$PY $SCRIPT --study-root $STUDY resume-latest --variant lr5e-6
```

After the complete artifact tree and selected checkpoints are verified
locally, audit the lock and run unchanged retrieval against the existing,
read-only Qdrant service at `127.0.0.1:6334`:

```powershell
$study = "outputs\qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle"
.\.venv-qwen\Scripts\python.exe qwen_phase2e_posttraining.py --study-root $study audit-selected
.\.venv-qwen\Scripts\python.exe qwen_phase2e_posttraining.py --study-root $study retrieve-selected
.\.venv-qwen\Scripts\python.exe qwen_phase2e_posttraining.py --study-root $study audit-final
```

The original legacy `.venv` is not used or modified by these commands.
The retrieval command must use the existing Qdrant collections read-only; it
must not create, delete, rebuild, or re-index a collection.
