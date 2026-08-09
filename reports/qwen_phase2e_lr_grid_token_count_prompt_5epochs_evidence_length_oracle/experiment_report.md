# Experiment Report: Qwen Phase 2E Learning-Rate Grid

## Executive summary

Phase 2E ran three fresh five-epoch, full-parameter sequence-classification
trials with learning rates `5e-6`, `1e-5`, and `2e-5`. The study otherwise
freezes the Phase 2D exact-token prompt, Qwen Base checkpoint, evidence-length
Oracle, data, inputs, classifier, uniform loss, seed, batching, and evaluation
protocol.

The predeclared primary selection metric was validation macro-F1. Across all
15 learning-rate-by-epoch candidates, the locked winner is `5e-6`, epoch 4,
`step-000284`, with macro-F1 `0.22777929657889012`. The winner has accuracy
`0.3484848484848485`, weighted F1 `0.3473258648868964`, balanced accuracy
`0.24232226137689133`, and top-2 accuracy `0.6190476190476191`.
Unchanged retrieval for the locked winner then achieved mean joined retrieval
F1 `0.2793735097402597` and median `0.267412` with 100% coverage.

The study did not improve on Phase 2D's primary macro-F1
`0.22994524079282935`. It slightly improves balanced accuracy, class-20 F1,
and class-80 F1, but reduces accuracy, weighted F1, top-2 accuracy, and
class-160 F1. Class 10 remains unrecovered.

This is a development-set hyperparameter result. The same 924 validation
questions have been examined in multiple earlier phases and select one of 15
Phase 2E candidates. The reported winner is not an unbiased test-set estimate.

## Experiment identity

| Field | Value |
|---|---|
| Phase | Phase 2E |
| Study ID | `qwen-phase2e-lr-grid-token-count-prompt-5epochs-seed42-v1` |
| Formulation | `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr-grid-v1` |
| Grid fingerprint | `dc80671e8635cb2e479c7e231662eedb1be0920e28497d8f8e8b016703ff2b2b` |
| Model | `Qwen/Qwen3.5-0.8B-Base` |
| Model revision | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Architecture | `AutoModelForSequenceClassification` |
| Seed | 42 |
| Winner | `lr5e-6`, epoch 4, `step-000284` |
| Selection lock time | `2026-08-08T20:26:31.564119+00:00` |
| Current result state | Complete, including locked-winner retrieval |

The global winner run ID is
`qwen-phase2e-base-sequence-classifier-token-count-prompt-lr5e-6-5epochs-full-parameter-20260808-seed42-v1`.

## Motivation and controlled changes

Phase 2D used learning rate `2e-5` for three epochs. Phase 2E tests whether a
lower learning rate and a longer schedule improve the evidence-length label
classifier. The predeclared changes from Phase 2D are limited to:

- Phase 2E artifact identity;
- learning rate from the grid `5e-6`, `1e-5`, `2e-5`;
- five epochs instead of three;
- the resulting 355-step cosine horizon and 18 warmup steps.

Everything else remains fixed. Because the cosine horizon changes from
213/11 total/warmup steps in Phase 2D to 355/18 in Phase 2E, the Phase 2E
`2e-5` trial is not a continuation of, or trajectory-identical control for,
Phase 2D.

## Oracle, dataset, and model inputs

The Oracle uses the GPT-2 token count of complete deduplicated ground-truth
evidence. It selects the closest class among 10, 20, 40, 80, and 160, uses the
smaller class for midpoint ties, clips values below 10 to 10, and clips values
above 160 to 160. It is independent of retrieval F1, embedding quality,
cosine similarity, retrieved chunks, and router predictions.

| Split | Examples | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation/development | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

The validation Oracle is strongly imbalanced: class 160 alone is 45.45% of
the split, whereas class 10 is about 1.41%.

Oracle file hashes are:

- train: `64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`;
- validation: `ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`.

The input is exactly the fixed instruction, two newline characters,
`Question: `, and the original question. The prompt is:

> You are a router for a retrieval-augmented generation system. Based only on
> the question, select the option representing the context size most suitable
> for retrieving the evidence required to answer it. Choose exactly one value
> from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, 4 = 80 tokens, 5 = 160
> tokens. Return only the number

Its SHA-256 is
`b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5`.
No evidence, evidence length, answer, paper text, retrieved chunk, retrieval
score, metadata, handcrafted feature, chat template, or assistant target is
included. The sequence-classification head maps IDs 0–4 to canonical chunk
sizes 10–160 and returns comparable five-class logits; it does not generate a
textual routing answer.

## Frozen training protocol

| Setting | Value |
|---|---|
| Objective | Uniform five-class cross-entropy |
| Learning rates | `5e-6`, `1e-5`, `2e-5` |
| Epochs | 5 |
| Steps per epoch / total | 71 / 355 |
| Checkpoint steps | 71, 142, 213, 284, 355 |
| Batch / accumulation / effective batch | 4 / 8 / 32 |
| Maximum sequence length | 128 |
| Weight decay | 0.01 |
| Scheduler | Cosine |
| Warmup | 0.05 = 18 steps |
| Gradient clipping | 1.0 |
| Class weights | Uniform |
| Evaluation/checkpointing | End of every epoch |
| Early stopping | None |
| Dtype / device | `torch.bfloat16` / CUDA |
| Quantization | None |

Fixed run IDs:

| Variant | Run ID |
|---|---|
| `lr5e-6` | `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr5e-6-5epochs-full-parameter-20260808-seed42-v1` |
| `lr1e-5` | `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr1e-5-5epochs-full-parameter-20260808-seed42-v1` |
| `lr2e-5` | `qwen-phase2e-base-sequence-classifier-token-count-prompt-lr2e-5-5epochs-full-parameter-20260808-seed42-v1` |

`Qwen2Tokenizer` uses right padding and `<|endoftext|>` as pad token ID
248044. No preflight input is truncated: train lengths are 95–121 with mean
`101.13363028953229`; validation lengths are 96–124 with mean
`100.59740259740259`.

## Model selection protocol

The grid winner is selected lexicographically using:

1. higher macro-F1;
2. higher accuracy;
3. higher weighted F1;
4. higher balanced accuracy;
5. lower validation cross-entropy;
6. earlier optimizer step;
7. lower numeric learning rate for an otherwise exact tie.

The winner was locked before retrieval. The selection artifact explicitly
records `retrieval_was_not_used_for_selection: true`. Downstream retrieval is
not allowed to choose another trial or epoch.

## All 15 validation candidates

The distribution column is ordered 10/20/40/80/160.

| LR | Epoch | Step | Validation CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 | Predicted distribution | Validation wall (s) |
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

The selected row's macro-F1 exceeds the next candidate, `5e-6` epoch 5, by
only `0.0021861577015921674`. Per-trial selections are:

| Variant | Selected epoch/step | CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 |
|---|---|---:|---:|---:|---:|---:|---:|
| `lr5e-6` | 4 / 284 | 1.3759860497016412 | 0.3484848484848485 | 0.22777929657889012 | 0.3473258648868964 | 0.24232226137689133 | 0.6190476190476191 |
| `lr1e-5` | 4 / 284 | 1.458046925016296 | 0.3235930735930736 | 0.21540884371375907 | 0.32864368966292395 | 0.22681571407388273 | 0.6038961038961039 |
| `lr2e-5` | 5 / 355 | 2.2110366181377725 | 0.3170995670995671 | 0.2252323080025679 | 0.3281524813999655 | 0.23285430021449655 | 0.5562770562770563 |

The macro-F1 rule deliberately does not choose the maximum accuracy
(`0.4090909090909091`, `2e-5` epoch 1), maximum top-2
(`0.7023809523809523`, `2e-5` epoch 2), or minimum CE
(`1.3393364873799412`, `2e-5` epoch 1).

## Selected-checkpoint classification

The post-training reload selected `lr5e-6` `step-000284` and reproduced the
saved epoch output scores exactly for all 924 examples.

| Metric | Result |
|---|---:|
| Validation cross-entropy | 1.3759860497016412 |
| Accuracy | 0.3484848484848485 |
| Correct predictions | 322/924 |
| Macro-F1 | 0.22777929657889012 |
| Weighted F1 | 0.3473258648868964 |
| Balanced accuracy | 0.24232226137689133 |
| Top-2 accuracy | 0.6190476190476191 |
| Top-2 correct | 572/924 |
| Valid output rate | 924/924 = 1.0 |
| Invalid outputs | 0 |
| Majority class | 160 |
| Majority accuracy | 0.45454545454545453 |
| Majority macro-F1 | 0.125 |

The majority baseline is a classification baseline only. There is no saved or
computed majority-router retrieval result.

### Oracle and prediction distributions

| Distribution | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Oracle | 13 | 81 | 178 | 232 | 420 |
| Predicted | 0 | 15 | 275 | 366 | 268 |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 |
| 20 | 0.2 | 0.037037037037037035 | 0.0625 | 81 |
| 40 | 0.2581818181818182 | 0.398876404494382 | 0.31346578366445915 | 178 |
| 80 | 0.26229508196721313 | 0.41379310344827586 | 0.3210702341137124 | 232 |
| 160 | 0.5671641791044776 | 0.3619047619047619 | 0.4418604651162791 | 420 |

The winner correctly classifies no true class-10 label because it never
predicts class 10. It correctly predicts 3/81 class-20 examples. These
rare-class failures remain visible in macro-F1 even though the majority class
dominates accuracy and weighted F1.

### Confusion matrix

Rows are Oracle classes and columns are predicted classes in order 10, 20,
40, 80, 160.

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 8 | 5 | 0 |
| 20 | 0 | 3 | 23 | 39 | 16 |
| 40 | 0 | 3 | 71 | 65 | 39 |
| 80 | 0 | 3 | 72 | 96 | 61 |
| 160 | 0 | 6 | 101 | 161 | 152 |

## Phase 2D contextual comparison

Phase 2D is the closest existing reference. It shares the prompt, model,
revision, five-logit classifier, preserved split, new Oracle, labels, loss,
seed, and non-learning-rate hyperparameters. It differs in its three-epoch
schedule and was not selected from the Phase 2E 15-way grid.

| Metric | Phase 2D | Phase 2E winner | E − D |
|---|---:|---:|---:|
| Validation CE | 1.3543887827303502 | 1.3759860497016412 | +0.02159726697129094 |
| Accuracy | 0.36904761904761907 | 0.3484848484848485 | -0.02056277056277056 |
| Macro-F1 | 0.22994524079282935 | 0.22777929657889012 | -0.0021659442139392304 |
| Weighted F1 | 0.3644656337102369 | 0.3473258648868964 | -0.017139768823340507 |
| Balanced accuracy | 0.2391812745015638 | 0.24232226137689133 | +0.003140986875327545 |
| Top-2 accuracy | 0.6341991341991342 | 0.6190476190476191 | -0.015151515151515138 |

| Class | Phase 2D F1 | Phase 2E F1 | E − D |
|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 |
| 20 | 0.041237113402061855 | 0.0625 | +0.021262886597938145 |
| 40 | 0.31738035264483627 | 0.31346578366445915 | -0.00391456898037712 |
| 80 | 0.29432624113475175 | 0.3210702341137124 | +0.02674399297896063 |
| 160 | 0.49678249678249675 | 0.4418604651162791 | -0.054922031666217674 |

Prediction counts change from `0/16/219/332/357` to
`0/15/275/366/268`. Phase 2E moves 89 predictions away from class 160 while
adding 56 to class 40 and 34 to class 80. This improves balanced accuracy but
reduces the aggregate metrics dominated by overall correctness and support.

Phase 2E is 19 correct labels and 14 top-2 inclusions below Phase 2D. The
result does not support a claim that the five-epoch LR grid improves the
router overall.

## Retrieval evaluation

Downstream retrieval completed with status `complete` after the classification
winner was locked. It uses only `lr5e-6` `step-000284` and preserves:

- source-paper restriction;
- predicted canonical granularity;
- `top_k=5`;
- the existing `text-embedding-3-small` embeddings with dimension 1,536;
- cosine similarity;
- existing Qdrant collections, chunk order, and concatenation;
- GPT-2 tokenization and unchanged joined token-level retrieval F1 version
  `qasper-token-prf-v2`;
- retrieval configuration hash
  `9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8`;
- normalization version
  `lowercase-remove-punctuation-collapse-whitespace-v1`.

| Retrieval metric | Value |
|---|---:|
| Evaluated examples | 924 |
| Valid-prediction retrievals | 924/924 |
| Invalid predictions without retrieval | 0 |
| Retrieval coverage | 1.0 = 100% |
| Valid-only mean joined retrieval F1 | 0.2793735097402597 |
| Valid-only median joined retrieval F1 | 0.267412 |
| Coverage-adjusted full-set mean joined retrieval F1 | 0.27937350974026 |
| Top-k | 5 |
| Paper restricted | true |
| Retrieval wall time | 282.3799051999813 seconds |
| Durable question-processing sum | 271.5550483992556 seconds |
| Runtime segments | 1 |

All 924 canonical classifier outputs are explicit five-logit argmaxes. No
default or parser fallback is used. The current, completed, uninterrupted, and
reported invocation wall times are all `282.3799051999813` seconds, with
reported basis `complete_uninterrupted_invocation`.

The independently recomputed retrieval result agrees with the saved summary.
Relative to Phase 2D:

| Retrieval metric | Phase 2D | Phase 2E | E − D |
|---|---:|---:|---:|
| Mean joined retrieval F1 | 0.2767166677489178 | 0.2793735097402597 | +0.0026568419913419 |
| Median joined retrieval F1 | 0.2558975 | 0.267412 | +0.0115145 |

These positive deltas are descriptive only: both rows use the repeatedly
observed development examples, and Phase 2E was selected using classification
macro-F1 rather than retrieval F1. No statistical improvement claim follows
from them.

The retrieval result cannot revise the selection. Classification metrics
measure Oracle-label prediction; joined retrieval F1 measures downstream
token overlap and is a distinct endpoint.

## Runtime and hardware

### Trial runs

| LR | Selected epoch/step | Initial recorded loss | Final recorded loss | Elapsed (s) | Peak GPU allocated/reserved (GiB) | RSS (GiB) |
|---:|---|---:|---:|---:|---:|---:|
| 5e-6 | 4/284 | 1.5487420707941055 | 1.0789286613464355 | 2044.1943467836827 | 9.0316162109375 / 9.62109375 | 2.0423965454101562 |
| 1e-5 | 4/284 | 1.5487420707941055 | 0.9514243721961975 | 2022.7333836276084 | 9.0316162109375 / 9.62109375 | 1.9690093994140625 |
| 2e-5 | 5/355 | 1.5487420707941055 | 0.1819002278149128 | 2067.4948720689863 | 9.0316162109375 / 9.62109375 | 1.9653129577636719 |

These logged initial/final losses are individual optimizer-window losses, not
epoch means. In particular, the low final `2e-5` window must not be interpreted
as proof of a low validation loss; its selected validation CE is
`2.2110366181377725`.

### Aggregate and selected-checkpoint timing

| Measurement | Value |
|---|---:|
| Sum of trial elapsed fields | 6134.422602480277 seconds |
| Sum of 15 validation walls | 497.5040703564882 seconds |
| Manifest-to-classification-final timestamp span | 6241.472588 seconds |
| Selected model load | 2.47247052565217 seconds |
| Isolated selected-checkpoint inference | 33.506004774942994 seconds |
| Mean inference/question | 0.03612477649043584 seconds |
| Median inference/question | 0.035192497074604034 seconds |
| Batch-forward allocation sum | 33.37929347716272 seconds |
| Selected epoch validation wall | 33.22264311462641 seconds |
| Final peak GPU allocated/reserved | 1.7161517143249512 / 1.77734375 GiB |
| Final RSS | 1.6993751525878906 GiB |
| Trial-elapsed sum plus final reload | 6170.401077780873 seconds |
| Retrieval wall time | 282.3799051999813 seconds |
| Durable retrieval question-processing sum | 271.5550483992556 seconds |
| Selected trial + final validation + retrieval | 2362.552727284259 seconds |
| Trial-elapsed sum + final reload + retrieval | 6452.780982980854 seconds |

The manifest-to-classification-final value is a timestamp difference that
includes orchestration gaps, not a dedicated uninterrupted timer.

### Environment

| Component | Value |
|---|---|
| Python | 3.10.7 |
| Environment executable | `/workspace/thesis-granularity-router/.venv-qwen/bin/python` |
| PyTorch | `2.8.0+cu128` |
| CUDA | 12.8 |
| Transformers | `5.15.0.dev0` |
| Transformers commit | `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7` |
| TensorBoard | 2.20.0 |
| GPU | NVIDIA A100-SXM4-40GB |

## Gradient and parameter audit

All 852,991,040 model parameters were marked trainable. In every trial, the
classifier head and language backbone received gradients. The audit reports
752,398,144 parameters across 321 tensors with gradients and 100,592,896
parameters across 153 tensors without gradients. The latter parameters are in
the unused vision tower of the composite checkpoint. With text-only input,
the vision tower is outside the computation graph and receives no update.

The correct description is therefore full-parameter sequence-classification
configuration with verified language-backbone and classifier-head gradients,
not a claim that every marked parameter, including the vision tower, changed.

## Reproducibility and source identity

Each trial starts from the same seeded classifier head, with shape `[5,1024]`,
no bias, BF16 weights, and SHA-256
`09826669f451891218742ea86926e0b484d1696e57999276889d97b5ccdcbda5`.

| Variant | Experiment fingerprint | Resume-contract SHA-256 |
|---|---|---|
| `lr5e-6` | `d57c1356578b402ddbf7b77fc64de5c7167a70908c45a05235b8ccb9f7e38adf` | `b4a94227445e20e24ebbff868cf74d18e7ad5fc1bf0680c487e6ffb5379bcac2` |
| `lr1e-5` | `96c18d9b75034000be73514afa360d8c2594d697ded6f291e3ee32a4e1fad722` | `2c7b25d3dbaf5fc320c2ee735576062dcc7ececd36ab964a1b459aab60e792b0` |
| `lr2e-5` | `b8c32c2f4501881db3ee7d04be5b57c455d5fcdd87cc46e17f422010074d9628` | `f4b661507bc7e03faa5a15da4b35ef4bd7547db8b2b687d786fc9fa319e1df1b` |

The recorded repository base commit is
`12c7b1a22f552f83d54a752f87f6687c98b52944`, but it does not contain the
new Phase 2E files by itself. The executed worktree is content-addressed:

| Source | SHA-256 |
|---|---|
| Phase 2E orchestrator | `fa686f3f363b6b54ec082baf4eb7ea7b21ffe01da8c50f10284d57c65042039e` |
| Frozen Phase 2D execution module | `99ba4f9a57b5547e776d81d2c7c94aece2670a9b5ccaf348c8a1fbeb0baa5efc` |
| Remote runner | `183f01df84a334f2bc013d6ae4f95e7d1e676b0deed4630220f5a53b53eda376` |
| `qwen_phase2.py` | `c9a6f2a277bd841d6bf0ede9e948b18e91e1a8f5a298f7d704d0b4279c99ed39` |
| `qwen_phase2b.py` | `60572d8c3054e7ef76055b2c40cf65c2999ef18000930f5a6967fd2ae673041c` |
| `qwen_phase2c_sequence_classifier.py` | `6eeb155296b239463d5ba7c8c75dfed8dd59f8c5285cab1edf7cb6a553f9aefb` |

The run-summary field `execution_implementation_phase: Phase 2D` records the
intentional reuse of the audited Phase 2D computational implementation behind
the Phase 2E orchestrator. The study identity, grid fingerprint, trial
fingerprints, configuration, and selection artifacts establish that these are
Phase 2E runs.

## Integrity and transfer audit

All three preflight checks passed. Each completed-run audit passed 15/15
assertions, including frozen configuration, data hashes, total steps, five
validation events, learning rate, warmup, selected checkpoint, fingerprint,
and initial head identity.

The winner reload reports:

- exact score match for all 924 selected-epoch records;
- no missing, unexpected, or mismatched model keys;
- final head SHA-256
  `eb2cdb99b95c6941967fa9ec772729fd27c6ae613ffd9a7215332e0ede39b933`;
- 924 canonical predictions, 924 raw records, 924 parsed records, and zero
  invalid records.

At transfer time, all 64 extracted metadata files matched all 64 SHA-256
entries in the remote metadata manifest. The authorized retrieval evaluation
then updated exactly two summaries:

- `comparison/selected_final_summary.json`, from transfer-time SHA-256
  `f45d5ec6beb22eeef6fc7ce12dbb43990ddb8d6578dffcbd90714bef4be38a7b`
  to final SHA-256
  `d7ec1d097851dafbb761883bfd76448106fafb8753c502c672ac94dc2bdc0c81`;
- `trials/lr5e-6/final_summary.json`, from transfer-time SHA-256
  `cabdd0e0ff3612941f9e0cb39c66a3ef324bf67842de7ae1e52d000ba1c5cf69`
  to final SHA-256
  `22332d8aabe52c3b30dfa374f04fd4044c4169e62bc8dae8db21b44b7a46454f`.

The final audit verifies all 62/62 remaining metadata entries unchanged and
both summary changes as authorized post-retrieval updates. The selected
checkpoint transfer verification passed: 13/13 manifest-bundle files
and 27/27 extracted checkpoint files match, with 9/9 files for each selected
checkpoint. The verified inventory is:

| Variant / selected checkpoint | Bytes | Archive SHA-256 | Chunks | Verified files |
|---|---:|---|---:|---:|
| `lr5e-6` / `step-000284` | 2,895,385,475 | `d9ad9a71a678ae646df1e1b1dac23d6317b21f8be3ed82834111c57cfb43aa02` | 28 | 9/9 |
| `lr1e-5` / `step-000284` | 2,884,152,550 | `b96825d2fe23908416bd7ea4ced4e7e89beba7ebe9a30c37f1d57fce2c3ee18f` | 28 | 9/9 |
| `lr2e-5` / `step-000355` | 2,860,205,438 | `f76fab941057ad6e10db20cff85ca9a86f7b67a28e67f32363d7b1d28108b349` | 28 | 9/9 |

The transferred metadata archive SHA-256 is
`26afbdc8a7f33454c2d27d1023b46daaa216fe36ced0173fe6ca09df946784e8`.
The transfer verification record is
`integrity/selected_checkpoints_transfer_verification.json`, with SHA-256
`701d78a6f0ba1f3c871ea4afd5f19d0a48138eb2276931da4385a7f80f01e4f3`.
It passed at `2026-08-08T21:25:36.5263304+00:00`; the verified manifest-bundle
SHA-256 is
`1ce870ebfcfcc2d370f6357f0006f393a4c732ee880394b5e1793a96b26d3b46`.

The authoritative post-retrieval audit is
`integrity/final_post_retrieval_audit.json`. It passed at
`2026-08-08T21:52:50.934743+00:00` and verifies:

- the immutable classification lock over all 15 candidates, with retrieval
  excluded from selection;
- zero forbidden payloads;
- 924 retrieval records and 924 unique question IDs;
- independently recomputed coverage 1.0, mean joined retrieval F1
  `0.2793735097402597`, and median `0.267412`;
- no experiment rerun, no retrieval rerun, and no Qdrant mutation.

## Scientific interpretation

Within the predeclared grid, reducing the rate to `5e-6` and selecting epoch 4
gives the highest development macro-F1. The small gap to two other candidates
and the oscillating class distributions show that the result is sensitive to
checkpoint timing. Higher-rate later epochs recover some class-10 predictions,
but those predictions are not correct; the selected model still has zero
class-10 recall.

The Phase 2E winner does not surpass Phase 2D on the primary metric. Its modest
balanced-accuracy improvement comes with fewer correct labels and lower
class-160 performance. The defensible conclusion is that this three-rate,
five-epoch grid found no clear improvement over the Phase 2D development
baseline, not that learning-rate tuning is generally ineffective.

## Limitations and comparability

- The 924 examples are a repeatedly observed development/model-selection set.
  Selecting one of 15 candidates increases optimistic selection bias.
- There is one seed, so the study provides no run-to-run variance, confidence
  interval, or statistical-significance estimate.
- Phase 2E and Phase 2D use different cosine horizons. Their difference is not
  attributable solely to learning rate or two extra epochs.
- The Oracle is strongly imbalanced. Accuracy and weighted F1 emphasize class
  160; macro-F1 and balanced accuracy expose weak rare-class performance.
- Phase 1 through Phase 2E reuse the same validation examples. A future final
  assessment requires a separate untouched test set and predeclared protocol.
- Logistic Regression and MLP results based on the old retrieval-F1 Oracle are
  not directly comparable to these evidence-length-Oracle classification
  metrics.
- Retrieval is a separate downstream endpoint; its completed result is not a
  classification metric and did not participate in checkpoint selection.

## Artifact inventory

Canonical root:
`outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/`.

Study-level artifacts:

- `configuration/grid_experiment.json`;
- `comparison/lr_grid_metrics.csv`;
- `comparison/selected_trial.json`;
- `comparison/selected_final_summary.json`.

Every trial contains:

- `configuration/experiment.json` and `preflight_manifest.json`;
- run `training_config.json` and `dataset_manifest.json`;
- `formatted_example_inspection.json` and `gradient_coverage_audit.json`;
- `training_history.jsonl` and `validation_history.jsonl`;
- five `validation/predictions_step-*.jsonl` files;
- `checkpoint_manifest.json`, `best_checkpoint.json`, `summary.json`, and
  `phase2e_completed_run_audit.json`;
- its retained selected checkpoint.

The `lr5e-6` winning trial additionally contains:

- `final_summary.json`;
- canonical `validation/predictions.jsonl`;
- `validation/raw_outputs.jsonl`;
- `validation/parsed_predictions.jsonl`;
- empty `validation/invalid_outputs.jsonl`;
- `validation/runtime_summary.json`;
- `classification/metrics.json`;
- `classification/confusion_matrix.csv`;
- `classification/predicted_vs_oracle.svg`.
- `retrieval/results.jsonl`;
- `retrieval/runtime_segments.jsonl`;
- `retrieval/summary.json`.

Absolute `/dev/shm/...` paths in original metadata are provenance from the
ephemeral CUDA host. Repository-relative paths under the canonical root are
the retained local artifacts.

## Exact reproduction commands

Use the preserved source snapshot and the existing Qwen environment on the
CUDA host:

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

Resume a valid interrupted trial with:

```bash
$PY $SCRIPT --study-root $STUDY resume-latest --variant lr5e-6
```

After transferring and verifying all artifacts, perform the read-only winner
audit and unchanged downstream retrieval locally against the existing Qdrant
service at `127.0.0.1:6334`:

```powershell
$study = "outputs\qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle"
.\.venv-qwen\Scripts\python.exe qwen_phase2e_posttraining.py --study-root $study audit-selected
.\.venv-qwen\Scripts\python.exe qwen_phase2e_posttraining.py --study-root $study retrieve-selected
.\.venv-qwen\Scripts\python.exe qwen_phase2e_posttraining.py --study-root $study audit-final
```

These commands use `.venv-qwen`; they do not modify the original legacy
`.venv`. Retrieval must use the existing Qdrant collections read-only and must
not create, rebuild, delete, or re-index a collection.
