# Question-Embedding Granularity Router

## Scope and leakage boundary

`granularity_router.py` trains a classifier from persisted `RouterDataset` records. The only inference feature is the existing question embedding. Evidence text, evidence vectors, per-granularity retrieval metrics, oracle scores, and label tie-break metadata are never included in the feature matrix.

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

## Current data readiness

## Pretrained Qwen3.5-0.8B Phase 1 result

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

As checked on 2026-06-22, the local `RouterDataset` contains two validation records, both targeting the 20-token class, under two different evaluation configuration hashes. It contains no train records. The live training command therefore stops with `No QASPER train router examples exist for the selected configuration`; no scientifically valid model has been trained from the live collection yet. Generate complete train and validation oracle records with one frozen evaluation configuration before training.
