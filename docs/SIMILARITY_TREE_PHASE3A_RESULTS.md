# Phase 3A Results: Multiscale Similarity-Tree Router

## Status and objective

Phase 3A is complete. It implements the supervisor-proposed complementary
router: instead of predicting granularity from question text alone, it uses
the distribution of semantic similarities between the question and every
chunk in the source paper at each preserved chunk level.

The experiment is deliberately separate from Qwen fine-tuning. It asks
whether the already available multiscale chunk hierarchy contains a signal
that correlates with the evidence-length Oracle label.

Authoritative machine-readable result:

`outputs/similarity_tree_phase3a_evidence_length_oracle/final_summary.json`

## Frozen data and leakage boundary

The preserved evidence-length Oracle and splits were not regenerated or
changed:

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

The Oracle remains
`oracle-evidence-length-gpt2-smaller-midpoint-v1`: calculate the GPT-2-token
length of the complete deduplicated ground-truth evidence, choose the closest
class in 10/20/40/80/160, use the smaller class at an exact midpoint, and
clip below/above the candidate range. The Oracle is used only as the target.

Feature construction uses:

- the existing 1,536-dimensional `text-embedding-3-small` question vector;
- existing `PaperChunk` vectors from the same source paper;
- cosine similarities at all five chunk levels; and
- chunk indices to reconstruct parent-child relationships.

It does not use ground-truth evidence, evidence length, answers, paper labels,
evidence embeddings, evidence-to-chunk similarity, retrieved evidence,
retrieval F1, or the Oracle label as a feature. The raw question vector is
used to compute similarities but is not itself passed to the classifier.

The complete score distribution is retained at every level. `top-5` summary
statistics are features or heuristic inputs, not a restriction on feature
extraction. Since a paper normally contains several 160-token chunks, the
physical structure is a forest of local 160 -> 80 -> 40 -> 20 -> 10 branches,
not one global root. Cosine similarities are ranking scores, not calibrated
probabilities.

## Integrity audit

The existing Qdrant service was used read-only at HTTP 6333/gRPC 6334. The
required collections were green before the run:

- `PaperChunk`: 1,701,822 points, 1,536-dimensional cosine vectors;
- `PaperQuestion`: 4,526 points, 1,536-dimensional cosine vectors.

No collection was created, deleted, rebuilt, re-indexed, or modified. The
full collection snapshot matched before extraction, after extraction, after
retrieval, and at finalization. A manual cosine audit reproduced Qdrant's
top-10 scores with maximum absolute difference
`1.0856323240382437e-07`, below the `1e-5` tolerance. The sampled hierarchy
contained 652/326/163/82/41 chunks for 10/20/40/80/160 tokens.

The Oracle file hashes remained:

- train: `64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`;
- validation: `ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`.

## Features and model-selection protocol

Two deterministic linear classifiers were evaluated:

1. a level-aggregate model with 85 distribution features; and
2. the predeclared primary tree model with 173 features: the same level
   summaries plus 88 hierarchy/parent-child features.

Features include distribution moments, quantiles, rank concentration,
softmax concentration, threshold counts, adjacent-level differences, and
parent-child alignment/contrast statistics. Feature standardization is fitted
on training data only.

Hyperparameters were selected by five-fold paper-grouped cross-validation on
the 2,245 train questions. All questions from a paper remain in the same fold.
The grid is learning rate 0.03/0.01/0.003, weight decay 0/0.001/0.01, 300
full-batch epochs, uniform cross-entropy, and seed 42. Macro-F1 is the primary
selection metric. The 924-question validation set was evaluated only after
the train-only choices were locked within Phase 3A.

The selected settings were:

| Model | Features | Learning rate | Weight decay | Epochs | Grouped OOF accuracy | Grouped OOF macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Level aggregate | 85 | 0.03 | 0.001 | 300 | 0.3242761692650334 | 0.22533833565985803 |
| Tree, primary | 173 | 0.01 | 0.0 | 300 | 0.3060133630289532 | 0.22217588936798274 |

The tree model was the predeclared primary formulation; it was not chosen
after inspecting validation. Importantly, hierarchy features did not improve
grouped out-of-fold macro-F1 over the level-only summaries.

## Validation classification results

| Metric | Phase 3A tree model |
|---|---:|
| Accuracy | 0.30303030303030304 (280/924) |
| Macro-F1 | 0.1928144851068439 |
| Weighted F1 | 0.3037047064693007 |
| Balanced accuracy | 0.19804594967839187 |
| Top-2 accuracy | 0.6136363636363636 |
| Mean absolute class-level distance | 1.0281385281385282 |
| Within-one-level accuracy | 0.7337662337662337 |
| Mean absolute token distance | 56.6991341991342 |
| Quadratic weighted kappa | 0.03960842466855796 |

All 924 predictions are valid five-class outputs. The predicted distribution
for 10/20/40/80/160 is 8/31/209/356/320; the Oracle distribution is
13/81/178/232/420.

| Oracle class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 |
| 20 | 0.12903225806451613 | 0.04938271604938271 | 0.07142857142857142 | 81 |
| 40 | 0.16267942583732056 | 0.19101123595505617 | 0.17571059431524547 | 178 |
| 80 | 0.25280898876404495 | 0.3879310344827586 | 0.30612244897959184 | 232 |
| 160 | 0.475 | 0.3619047619047619 | 0.41081081081081083 | 420 |

Confusion matrix, with Oracle rows and predicted columns in class order
10/20/40/80/160:

```text
[[  0,   0,   4,   6,   3],
 [  2,   4,  23,  23,  29],
 [  3,   4,  34,  79,  58],
 [  1,   9,  54,  90,  78],
 [  2,  14,  94, 158, 152]]
```

## Baselines, heuristics, and hierarchy ablation

The deployable train-prior majority is class 80. On validation it has accuracy
0.2510822510822511 and macro-F1 0.08027681660899653. The validation-label
majority is class 160, with descriptive accuracy 0.45454545454545453 and
macro-F1 0.125; it is non-deployable because its identity is obtained from
validation labels.

| Method | Validation accuracy | Macro-F1 | Weighted F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Maximum-similarity level heuristic | 0.049783549783549784 | 0.060034573030522155 | 0.04631400762751346 | 0.25246839439206736 |
| Top-5-mean level heuristic | 0.04329004329004329 | 0.05725663186510825 | 0.04116333385531705 | 0.24406902428341146 |
| Train-tuned penalized-top-5 heuristic | 0.18506493506493507 | 0.14337561538823293 | 0.21885972408993493 | 0.23283692135493764 |
| Train-tuned leaf-breadth heuristic | 0.37554112554112556 | 0.12417130346260578 | 0.2755908183260398 | 0.1667816091954023 |
| Level-aggregate logistic model | 0.3235930735930736 | 0.1891945748453902 | 0.31267712296852007 | 0.20515866865924984 |
| Tree logistic model, primary | 0.30303030303030304 | 0.1928144851068439 | 0.3037047064693007 | 0.19804594967839187 |

On validation, adding hierarchy features raises macro-F1 by
0.003619910261453696 but lowers accuracy by 0.02056277056277056 relative to the
level-only model. Because the grouped out-of-fold result slightly favors the
level-only model, this small validation difference is not robust evidence that
parent-child features add generalizable information.

The primary model beats the deployable train-majority reference on accuracy
and macro-F1. It remains below the descriptive class-160 majority in accuracy,
although above it in macro-F1. Class 10 is not correctly classified, class 20
recall is low, and quadratic weighted kappa is near zero.

## Same-Oracle context

Phases 2D, 2E, and 3A use the same evidence-length Oracle and preserved splits,
so the following descriptive comparison is valid. It is not an independent
test comparison because the validation split has been repeatedly observed.

| Metric | Phase 2D Qwen | Phase 2E Qwen | Phase 3A tree |
|---|---:|---:|---:|
| Accuracy | 0.36904761904761907 | 0.3484848484848485 | 0.30303030303030304 |
| Macro-F1 | 0.22994524079282935 | 0.22777929657889012 | 0.1928144851068439 |
| Weighted F1 | 0.3644656337102369 | 0.3473258648868964 | 0.3037047064693007 |
| Balanced accuracy | 0.2391812745015638 | 0.24232226137689133 | 0.19804594967839187 |
| Top-2 accuracy | 0.6341991341991342 | 0.6190476190476191 | 0.6136363636363636 |
| Mean joined retrieval F1 | 0.2767166677489178 | 0.2793735097402597 | 0.26773840692640694 |
| Median joined retrieval F1 | 0.2558975 | 0.267412 | 0.25228 |

Phase 3A therefore does not improve on Phase 2D or Phase 2E. It does show that
score distributions contain some label signal beyond a train-prior constant,
but the hierarchy-only linear formulation does not solve evidence-length
label learnability.

## Downstream retrieval

The frozen Phase 3A predictions control the unchanged retrieval pipeline:
same-paper filtering, predicted granularity, `top_k=5`, existing
`text-embedding-3-small` vectors, cosine ordering, existing chunk ordering and
concatenation, and GPT-2-token joined retrieval F1.

| Retrieval item | Result |
|---|---:|
| Coverage | 924/924 = 1.0 |
| Mean joined retrieval F1 | 0.26773840692640694 |
| Median joined retrieval F1 | 0.25228 |
| Coverage-adjusted full-set mean | 0.26773840692640705 |
| Retrieval wall time | 168.46491879993118 seconds |

Classification metrics measure agreement with the evidence-length Oracle.
Joined retrieval F1 measures token overlap after the chosen granularity drives
retrieval. They are distinct outcomes.

## Runtime and recovery record

The main feature extraction ran in the unchanged legacy `.venv` with Python
3.9.12, NumPy 2.0.2, and PyTorch 2.8.0+cpu. Its initial long gRPC scroll hit a
Qdrant `DEADLINE_EXCEEDED` after safely persisting 1,844 train rows. The last
pre-failure progress record was 1,827/2,245 at 3,276.9802199 seconds. No
collection changed and no completed row was lost.

The code was made retryable and the existing recovery data was resumed over
REST with a 300-second timeout. The successful/resumed recorded times are:

| Stage | Wall time (seconds) |
|---|---:|
| Resume and finish train extraction | 827.396431 |
| Validation extraction | 1311.1521062 |
| Grouped CV, final fitting, validation | 56.6241827 |
| Retrieval | 168.46491879993118 |
| Sum of successful/resumed timed stages | 2363.6376387 |

The exact total including the failed first extraction attempt is unavailable;
it must not be inferred from the successful-stage sum. Retrieval was executed
with the already existing `.venv-qwen` only because the frozen Phase 1
retrieval module imports `psutil`, which is absent from legacy `.venv`. No
environment was modified.

## Interpretation and limitations

- Simple maximum or mean similarity is a poor proxy for the required evidence
  length; fine-grained chunks often have the largest raw similarity even when
  the answer requires broader evidence.
- A learned linear model over complete per-level distributions is materially
  better than those naive heuristics.
- Explicit parent-child features add only a small validation macro-F1 change
  and do not improve grouped train-paper OOF macro-F1.
- The smallest two classes remain difficult because they are rare and because
  score shape is only indirectly related to complete evidence length.
- Phase 3A is a one-seed development experiment. The preserved validation set
  was already examined in Phases 1 through 2E, so it is not an unbiased final
  test set. No confidence interval or statistical-significance claim is made.
- Previous Logistic Regression/MLP results trained on the old retrieval-F1
  Oracle remain not directly comparable. Only same-new-Oracle results should
  be placed in one classification table.

A justified next study would combine the complementary sources under a
train-only protocol: question-text/Qwen representation plus similarity-tree
features, followed by repeated paper-grouped cross-validation. A separate
untouched test split is required before any final generalization claim.

## Artifacts

All authoritative artifacts are under
`outputs/similarity_tree_phase3a_evidence_length_oracle/`:

- `final_summary.json`;
- `configuration/experiment.json`;
- `integrity/preflight_audit.json` and `integrity/final_audit.json`;
- `features/extraction_summary.json`, full compressed score trees, and slim
  feature records for train and validation;
- `cross_validation/paper_grouped_folds.json` and both model-selection files;
- `models/` serialized classifiers and metadata;
- `heuristics/metrics.json`;
- `validation/predictions.jsonl`;
- `classification/metrics.json`, `confusion_matrix.csv`, and
  `predicted_vs_oracle.svg`; and
- `retrieval/results.jsonl` and `retrieval/summary.json`.

The standalone narrative report is
`reports/similarity_tree_phase3a_evidence_length_oracle/experiment_report.md`.

## Reproduction commands

With the existing local Qdrant service running and no collection mutation:

```powershell
.\.venv\Scripts\python.exe similarity_tree_phase3a.py audit
.\.venv\Scripts\python.exe similarity_tree_phase3a.py extract
.\.venv\Scripts\python.exe similarity_tree_phase3a.py train-evaluate
.\.venv-qwen\Scripts\python.exe similarity_tree_phase3a.py retrieve
.\.venv\Scripts\python.exe similarity_tree_phase3a.py finalize
.\.venv-qwen\Scripts\python.exe -m pytest -q tests\test_similarity_tree_phase3a.py tests\test_router.py
```

`extract` and `retrieve` save incrementally and resume by question ID. Existing
completed artifacts should be preserved when reproducing the archived run.
