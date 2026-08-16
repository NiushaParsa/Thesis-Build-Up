# Experiment Report: Phase 3A Similarity-Tree Router

## Executive result

Phase 3A completed the supervisor-proposed multiscale similarity-tree
experiment on the preserved evidence-length Oracle. The predeclared primary
tree logistic model achieved validation accuracy 0.30303030303030304,
macro-F1 0.1928144851068439, weighted F1 0.3037047064693007, balanced accuracy
0.19804594967839187, and top-2 accuracy 0.6136363636363636. Its downstream
same-paper `top_k=5` mean/median joined retrieval F1 is
0.26773840692640694/0.25228 at 924/924 coverage.

The result is a completed negative/mixed finding, not an improvement claim.
It beats the deployable class-80 train-prior constant on accuracy and macro-F1
but remains below Phase 2D and Phase 2E on the primary macro-F1 and downstream
retrieval metrics. The explicit hierarchy features produce only a small
validation macro-F1 increase over level-only score summaries, and that
increase is not present in grouped train-paper cross-validation.

## Research question

Can the shape and parent-child structure of same-paper question-to-chunk
semantic similarity scores predict the new evidence-length Oracle class more
effectively than question-only routing?

The design follows the proposed 160 -> 80 -> 40 -> 20 -> 10 hierarchy. A full
paper contains multiple top-level 160-token chunks, so the implementation is a
multiscale forest. Children are reconstructed from existing chunk indices.
Cosine scores are treated as similarities rather than calibrated
probabilities.

## Protocol

- Preserved train: 2,245 questions from 845 papers.
- Preserved validation: 924 questions from 277 disjoint papers.
- Labels: 10/20/40/80/160 from the unchanged GPT-2 evidence-length Oracle.
- Inputs: all same-paper question-to-chunk cosine scores at all five levels.
- Embeddings: existing 1,536-dimensional `text-embedding-3-small` vectors.
- Forbidden features: evidence, evidence length, answers, evidence embeddings,
  evidence-to-chunk similarity, retrieval F1, and Oracle label.
- Selection: five-fold paper-grouped train-only cross-validation, macro-F1.
- Primary model: predeclared 173-feature tree logistic classifier.
- Comparator: 85-feature level-aggregate logistic classifier.
- Training: deterministic seed 42, uniform cross-entropy, AdamW, 300 epochs.
- Validation: evaluated after train-only hyperparameter lock.
- Retrieval: unchanged read-only Qdrant, source-paper restriction, `top_k=5`.

The selected tree settings are learning rate 0.01 and weight decay 0.0. The
selected level-only settings are learning rate 0.03 and weight decay 0.001.

## Label and prediction distributions

| Split/distribution | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Train Oracle | 55 | 267 | 586 | 687 | 650 |
| Validation Oracle | 13 | 81 | 178 | 232 | 420 |
| Phase 3A prediction | 8 | 31 | 209 | 356 | 320 |

The validation Oracle remains strongly imbalanced: class 160 represents
420/924 = 45.45%. The primary model shifts mass toward classes 40 and 80 and
underpredicts class 160.

## Classification result

| Metric | Result |
|---|---:|
| Accuracy | 0.30303030303030304 |
| Macro-F1 | 0.1928144851068439 |
| Weighted F1 | 0.3037047064693007 |
| Balanced accuracy | 0.19804594967839187 |
| Top-2 accuracy | 0.6136363636363636 |
| Mean absolute class distance | 1.0281385281385282 |
| Within-one-level accuracy | 0.7337662337662337 |
| Mean absolute token distance | 56.6991341991342 |
| Quadratic weighted kappa | 0.03960842466855796 |

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 |
| 20 | 0.12903225806451613 | 0.04938271604938271 | 0.07142857142857142 | 81 |
| 40 | 0.16267942583732056 | 0.19101123595505617 | 0.17571059431524547 | 178 |
| 80 | 0.25280898876404495 | 0.3879310344827586 | 0.30612244897959184 | 232 |
| 160 | 0.475 | 0.3619047619047619 | 0.41081081081081083 | 420 |

Confusion matrix (Oracle rows, prediction columns, order 10/20/40/80/160):

```text
[[  0,   0,   4,   6,   3],
 [  2,   4,  23,  23,  29],
 [  3,   4,  34,  79,  58],
 [  1,   9,  54,  90,  78],
 [  2,  14,  94, 158, 152]]
```

Top-2 accuracy is available here because the linear classifier produces five
comparable softmax scores. This differs from Phase 1 deterministic generated
text, for which top-2 was unavailable.

## Model and heuristic analysis

| Method | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Train-prior majority (class 80) | 0.2510822510822511 | 0.08027681660899653 | 0.10078041911951946 | 0.2 |
| Validation-label majority (class 160; descriptive only) | 0.45454545454545453 | 0.125 | 0.2840909090909091 | 0.2 |
| Maximum-similarity heuristic | 0.049783549783549784 | 0.060034573030522155 | 0.04631400762751346 | 0.25246839439206736 |
| Top-5-mean heuristic | 0.04329004329004329 | 0.05725663186510825 | 0.04116333385531705 | 0.24406902428341146 |
| Tuned penalized-top-5 heuristic | 0.18506493506493507 | 0.14337561538823293 | 0.21885972408993493 | 0.23283692135493764 |
| Tuned leaf-breadth heuristic | 0.37554112554112556 | 0.12417130346260578 | 0.2755908183260398 | 0.1667816091954023 |
| Level-aggregate classifier | 0.3235930735930736 | 0.1891945748453902 | 0.31267712296852007 | 0.20515866865924984 |
| Tree classifier, primary | 0.30303030303030304 | 0.1928144851068439 | 0.3037047064693007 | 0.19804594967839187 |

The tree-minus-level validation macro-F1 difference is
+0.003619910261453696, while accuracy changes by -0.02056277056277056. Grouped
OOF macro-F1 instead favors level-only 0.22533833565985803 over tree
0.22217588936798274. The validation hierarchy gain is therefore too small and
inconsistent to support a strong claim.

## Same-Oracle Phase 2 context

| Metric | Phase 2D | Phase 2E | Phase 3A |
|---|---:|---:|---:|
| Accuracy | 0.36904761904761907 | 0.3484848484848485 | 0.30303030303030304 |
| Macro-F1 | 0.22994524079282935 | 0.22777929657889012 | 0.1928144851068439 |
| Weighted F1 | 0.3644656337102369 | 0.3473258648868964 | 0.3037047064693007 |
| Balanced accuracy | 0.2391812745015638 | 0.24232226137689133 | 0.19804594967839187 |
| Top-2 accuracy | 0.6341991341991342 | 0.6190476190476191 | 0.6136363636363636 |
| Mean joined retrieval F1 | 0.2767166677489178 | 0.2793735097402597 | 0.26773840692640694 |
| Median joined retrieval F1 | 0.2558975 | 0.267412 | 0.25228 |

These methods share the new Oracle and split, so this is a meaningful
development-set comparison. It is not an unbiased final test estimate because
the validation set has been repeatedly examined and was also used in Phase 2E
checkpoint selection. Old-Oracle Logistic Regression/MLP classification
numbers are not included because their target definition differs.

## Retrieval result

All 924 predictions were evaluated through the unchanged downstream pipeline.

| Item | Result |
|---|---:|
| Coverage | 924/924 = 1.0 |
| Mean joined retrieval F1 | 0.26773840692640694 |
| Median joined retrieval F1 | 0.25228 |
| Coverage-adjusted mean | 0.26773840692640705 |
| Wall time | 168.46491879993118 seconds |

Classification accuracy/F1 assess Oracle-label prediction. Joined retrieval
F1 assesses GPT-2-token overlap after routed same-paper retrieval. They must
not be interpreted as the same metric.

## Runtime, recovery, and environment

Audit, extraction, training, and finalization used the unchanged legacy
`.venv`: Python 3.9.12, NumPy 2.0.2, and PyTorch 2.8.0+cpu. The initial gRPC
extraction hit `DEADLINE_EXCEEDED` after 1,844 train rows had been durably
saved. The last pre-failure progress log was 1,827/2,245 at 3,276.9802199
seconds. Extraction resumed without duplication over REST and finished all
splits.

Recorded successful/resumed timings are 827.396431 seconds for the remaining
train extraction, 1311.1521062 seconds for validation extraction,
56.6241827 seconds for training/evaluation, and 168.46491879993118 seconds for
retrieval. Their sum is 2363.6376387 seconds. The exact total including the
failed first attempt is unavailable.

Retrieval used the unchanged `.venv-qwen` because the frozen shared retrieval
module imports `psutil`, absent from `.venv`. No package or environment was
installed, removed, rebuilt, or changed.

## Qdrant and artifact integrity

Qdrant was read-only throughout. `PaperChunk` remained at 1,701,822 points and
`PaperQuestion` at 4,526 points; the complete final collection snapshot equals
the preflight snapshot. No collection or record was created, deleted,
re-indexed, or updated.

The final manifest records 22 pre-summary artifacts with SHA-256 hashes. Core
paths are:

- `outputs/similarity_tree_phase3a_evidence_length_oracle/final_summary.json`;
- `configuration/experiment.json`;
- `integrity/preflight_audit.json` and `integrity/final_audit.json`;
- `features/train_similarity_trees.jsonl.gz` and
  `features/validation_similarity_trees.jsonl.gz`;
- `cross_validation/` and `models/`;
- `validation/predictions.jsonl`;
- `classification/`;
- `heuristics/metrics.json`; and
- `retrieval/results.jsonl` and `retrieval/summary.json`.

## Conclusion and next step

Similarity distributions are learnable enough to outperform a deployable
train-prior constant on both accuracy and macro-F1, but the Phase 3A tree
router is weaker than the Phase 2D/2E Qwen development results and leaves the
rare classes poorly learned. Naive level-selection heuristics fail badly,
showing that raw maximum similarity is not a direct estimator of evidence
length.

The most defensible next experiment is a predeclared multimodal/feature-fusion
study combining question representation and similarity-tree features, with
paper-grouped train-only selection and repeated seeds. A held-out test set not
used in any earlier decision is necessary before a final generalization
claim.

## Reproduction

```powershell
.\.venv\Scripts\python.exe similarity_tree_phase3a.py audit
.\.venv\Scripts\python.exe similarity_tree_phase3a.py extract
.\.venv\Scripts\python.exe similarity_tree_phase3a.py train-evaluate
.\.venv-qwen\Scripts\python.exe similarity_tree_phase3a.py retrieve
.\.venv\Scripts\python.exe similarity_tree_phase3a.py finalize
.\.venv-qwen\Scripts\python.exe -m pytest -q tests\test_similarity_tree_phase3a.py tests\test_router.py
```
