# Phase 3B Results: Class-Balanced Nonlinear Similarity-Tree Router

## Status and research question

Phase 3B is complete. It tests whether Phase 3A was limited by its linear
classifier by training shallow nonlinear XGBoost models over the frozen Phase
3A features. No similarity was recomputed, no Oracle label changed, and no
Phase 3A artifact was overwritten.

The experiment answers two questions:

1. Does a nonlinear model improve over the Phase 3A linear router?
2. Do the 88 explicit parent-child features add value beyond the 85 per-level
   distribution features when the model can learn nonlinear interactions?

The authoritative result is
`outputs/similarity_tree_phase3b_xgboost_evidence_length_oracle/final_summary.json`.

## Frozen inputs and environment

Phase 3B reuses these exact Phase 3A feature files:

| Split | Examples | Papers | SHA-256 |
|---|---:|---:|---|
| Train | 2,245 | 845 | `6d55e1d10872c8db24cf9af9becfb8e2e6570e13a7697151febc7f44ecebdd9c` |
| Validation | 924 | 277 | `548e3cccab3b19dee644eb9858081ff380b6375765433f1d2369c6d7d2ecb893` |

The train and validation papers remain disjoint. The grouped-fold manifest is
byte-identical to Phase 3A, with SHA-256
`584b0e5f9a0a81d05ee7f292ad0c14fae949a2145dd64f76477c710f6268fa2a`.

Training ran locally on CPU in the new isolated `.venv-phase3b` environment:

- Python 3.12.6;
- XGBoost 3.0.2;
- NumPy 2.2.6;
- SciPy 1.15.3;
- `tree_method=hist`, `device=cpu`.

The complete package lock is stored in `environment/package_lock.txt`.
Neither legacy `.venv` nor `.venv-qwen` was modified. The latter was used only
to execute the already frozen retrieval implementation after classification.

## Leakage boundary

The input remains the saved same-paper question-to-chunk score summaries.
Ground-truth evidence, evidence length, answers, evidence embeddings,
evidence-to-chunk similarity, retrieval F1, and Oracle labels are not feature
inputs. The evidence-length Oracle is the target only.

Training did not access Qdrant. Qdrant was contacted only after the primary
classifier was locked, to run the unchanged downstream retrieval evaluation.

## Predeclared models and weighting

| Variant | Features | Purpose |
|---|---:|---|
| `level_xgboost` | 85 | Nonlinear model over per-level score distributions |
| `tree_xgboost` | 173 | Same 85 features plus 88 parent-child features |

Both variants use the five-class `multi:softprob` objective. Each training
fold uses square-root inverse-frequency example weights:

```text
weight(c) = sqrt(maximum fold class count / class count(c))
```

The final all-train weights for classes 10/20/40/80/160 are
3.534248023400323/1.6040678626121678/1.0827534108953252/1.0/
1.0280676421924175. These weights increase minority influence without the
extreme amplification of direct inverse-frequency weighting.

## Train-only model selection

The grid was frozen before training:

- maximum depth: 2/3/4;
- learning rate: 0.03/0.05;
- estimators: 200/400;
- minimum child weight: 5;
- row subsampling: 0.8;
- column subsampling: 0.8;
- L2/L1 regularization: 1.0/0.0;
- seed: 42;
- 12 candidates per feature variant.

Every candidate was evaluated with the same five paper-grouped folds.
Macro-F1 was the primary criterion, followed by balanced accuracy and
accuracy. Simpler settings were preferred only on an exact metric tie.

| Variant | Selected candidate | OOF accuracy | OOF macro-F1 | OOF balanced accuracy | OOF top-2 |
|---|---|---:|---:|---:|---:|
| Level | depth 2, LR 0.03, 200 trees | 0.3051224944320713 | 0.21641959654127185 | 0.22237614125234278 | 0.5755011135857461 |
| Tree | depth 2, LR 0.05, 200 trees | 0.2868596881959911 | 0.2197631383461649 | 0.21850702043132825 | 0.578173719376392 |

The tree variant was locked as primary because its train-only OOF macro-F1 is
higher by 0.0033435418048930465. At the durable selection lock,
`validation_metrics_observed_at_lock` is false. Retrieval was not used for
model or hyperparameter selection.

## Validation ablation

| Metric | Level XGBoost | Tree XGBoost, primary | Tree minus level |
|---|---:|---:|---:|
| Accuracy | 0.29329004329004327 | 0.329004329004329 | +0.035714285714285754 |
| Macro-F1 | 0.18218086837881048 | 0.2246699873714014 | +0.04248911899259092 |
| Weighted F1 | 0.2881468780711505 | 0.329619858512113 | +0.04147298044096248 |
| Balanced accuracy | 0.19528817289827233 | 0.2327228358766522 | +0.037434662978379885 |
| Top-2 accuracy | 0.5800865800865801 | 0.5898268398268398 | +0.009740259740259716 |

The tree variant is better on all five validation classification metrics.
That supports the possibility that nonlinear interactions make the hierarchy
features more useful. The train-only OOF advantage is much smaller, however,
so the larger validation gap must not be treated as a robust effect estimate.

## Primary classification result

| Metric | Phase 3B tree XGBoost |
|---|---:|
| Accuracy | 0.329004329004329 (304/924) |
| Macro-F1 | 0.2246699873714014 |
| Weighted F1 | 0.329619858512113 |
| Balanced accuracy | 0.2327228358766522 |
| Top-2 accuracy | 0.5898268398268398 |
| Mean absolute class distance | 0.9848484848484849 |
| Within-one-level accuracy | 0.7521645021645021 |
| Mean absolute token distance | 53.61471861471861 |
| Quadratic weighted kappa | 0.09084101277786871 |

All 924 predictions are valid softmax outputs. The predicted distribution for
10/20/40/80/160 is 13/32/201/377/301, against Oracle support
13/81/178/232/420.

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.07692307692307693 | 0.07692307692307693 | 0.07692307692307693 | 13 |
| 20 | 0.09375 | 0.037037037037037035 | 0.05309734513274336 | 81 |
| 40 | 0.21890547263681592 | 0.24719101123595505 | 0.2321899736147757 | 178 |
| 80 | 0.26525198938992045 | 0.43103448275862066 | 0.3284072249589491 | 232 |
| 160 | 0.5182724252491694 | 0.37142857142857144 | 0.43273231622746183 | 420 |

Confusion matrix, with Oracle rows and predicted columns ordered
10/20/40/80/160:

```text
[[  1,   0,   3,   7,   2],
 [  3,   3,  18,  30,  27],
 [  3,   8,  44,  78,  45],
 [  2,   8,  51, 100,  71],
 [  4,  13,  85, 162, 156]]
```

Phase 3B is the first of the Phase 3A/3B score-tree models to classify a
validation class-10 example correctly, but class-10 and class-20 performance
remains very weak. It still overpredicts class 80 and underpredicts class 160.

## Phase 3A comparison

| Metric | Phase 3A linear tree | Phase 3B nonlinear tree | Phase 3B minus 3A |
|---|---:|---:|---:|
| Accuracy | 0.30303030303030304 | 0.329004329004329 | +0.025974025974025983 |
| Macro-F1 | 0.1928144851068439 | 0.2246699873714014 | +0.031855502264557495 |
| Weighted F1 | 0.3037047064693007 | 0.329619858512113 | +0.025915152042812306 |
| Balanced accuracy | 0.19804594967839187 | 0.2327228358766522 | +0.03467688619826034 |
| Top-2 accuracy | 0.6136363636363636 | 0.5898268398268398 | -0.023809523809523836 |
| Quadratic weighted kappa | 0.03960842466855796 | 0.09084101277786871 | +0.05123258810931075 |

The nonlinear model improves the primary macro-F1 and most exact/ordinal
metrics. Top-2 accuracy decreases, meaning its second-choice ranking is less
often correct even though its first choice improves.

The feature-importance file shows that several level-distribution statistics
remain dominant. The highest-gain explicit hierarchy feature is
`edge_40_to_20_sibling_abs_gap_mean`; importance is descriptive and does not
establish causality.

## Same-Oracle contextual comparison

| Metric | Phase 2D Qwen | Phase 2E Qwen | Phase 3A | Phase 3B |
|---|---:|---:|---:|---:|
| Accuracy | 0.36904761904761907 | 0.3484848484848485 | 0.30303030303030304 | 0.329004329004329 |
| Macro-F1 | 0.22994524079282935 | 0.22777929657889012 | 0.1928144851068439 | 0.2246699873714014 |
| Weighted F1 | 0.3644656337102369 | 0.3473258648868964 | 0.3037047064693007 | 0.329619858512113 |
| Balanced accuracy | 0.2391812745015638 | 0.24232226137689133 | 0.19804594967839187 | 0.2327228358766522 |
| Top-2 accuracy | 0.6341991341991342 | 0.6190476190476191 | 0.6136363636363636 | 0.5898268398268398 |
| Mean joined retrieval F1 | 0.2767166677489178 | 0.2793735097402597 | 0.26773840692640694 | 0.27172125974025974 |
| Median joined retrieval F1 | 0.2558975 | 0.267412 | 0.25228 | 0.2487165 |

Phase 3B closes most of Phase 3A's macro-F1 gap but remains slightly below
Phase 2D and Phase 2E. It uniquely obtains nonzero class-10 F1 among these
three later methods. This suggests potentially complementary errors, but does
not by itself establish that a fusion model will improve.

These are development-set comparisons. The same validation split was observed
in prior phases and selected among Phase 2E checkpoints. It is not an unbiased
final test set. Old retrieval-F1-Oracle Logistic Regression/MLP classification
results remain not directly comparable.

## Downstream retrieval

The train-locked primary predictions were evaluated with unchanged same-paper
retrieval: predicted granularity, `top_k=5`, existing
`text-embedding-3-small` vectors, cosine ordering, original chunk ordering and
concatenation, and joined GPT-2-token F1.

| Retrieval item | Result |
|---|---:|
| Coverage | 924/924 = 1.0 |
| Mean joined retrieval F1 | 0.27172125974025974 |
| Median joined retrieval F1 | 0.2487165 |
| Coverage-adjusted full-set mean | 0.27172125974025974 |
| Retrieval wall time | 314.059133999981 seconds |

Relative to Phase 3A, mean retrieval F1 increases by
0.0039828528138528 while median retrieval F1 decreases by
0.0035634999999999972. Classification improvement therefore does not translate
uniformly into retrieval improvement across questions.

## Runtime and integrity

| Recorded computational stage | Seconds |
|---|---:|
| Level XGBoost cross-validation | 197.80893129995093 |
| Tree XGBoost cross-validation | 492.87182669946924 |
| Selection lock, final fits, and validation recovery invocation | 7.2022769001778215 |
| Retrieval | 314.059133999981 |
| Known recorded stage sum | 1011.942168899579 |

The recorded stage sum is approximately 16 minutes 52 seconds. Interactive
audit and orchestration overhead are excluded rather than estimated.

The saved primary XGBoost model was independently reloaded. All 924 identities,
probabilities, labels, and recomputed metrics match exactly; maximum absolute
probability difference is 0.0. Qdrant's complete before/after snapshots match,
and no collection was created, deleted, rebuilt, re-indexed, or updated.

## Interpretation

Phase 3B supports the hypothesis that the Phase 3A linear model was too simple
for some of the feature interactions. A shallow nonlinear classifier improves
macro-F1 by approximately 0.032 and also improves balanced accuracy and
ordinal agreement. The hierarchy variant wins train-only OOF macro-F1 and is
materially stronger than level-only XGBoost on validation.

The limitation is not fully resolved. OOF hierarchy advantage is only 0.00334,
minority recall remains poor, accuracy remains below the descriptive
validation-majority baseline 0.45454545454545453, and Phase 2D/2E remain
slightly stronger. The evidence-length target is still only partially
observable from semantic-score distributions.

A defensible next step is Phase 3C fusion of question-text/Qwen information
with score-tree information. It must use out-of-fold Qwen representations or
a joint architecture; using in-sample Phase 2D train logits in a stacking
model would leak training information. Repeated paper-grouped seeds and an
untouched final test set are required for a generalization claim.

## Artifacts and reproduction

Core artifacts under
`outputs/similarity_tree_phase3b_xgboost_evidence_length_oracle/` include:

- `final_summary.json`;
- `configuration/experiment.json`;
- `environment/python_environment.json` and `environment/package_lock.txt`;
- `integrity/preflight_audit.json` and `integrity/final_audit.json`;
- `selection/selection_lock.json`;
- both complete 12-candidate records under `cross_validation/`;
- both XGBoost models and metadata under `models/`;
- both feature-importance CSV files;
- per-variant and canonical predictions under `validation/`;
- classification metrics, confusion matrix, and histogram;
- 924 retrieval records and their summary; and
- `runtime/summary.json`.

Clean reproduction into a separate output root:

```powershell
py -3.12 -m venv .venv-phase3b
.\.venv-phase3b\Scripts\python.exe -m pip install -r requirements-phase3b.txt
.\.venv-phase3b\Scripts\python.exe similarity_tree_phase3b.py --output-root outputs\similarity_tree_phase3b_reproduction audit
.\.venv-phase3b\Scripts\python.exe similarity_tree_phase3b.py --output-root outputs\similarity_tree_phase3b_reproduction train-evaluate
.\.venv-qwen\Scripts\python.exe similarity_tree_phase3b.py --output-root outputs\similarity_tree_phase3b_reproduction retrieve
.\.venv-phase3b\Scripts\python.exe similarity_tree_phase3b.py --output-root outputs\similarity_tree_phase3b_reproduction finalize
.\.venv-phase3b\Scripts\python.exe -m pytest -q tests\test_similarity_tree_phase3b.py
```
