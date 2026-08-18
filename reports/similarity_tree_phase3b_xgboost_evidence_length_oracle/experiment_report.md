# Experiment Report: Phase 3B Nonlinear Similarity-Tree Router

## Executive result

Phase 3B completed the planned class-balanced nonlinear follow-up to Phase 3A.
It reused the frozen 85 level and 173 hierarchy feature sets and compared two
shallow XGBoost classifiers through five-fold paper-grouped train-only model
selection. The 173-feature tree variant won train-only OOF macro-F1 and was
durably locked before validation.

The primary validation result is accuracy 0.329004329004329, macro-F1
0.2246699873714014, weighted F1 0.329619858512113, balanced accuracy
0.2327228358766522, and top-2 accuracy 0.5898268398268398. Mean/median joined
retrieval F1 is 0.27172125974025974/0.2487165 at 924/924 coverage.

This is a genuine improvement over Phase 3A's primary linear model on macro-F1
and most exact/ordinal metrics, but it remains slightly below the Phase 2D and
Phase 2E Qwen classifiers. It is a positive but limited result rather than a
solution to the evidence-length learnability problem.

## Experimental controls

- No Phase 3A extraction was rerun.
- Train/validation feature SHA-256 values are
  `6d55e1d10872c8db24cf9af9becfb8e2e6570e13a7697151febc7f44ecebdd9c`
  and `548e3cccab3b19dee644eb9858081ff380b6375765433f1d2369c6d7d2ecb893`.
- The 2,245/924 examples and 845/277 papers remain unchanged and disjoint.
- The exact Phase 3A paper-grouped folds were reproduced.
- Evidence, answers, evidence length, evidence-derived similarity, retrieval
  F1, and Oracle label were not features.
- Qdrant was not contacted during training or model selection.
- Retrieval was not used for selection.
- Both protected environments remained unchanged.

The isolated CPU training environment is `.venv-phase3b`, Python 3.12.6,
XGBoost 3.0.2, NumPy 2.2.6, and SciPy 1.15.3. Its complete package lock is an
artifact.

## Models and selection

Both models use `multi:softprob` with square-root inverse-frequency training
weights. The fixed 12-candidate grid crosses depths 2/3/4, learning rates
0.03/0.05, and 200/400 trees; minimum child weight 5, subsample 0.8,
column-subsample 0.8, L2 1.0, and L1 0.0 remain fixed.

| Variant | Features | Selected setting | OOF accuracy | OOF macro-F1 | OOF balanced accuracy |
|---|---:|---|---:|---:|---:|
| Level | 85 | depth 2, LR 0.03, 200 | 0.3051224944320713 | 0.21641959654127185 | 0.22237614125234278 |
| Tree | 173 | depth 2, LR 0.05, 200 | 0.2868596881959911 | 0.2197631383461649 | 0.21850702043132825 |

The tree model's OOF macro-F1 advantage is 0.0033435418048930465. It was
selected solely for that predeclared primary metric. Its OOF accuracy and
balanced accuracy are lower, which is an important tradeoff rather than a
hidden result.

## Validation results

| Metric | Level XGBoost | Tree XGBoost |
|---|---:|---:|
| Accuracy | 0.29329004329004327 | 0.329004329004329 |
| Macro-F1 | 0.18218086837881048 | 0.2246699873714014 |
| Weighted F1 | 0.2881468780711505 | 0.329619858512113 |
| Balanced accuracy | 0.19528817289827233 | 0.2327228358766522 |
| Top-2 accuracy | 0.5800865800865801 | 0.5898268398268398 |

The primary predicted counts are 13/32/201/377/301 for
10/20/40/80/160. Oracle counts are 13/81/178/232/420.

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.07692307692307693 | 0.07692307692307693 | 0.07692307692307693 | 13 |
| 20 | 0.09375 | 0.037037037037037035 | 0.05309734513274336 | 81 |
| 40 | 0.21890547263681592 | 0.24719101123595505 | 0.2321899736147757 | 178 |
| 80 | 0.26525198938992045 | 0.43103448275862066 | 0.3284072249589491 | 232 |
| 160 | 0.5182724252491694 | 0.37142857142857144 | 0.43273231622746183 | 420 |

```text
[[  1,   0,   3,   7,   2],
 [  3,   3,  18,  30,  27],
 [  3,   8,  44,  78,  45],
 [  2,   8,  51, 100,  71],
 [  4,  13,  85, 162, 156]]
```

The class-160 validation majority has descriptive accuracy
0.45454545454545453 and macro-F1 0.125. Phase 3B remains below it on accuracy
and above it on macro-F1. The deployable train-majority class-80 reference has
accuracy 0.2510822510822511 and macro-F1 0.08027681660899653; Phase 3B exceeds
both.

## Phase 3A and Qwen context

Compared with the Phase 3A tree model, Phase 3B changes accuracy by
+0.025974025974025983, macro-F1 by +0.031855502264557495, weighted F1 by
+0.025915152042812306, balanced accuracy by +0.03467688619826034, and
quadratic weighted kappa by +0.05123258810931075. Top-2 accuracy changes by
-0.023809523809523836.

Phase 3B macro-F1 0.2246699873714014 remains below Phase 2D
0.22994524079282935 and Phase 2E 0.22777929657889012. The difference is small,
but all comparisons remain descriptive development-set comparisons because
the 924-example validation set has been repeatedly observed.

## Retrieval

The unchanged same-paper `top_k=5` pipeline completed with 100% coverage:

| Metric | Result |
|---|---:|
| Mean joined retrieval F1 | 0.27172125974025974 |
| Median joined retrieval F1 | 0.2487165 |
| Coverage-adjusted mean | 0.27172125974025974 |
| Wall time | 314.059133999981 seconds |

Mean retrieval F1 improves over Phase 3A by 0.0039828528138528; median falls
by 0.0035634999999999972. Phase 2D and 2E remain higher on both the reported
mean and median retrieval metrics.

Classification metrics measure evidence-length-Oracle label prediction.
Joined retrieval F1 measures token overlap after the prediction drives
retrieval. They are distinct outcomes.

## Runtime and integrity

The durable candidate timings total 197.80893129995093 seconds for level
features and 492.87182669946924 seconds for tree features. The recovered
selection/final-fit/validation invocation took 7.2022769001778215 seconds, and
retrieval took 314.059133999981 seconds. The known recorded computational sum
is 1011.942168899579 seconds, approximately 16 minutes 52 seconds. Audit and
interactive orchestration overhead are not fabricated into that sum.

The final model reload reproduced every probability, label, and metric exactly
with maximum probability difference 0.0. The primary model SHA-256 is
`3c44a6b1290532295ec74528063c8b4f23dd82bd381bf18f002c37db0c08d801`.
The primary prediction SHA-256 is
`d997c9d2363c2b909af0228f9d9f742377c2cc812020834203858ee1a762105b`.

Qdrant's complete before/after snapshots are equal. `PaperChunk` remains at
1,701,822 points and `PaperQuestion` at 4,526 points. No collection or record
was created, deleted, rebuilt, re-indexed, or changed.

## Conclusion

The Phase 3A classifier was partly capacity-limited: shallow nonlinear trees
extract more useful signal from the same frozen features. The 173-feature
variant also beats the 85-feature nonlinear ablation on validation and wins
train-only OOF macro-F1, supporting continued study of hierarchy information.

The remaining OOF advantage is small, minority recall is poor, and Qwen
Phases 2D/2E remain slightly stronger. Phase 3B therefore motivates a carefully
leakage-controlled fusion experiment rather than a claim that nonlinear score
trees solve the task.

## Artifacts

The global authority is
`outputs/similarity_tree_phase3b_xgboost_evidence_length_oracle/final_summary.json`.
Supporting records include `selection/selection_lock.json`, both
`cross_validation/*.json` model-selection files, `models/`,
`feature_importance/`, `validation/`, `classification/`, `retrieval/`,
`runtime/`, `environment/`, and `integrity/`.

Detailed commands and the complete artifact inventory are recorded in
`docs/SIMILARITY_TREE_PHASE3B_RESULTS.md`.
