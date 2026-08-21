# Experiment Report: Phase 3C Qwen and Similarity-Tree Fusion

## Executive result

Phase 3C completed Lorenzo's proposed fusion of the Phase 2D Qwen router with
the Phase 3B paper-specific similarity tree. The primary model combines the
five frozen Qwen logits with 173 tree features and uses class-weighted shallow
XGBoost. It was selected over a 1,197-feature hidden-state fusion using the
preserved paper-grouped training folds.

Validation accuracy is 0.31926406926406925, macro-F1 is
0.2306766979333351, weighted F1 is 0.33398724658349993, balanced accuracy is
0.2358503567311523, and top-2 accuracy is 0.5768398268398268. The unchanged
downstream retrieval pipeline achieves mean/median joined F1
0.2872016341991342/0.271346 with 924/924 coverage.

The fusion has the best macro-F1 of Phase 2D, Phase 3B, and Phase 3C, but only
by 0.00073145714050575 over Phase 2D. Its accuracy, weighted F1, balanced
accuracy, and top-2 accuracy are lower than Phase 2D. The clearer gain is
downstream retrieval: +0.0104849664502164 mean joined F1 over Phase 2D and
+0.01548037445887446 over Phase 3B.

## Method

The exact Phase 2D checkpoint (`step-000213`) is frozen in bfloat16 on an A100.
It receives the unchanged token-count instruction plus original question. Its
five logits and 1,024-dimensional final non-padding hidden state are extracted
without gradients. All 924 extracted argmax predictions reproduce the saved
Phase 2D predictions.

Two fusion variants are evaluated:

- five Qwen logits + 173 tree features = 178 features;
- 1,024 Qwen hidden features + 173 tree features = 1,197 features.

Both use Phase 3B's square-root inverse-frequency weights, fixed 12-candidate
XGBoost grid, five paper-grouped folds, and macro-F1 selection. The compact
variant wins OOF macro-F1 0.38383459364081357 versus
0.35436172573956737 and is locked before Phase 3C validation inference.

The frozen Qwen checkpoint was trained on all preserved training questions, so
the fusion OOF scores are not fully nested end-to-end estimates. It was also
previously selected using the same validation split. The reported validation
result is therefore a development result.

## Classification and distributions

| Metric | Phase 3C |
|---|---:|
| Accuracy | 0.31926406926406925 (295/924) |
| Macro-F1 | 0.2306766979333351 |
| Weighted F1 | 0.33398724658349993 |
| Balanced accuracy | 0.2358503567311523 |
| Top-2 accuracy | 0.5768398268398268 |
| Quadratic weighted kappa | 0.16358230799067508 |

Oracle 10/20/40/80/160 counts are 13/81/178/232/420. Predictions are
27/88/251/300/258. Class F1 values are
0.0/0.15384615384615383/0.2890442890442891/0.28571428571428575/
0.4247787610619469. Class 10 remains unsolved, while classes 20 and 40 improve
relative to the component models at the expense of fewer class-160 predictions.

## Component comparison

| Metric | Phase 2D | Phase 3B | Phase 3C |
|---|---:|---:|---:|
| Accuracy | 0.36904761904761907 | 0.329004329004329 | 0.31926406926406925 |
| Macro-F1 | 0.22994524079282935 | 0.2246699873714014 | 0.2306766979333351 |
| Weighted F1 | 0.3644656337102369 | 0.329619858512113 | 0.33398724658349993 |
| Balanced accuracy | 0.2391812745015638 | 0.2327228358766522 | 0.2358503567311523 |
| Top-2 accuracy | 0.6341991341991342 | 0.5898268398268398 | 0.5768398268398268 |
| Mean retrieval F1 | 0.2767166677489178 | 0.27172125974025974 | 0.2872016341991342 |

Classification scores measure exact evidence-length-Oracle prediction.
Retrieval F1 measures joined token overlap after the predicted granularity
drives same-paper top-5 retrieval. They are different outcomes.

## Integrity, runtime, and artifacts

No Qwen parameter changed, no similarity feature was recomputed, and no Qdrant
collection changed. Both XGBoost models reload successfully and reproduce all
labels and metrics. The complete before/after Qdrant snapshots match.

Known sequential stages total 1675.8701351771597 seconds, approximately
27 minutes 56 seconds, excluding unrecorded Qwen model-loading time. Detailed
runtime accounting avoids double-counting CV inside the train-evaluate command.

The source of truth is
`outputs/qwen_phase3c_fusion_evidence_length_oracle/final_summary.json`.
The complete method, commands, comparisons, caveats, and artifact description
are in `docs/QWEN_PHASE3C_FUSION_RESULTS.md`.
