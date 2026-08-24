# Phase 4 — retrieval-utility-aware expected-regret routing

## Objective

Phase 4 changes the learning target from the evidence-length Oracle to the
actual downstream utility of each available routing action. For every question
and each granularity in 10, 20, 40, 80, and 160, utility is the frozen joined
token-level F1 produced by the unchanged source-paper-restricted top-5
retrieval pipeline. The per-action cost is retrieval regret:

`C(q,g) = max_h U(q,h) - U(q,g)`

The model estimates the conditional regret of every action and selects the
granularity with minimum predicted regret. Multiple granularities with the
same maximum observed utility all have zero target regret; the formulation
therefore does not force an arbitrary hard label between equally good actions.

## Frozen method

- Training examples/papers: 2,245 / 845.
- Validation examples/papers: 924 / 277.
- Inputs: the five clean paper-grouped OOF Qwen logits from Phase 3C-OOF plus
  the unchanged 173 inference-safe similarity-tree features, for 178 features.
- Validation Qwen inputs: logits from the frozen all-training-data Phase 3C-OOF
  refit.
- Utility source: the preserved fixed-separate evaluation with top-k 5,
  source-paper restriction, `text-embedding-3-small`, cosine similarity, and
  joined token-level retrieval F1.
- Training utility coverage: 2,223 rows from the original train evaluation and
  the 22 completed recovery rows, giving 2,245 complete five-action vectors.
- Learner: five independent XGBoost squared-error regressors, one per action.
- Decision rule: `argmin_g predicted_E[C(q,g) | x]`.
- Exact predicted-regret tie: smaller granularity.
- Fixed parameters inherited from clean Phase 3C-OOF: depth 2, learning rate
  0.05, 200 trees, minimum child weight 5, subsample 0.8, column subsample
  0.8, L2 1, and L1 0.
- Phase 4 hyperparameter search: none.
- Seed/device: 42 / CPU.

The paper-grouped training-fold result is descriptive only. It did not select
the model or any setting because globally cross-fitted base logits are not a
fully nested stacking representation for meta-level cross-validation. The
fixed procedure was written to a durable selection lock before validation
prediction. Validation predictions were then saved and SHA-256 hashed before
the validation gold-utility records were opened.

Gold evidence affects the training data only through the five training utility
targets. Validation gold evidence/utility was used only after predictions were
frozen, to calculate final retrieval F1, regret, retrieval-optimal agreement,
and secondary evidence-length-label diagnostics. No retrieval, embedding, or
Qdrant operation was rerun.

## Retrieval-utility results

| Method | Mean joined F1 | Median joined F1 | Mean regret | Median regret |
|---|---:|---:|---:|---:|
| Retrieval Oracle upper bound | 0.382073 | 0.383553 | 0.000000 | 0.000000 |
| Fixed 40 | 0.315856 | 0.309168 | 0.066217 | 0.041771 |
| **Phase 4 expected regret** | **0.307506** | **0.300849** | **0.074567** | **0.047624** |
| Original Phase 3C | 0.287202 | 0.271346 | 0.094872 | 0.068895 |
| Fixed 20 | 0.288750 | 0.280918 | 0.093324 | 0.053509 |
| Clean Phase 3C-OOF | 0.278303 | 0.255051 | 0.103770 | 0.081587 |
| Phase 2D | 0.276717 | 0.255898 | 0.105357 | 0.082052 |
| Phase 3B | 0.271721 | 0.248717 | 0.110352 | 0.087627 |

The Phase 4 mean joined-F1 paper-cluster bootstrap 95% interval is
`[0.297930, 0.317065]`. Paired differences are:

| Contrast | Mean F1 difference | Paired paper-cluster 95% CI |
|---|---:|---:|
| Phase 4 − Phase 2D | +0.030790 | [+0.021736, +0.040034] |
| Phase 4 − Phase 3B | +0.035785 | [+0.026634, +0.044826] |
| Phase 4 − original Phase 3C | +0.020305 | [+0.011623, +0.028768] |
| Phase 4 − clean Phase 3C-OOF | +0.029203 | [+0.019873, +0.038640] |
| Phase 4 − fixed 40 | **−0.008350** | **[−0.012661, −0.004156]** |

All intervals use 10,000 bootstrap samples with the paper, not the question, as
the resampling unit. Phase 4 is clearly better than the previous learned
routers on the saved development set. It remains worse than always selecting
40, so it does not yet establish an operational benefit for adaptive routing.

The router selected 10/20/40/80/160 for 4/359/561/0/0 validation questions.
The retrieval-optimal smaller-tie distribution is 138/235/282/187/82. Phase 4
selected a retrieval-optimal action for 280/924 questions (0.303030). Its
concentration on 20 and 40 explains both its strong improvement over previous
learned routers and its inability to exceed fixed 40: it learns the globally
strong middle granularities but does not reliably identify the questions for
which 10, 80, or 160 is optimal.

## Secondary evidence-length-label diagnostics

These values do not measure the Phase 4 training objective. They are included
only to maintain continuity with earlier experiments:

| Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy |
|---:|---:|---:|---:|
| 0.141775 | 0.106033 | 0.065376 | 0.205855 |

The predicted evidence-length-class distribution is 4/359/561/0/0, while the
evidence-length Oracle distribution is 13/81/178/232/420. Low agreement is not
surprising because Phase 4 is optimized for retrieval regret, not annotated
evidence length. Its retrieval performance, not this classification table, is
the primary result.

## Runtime and environment

- Environment: `.venv-phase4` (new parallel environment; not committed).
- Python: 3.10.7.
- NumPy/SciPy/XGBoost: 2.0.2 / 1.15.3 / 3.0.2.
- Device: CPU; GPU and Vast.ai were not used.
- Fixed-procedure fit: 12.1663 seconds.
- Validation prediction: 0.0631 seconds.
- Descriptive grouped diagnostic: 52.0300 seconds.
- Total recorded wall time: 94.0270 seconds.
- Qdrant/retrieval rerun: none.

The original `.venv` and `.venv-qwen` were not modified.

## Reproduction

From the repository root in PowerShell:

```powershell
py -3.10 -m venv .venv-phase4
.\.venv-phase4\Scripts\python.exe -m pip install -r requirements-phase4.txt
.\.venv-phase4\Scripts\python.exe -m pytest tests/test_qwen_phase4_expected_regret.py -q
.\.venv-phase4\Scripts\python.exe qwen_phase4_expected_regret.py audit
.\.venv-phase4\Scripts\python.exe qwen_phase4_expected_regret.py run
```

Authoritative machine-readable result:
`outputs/qwen_phase4_expected_regret_retrieval_utility/final_summary.json`.
The detailed experiment report is
`reports/qwen_phase4_expected_regret_retrieval_utility/experiment_report.md`.
