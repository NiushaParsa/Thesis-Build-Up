# Phase 4 experiment report — retrieval-utility-aware routing

## Research question

Can the combined Qwen/question and same-paper similarity-tree representation
choose chunk granularity more effectively when trained against the retrieval
utility of all five actions, rather than against a hard evidence-length class?

Phase 4 treats granularity selection as a cost-sensitive decision problem. For
question `q` and action `g`, the saved utility `U(q,g)` is joined token-level F1
after unchanged top-5 source-paper retrieval. Cost is regret relative to the
best saved action for that question:

`C(q,g) = max_h U(q,h) - U(q,g)`.

Five regressors estimate conditional action regret. The deployed action is the
one with minimum predicted regret. This plug-in decision rule directly targets
expected retrieval regret and retains the magnitude of the difference between
actions. It is not another evidence-length Oracle and is not conventional
five-class cross-entropy training.

## Inputs and integrity

The experiment reused, without recomputation:

- 2,245 preserved training questions from 845 papers;
- 924 preserved validation questions from 277 papers;
- five paper-grouped OOF Phase 2D Qwen logits for every training question;
- five frozen all-train-refit Qwen logits for every validation question;
- 173 inference-safe Phase 3B tree features per question;
- the fixed-separate five-granularity top-5 retrieval evaluations.

The original train utility artifact contained 2,223 complete questions. Its
documented recovery artifact contains the remaining 22, giving 2,245 unique
question IDs and exactly five utilities per question. Validation contains
924 unique complete utility vectors. Question and document IDs align exactly
with the Phase 3A/3B feature rows. All utility values are finite and in [0,1].

Frozen hashes include:

- train tree features:
  `6d55e1d10872c8db24cf9af9becfb8e2e6570e13a7697151febc7f44ecebdd9c`;
- validation tree features:
  `548e3cccab3b19dee644eb9858081ff380b6375765433f1d2369c6d7d2ecb893`;
- train OOF logits:
  `ee73c962d45b86034da7dfa903f9ff9ce92abd5707c6cec3b49a6ca7d1f4115d`;
- validation full-refit logits:
  `721e9bcdcf2f026e458e504ced14ccfc52d3031c2694b5e4e4717c146c731233`;
- train utility main/recovery:
  `1308a4188965e7afa6d84820c8f3be5dd83acb7998e9cf4af4876321519f9da6` /
  `53cc9375d3e5b934a396fee6be2e407dee63b2350e7b05d0f3a0ca66454e9c71`;
- validation utility:
  `92fffae69b6e33f37ea55f2d05593eccd7045797603bd3131ca47c7cd5de7bed`.

No Qwen inference, embedding request, Qdrant query, collection change, Oracle
generation, or retrieval rerun occurred.

## Leakage controls and frozen procedure

Training Qwen features remain the methodologically clean paper-grouped OOF
logits created for Phase 3C-OOF: the Qwen model producing a training row never
saw that question or its source paper. Validation Qwen features come from the
separate fixed full-training refit.

Phase 4 used no hyperparameter grid. Its XGBoost capacity and regularization
were inherited from the already frozen clean Phase 3C settings: depth 2,
learning rate 0.05, 200 rounds, minimum child weight 5, row/column subsampling
0.8, L2 1, and L1 0. Five independent `reg:squarederror` models predict regret
for 10, 20, 40, 80, and 160. Exact prediction ties select the smaller action.

A five-fold paper-grouped training diagnostic was recorded but selected
nothing. It is not a fully nested stacking estimate because Qwen models that
produced some meta-training rows may have seen papers in the current
meta-held-out fold. Avoiding that issue for model selection would require
nested base-model cross-fitting. Phase 4 avoids the need by predeclaring one
fixed procedure and treating the diagnostic as descriptive only.

The procedure lock was saved before validation prediction. The 924 predictions
were then saved and hashed before the validation utility artifact was opened.
Validation gold utility therefore influenced only final scoring. The validation
set had been used in earlier phases, so this is a development result rather
than an unbiased final test estimate.

## Training utility structure

Mean training utility for fixed 10/20/40/80/160 is
0.225549/0.277659/0.283775/0.239040/0.170655. The smaller-tie retrieval-optimal
action distribution is 463/672/686/343/81. Thus training utility is much less
imbalanced than the evidence-length Oracle, but fixed 40 has the highest mean
training utility.

The descriptive grouped diagnostic selects 10/20/40/80/160 for
16/944/1,275/10/0 questions. It obtains mean joined F1 0.283326, mean regret
0.063541, and retrieval-optimal-any-tie agreement 0.316258. These values are
not used as a model-selection claim.

## Final development results

Phase 4 selects 10/20/40/80/160 for 4/359/561/0/0 validation questions. The
retrieval-optimal smaller-tie distribution is 138/235/282/187/82.

| Method | Mean joined F1 | Median joined F1 | Mean regret | 95% CI for mean F1 |
|---|---:|---:|---:|---:|
| Retrieval Oracle upper bound | 0.382073 | 0.383553 | 0.000000 | [0.371677, 0.392311] |
| Fixed 40 | 0.315856 | 0.309168 | 0.066217 | [0.305815, 0.326028] |
| **Phase 4** | **0.307506** | **0.300849** | **0.074567** | **[0.297930, 0.317065]** |
| Original Phase 3C | 0.287202 | 0.271346 | 0.094872 | [0.276382, 0.297862] |
| Clean Phase 3C-OOF | 0.278303 | 0.255051 | 0.103770 | [0.267009, 0.289508] |
| Phase 2D | 0.276717 | 0.255898 | 0.105357 | [0.265546, 0.287638] |
| Phase 3B | 0.271721 | 0.248717 | 0.110352 | [0.260402, 0.282990] |

Paired paper-cluster bootstrap differences, with 10,000 resamples and seed 42:

| Contrast | Difference | 95% CI |
|---|---:|---:|
| Phase 4 − Phase 2D | +0.030790 | [+0.021736, +0.040034] |
| Phase 4 − Phase 3B | +0.035785 | [+0.026634, +0.044826] |
| Phase 4 − original Phase 3C | +0.020305 | [+0.011623, +0.028768] |
| Phase 4 − clean Phase 3C-OOF | +0.029203 | [+0.019873, +0.038640] |
| Phase 4 − fixed 40 | −0.008350 | [−0.012661, −0.004156] |

Phase 4 selects an action tied for best observed retrieval on 280/924 questions
(0.303030). Its median regret is 0.0476235. It materially improves on all
previous learned routers in this comparison, but the paired interval against
fixed 40 is entirely negative. The correct operational conclusion is therefore
that utility-aware learning recovers much of the gap, but still does not beat
the strongest fixed policy.

The prediction collapse is different from the evidence-length classifiers:
Phase 4 concentrates on the globally strong 20/40 actions and never selects 80
or 160. This protects mean F1 from the weak fixed-160 utility but fails to
capture question-specific tail cases. The remaining gap to the retrieval
upper bound is 0.074567 mean F1, indicating substantial unrealized conditional
action value.

## Secondary evidence-length diagnostics

Evidence-length-label accuracy/macro-F1/weighted-F1/balanced accuracy are
0.141775/0.106033/0.065376/0.205855. The evidence-length Oracle distribution
is 13/81/178/232/420, while Phase 4 predicts 4/359/561/0/0. These metrics are
not the Phase 4 objective. Their weakness reinforces the earlier finding that
evidence-length agreement and downstream retrieval utility are different
quantities.

## Runtime and reproducibility

The experiment ran locally on CPU in a new ignored `.venv-phase4` using Python
3.10.7, NumPy 2.0.2, SciPy 1.15.3, and XGBoost 3.0.2. The original `.venv` and
`.venv-qwen` were unchanged. Fit/prediction/diagnostic/total recorded times are
12.1663/0.0631/52.0300/94.0270 seconds. Eight focused tests pass.

Commands:

```powershell
py -3.10 -m venv .venv-phase4
.\.venv-phase4\Scripts\python.exe -m pip install -r requirements-phase4.txt
.\.venv-phase4\Scripts\python.exe -m pytest tests/test_qwen_phase4_expected_regret.py -q
.\.venv-phase4\Scripts\python.exe qwen_phase4_expected_regret.py audit
.\.venv-phase4\Scripts\python.exe qwen_phase4_expected_regret.py run
```

The authoritative result is
`outputs/qwen_phase4_expected_regret_retrieval_utility/final_summary.json`.
