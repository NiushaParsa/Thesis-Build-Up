# Phase 5B-v2 — Tree-Local Similarity Router with Constrained Zero-Shot Qwen Context Feature

## Method

Frozen zero-shot `Qwen/Qwen3.5-0.8B` produces one question-level context indication:
`short`, `medium`, or `long`. The result is one-hot encoded into three features
and appended to each tree's 173 Phase 5A similarity features. The same Qwen
value is repeated across a question's trees; it is auxiliary information, not
a granularity decision. Qwen receives only the fixed instruction and original
question and is never fitted on the thesis data, so OOF Qwen inference is not
needed.

The 4,081 Phase 5A gold-overlap training trees and their local labels are
unchanged. XGBoost settings, class weights, paper-grouped diagnostic folds,
TreeScore, top-five tree selection, one-chunk-per-tree rule, and joined metric
are also unchanged. The v2 prompt and constrained categorical decision repaired a format failure observed only in Phase 5B-v1 training outputs; no validation example informed the repair, and no classifier hyperparameter search was performed.

## Qwen feature distributions

| Split | Short | Medium | Long | Invalid |
|---|---:|---:|---:|---:|
| Train (2101) | 24 | 2076 | 1 | 0 |
| Validation (924) | 12 | 912 | 0 | 0 |

## Results

| Method | Mean joined F1 |
|---|---:|
| Phase 5B-v2 Qwen + local-tree features | 0.304536 |
| Phase 5A local-tree features only | 0.305266 |
| Same-tree fixed 40 | 0.307462 |

Phase 5B-v2 minus Phase 5A is -0.000730, with
paired paper-cluster bootstrap 95% CI
[-0.002363, 0.000880].
Phase 5B-v2 minus same-tree fixed 40 is -0.002925,
with 95% CI [-0.006459,
0.000693].

Phase 5B-v2 mean precision is 0.273905, mean recall
is 0.511996, and median joined F1 is
0.301139. Retrieval covers all 924 questions.

The 4,620 selected validation-tree predictions are: 10=247,
20=506, 40=2804, 80=1008, and
160=55.

The Qwen category is strongly concentrated on `medium`: this label covers
98.81% of training questions and
98.70% of validation
questions, while validation contains no `long` prediction. Both paired confidence
intervals above include zero, so the auxiliary Qwen feature does not show a reliable
retrieval improvement over Phase 5A or the matched fixed-40 strategy.

Secondary local-tree classification accuracy is 0.281099
and macro-F1 is 0.212327. This diagnostic measures the
local evidence-length label; joined retrieval F1 remains the operational metric.

## Integrity

- Validation was not used for training, model selection, prompt selection, or thresholding.
- Predictions were saved and hashed before validation evidence was requested for evaluation.
- Qdrant was read-only and its before/after snapshots matched.
- Phase 5A and all earlier source artifacts remained hash-identical.
- Qwen used `cuda:0` with `torch.bfloat16`, with zero gradients,
  optimizers, backward passes, or parameter updates.
- The result is a development result because validation was reused in earlier phases.

## Reproduction

```powershell
.\.venv-qwen\Scripts\python.exe similarity_tree_phase5b_v2.py qwen-infer
# Remote CUDA alternative: .venv-phase5b-gpu/bin/python similarity_tree_phase5b_v2.py qwen-infer
.\.venv-phase5a\Scripts\python.exe similarity_tree_phase5b_v2.py train-evaluate
.\.venv-phase5a\Scripts\python.exe -m pytest tests/test_similarity_tree_phase5b.py tests/test_similarity_tree_phase5b_v2.py -q
```

Artifacts: `outputs/similarity_tree_phase5b_v2_zero_shot_qwen_context_feature`.
