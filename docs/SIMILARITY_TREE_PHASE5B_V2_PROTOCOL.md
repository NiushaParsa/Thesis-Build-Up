# Phase 5B-v2 — Frozen protocol

Phase 5B-v1 completed its 3,025 zero-shot Qwen generations but was stopped
before classifier training because 231 outputs did not contain one of the three
allowed context labels. Training-only inspection showed that Qwen was directly
answering many yes/no questions. No invalid output was mapped to a default.

Phase 5B-v2 is a separate format-repair run. It does not overwrite or reinterpret
the v1 outputs. Its revised prompt is motivated only by the v1 training failure
mode and explicitly tells Qwen not to answer the question. A training-only check
on the 175 v1 failures corrected 173 cases but still produced two direct answers.
The final procedure therefore restricts the Qwen decision to the three allowed
labels. No validation example is used to select or revise the v2 procedure.

## Frozen Qwen decision

- Model: `Qwen/Qwen3.5-0.8B`.
- Revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Transformers commit: `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`.
- Qwen remains frozen and task-untrained; parameter updates, gradients,
  optimizers, adapters, and fine-tuning are all zero/absent.
- Input remains only the fixed instruction and original question.
- The three labels are verified single tokenizer tokens: `short` = 8412,
  `medium` = 25252, and `long` = 4670. One forward pass obtains the next-token
  logits and the highest-logit allowed label is selected.
- The decision is deterministic and categorical. This is constrained label
  selection, not free-form answer generation and not a post-hoc default.
- The parser and invalid-output gate remain unchanged. The selected token is
  parsed and must still validate before classifier training.
- The category is still one-hot encoded as three states: `short`, `medium`, and
  `long`.

The fixed instruction is:

> You are not being asked to answer the question. Your only task is to classify
> how much supporting context would be needed to answer it. Respond with exactly
> one lowercase label from: short, medium, long. Do not output yes, no, the answer
> to the question, an explanation, or punctuation. Output one label only.

## Unchanged tree experiment

Everything after Qwen generation remains inherited from Phase 5A and Phase
5B-v1: 4,081 gold-overlap training trees, local per-tree labels, 173 similarity
features, three Qwen one-hot features, the fixed 176-feature XGBoost classifier,
class weights, paper-grouped diagnostic folds, TreeScore, top five trees, one
most-similar chunk at the predicted level per tree, and joined GPT-2 token-level
precision/recall/F1.

Outputs are isolated under:

`outputs/similarity_tree_phase5b_v2_zero_shot_qwen_context_feature/`

Remote Qwen stage:

```bash
.venv-phase5b-gpu/bin/python similarity_tree_phase5b_v2.py smoke --count 3
.venv-phase5b-gpu/bin/python similarity_tree_phase5b_v2.py qwen-infer
```

Local classifier and retrieval stage:

```powershell
.\.venv-phase5a\Scripts\python.exe similarity_tree_phase5b_v2.py train-evaluate
.\.venv-phase5a\Scripts\python.exe -m pytest tests/test_similarity_tree_phase5b.py tests/test_similarity_tree_phase5b_v2.py -q
```
