# Phase 5B — Frozen protocol

Phase 5B is a new experiment and does not modify Phase 5A or any earlier
result. It tests Lorenzo's final proposal: add a question-level indication
from frozen zero-shot Qwen to the independently classified local chunk trees.

## Frozen design

- Qwen model: `Qwen/Qwen3.5-0.8B`, revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Qwen is not task-fine-tuned and receives only a fixed instruction plus the
  original question.
- Deterministic generation returns one of `short`, `medium`, or `long`.
- The parsed category is encoded using three one-hot features. No ordinal
  numerical relationship is imposed between the categories.
- The same three question-level values are appended to each local tree for
  that question, giving 176 inputs: 173 Phase 5A similarity features plus 3
  Qwen features.
- Invalid or ambiguous Qwen outputs are not mapped to a default. They stop the
  classifier stage and are reported.
- OOF Qwen inference is unnecessary because this frozen Qwen model is never
  fitted on the thesis training questions.

Everything else is inherited unchanged from Phase 5A: the 4,081 training
trees, local gold-overlap labels, XGBoost settings, class weighting,
paper-grouped diagnostic folds, TreeScore, top-five trees, one most-similar
chunk per predicted level, and joined GPT-2 token-level retrieval metric.
There is no Phase 5B prompt or hyperparameter search.

## Isolation and execution

The Qwen stage uses either the existing local `.venv-qwen` without changing it,
or an isolated remote `.venv-phase5b-gpu` with Python 3.10.7, PyTorch
2.8.0+cu128, and the same frozen Transformers commit. CPU and CUDA execution
both use `torch.bfloat16`; the compute backend is recorded in the Qwen model and
environment artifacts. The tree stage uses the existing `.venv-phase5a`
without changing it. Outputs are saved only under:

`outputs/similarity_tree_phase5b_zero_shot_qwen_context_feature/`

Commands:

```powershell
.\.venv-phase5a\Scripts\python.exe -m pytest tests/test_similarity_tree_phase5b.py -q
.\.venv-qwen\Scripts\python.exe similarity_tree_phase5b.py smoke --count 3
.\.venv-qwen\Scripts\python.exe similarity_tree_phase5b.py qwen-infer
.\.venv-phase5a\Scripts\python.exe similarity_tree_phase5b.py train-evaluate
```

Remote Qwen-only CUDA alternative:

```bash
.venv-phase5b-gpu/bin/python similarity_tree_phase5b.py smoke --count 3
.venv-phase5b-gpu/bin/python similarity_tree_phase5b.py qwen-infer
```

Qwen output generation is incremental and resumable. Validation predictions
will be saved and hashed before validation evidence is requested for retrieval
evaluation. Qdrant access in the evaluation stage is read-only.

## Current execution note

The first local smoke attempt reached model loading but the process was
terminated while the laptop had approximately 0.7 GiB free of 7.8 GiB RAM.
The preserved model weights and package versions were present and verified.
The experiment can proceed locally after sufficient RAM is freed, or its
Qwen-only inference stage can be run on a GPU instance. The remote environment
is execution-only and does not modify either local environment. XGBoost
training and retrieval evaluation do not require a GPU.
