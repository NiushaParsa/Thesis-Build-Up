#!/usr/bin/env python
"""Phase 5B-v2: tree-local routing with a repaired zero-shot Qwen category prompt.

Phase 5B-v1 was stopped by its frozen validity gate because zero-shot Qwen
answered 231 questions directly instead of returning a context category.  This
separate revision addresses that training-observed format failure without
changing the model, inputs, category encoding, tree classifier, or retrieval
evaluation.  The v1 artifacts remain untouched.
"""

from __future__ import annotations

from pathlib import Path

import similarity_tree_phase5b as base


PHASE = "Phase 5B-v2"
EXPERIMENT_NAME = "Tree-Local Similarity Router with Constrained Zero-Shot Qwen Context Feature"
FORMULATION_VERSION = "phase5b-v2-tree-local-constrained-zero-shot-qwen-context-v1"
OUTPUT_ROOT = Path("outputs/similarity_tree_phase5b_v2_zero_shot_qwen_context_feature")
REPORT_PATH = Path(
    "reports/similarity_tree_phase5b_v2_zero_shot_qwen_context_feature/experiment_report.md"
)
DOC_PATH = Path("docs/SIMILARITY_TREE_PHASE5B_V2_RESULTS.md")
ENTRYPOINT_SCRIPT = "similarity_tree_phase5b_v2.py"
FIXED_INSTRUCTION = (
    "You are not being asked to answer the question. Your only task is to "
    "classify how much supporting context would be needed to answer it. "
    "Respond with exactly one lowercase label from: short, medium, long. "
    "Do not output yes, no, the answer to the question, an explanation, or "
    "punctuation. Output one label only."
)
LABEL_TOKEN_IDS = {"short": 8412, "medium": 25252, "long": 4670}
DECISION_CONFIG = {
    "method": "restricted next-token argmax over the three allowed one-token labels",
    "do_sample": False,
    "candidate_labels": list(base.CONTEXT_LABELS),
    "candidate_token_ids": dict(LABEL_TOKEN_IDS),
    "candidate_tokens_are_single_token": True,
}
PROCEDURE_DEVELOPMENT_NOTE = (
    "The v2 prompt and constrained categorical decision repaired a format failure "
    "observed only in Phase 5B-v1 training outputs; no validation example informed "
    "the repair, and no classifier hyperparameter search was performed"
)
TEST_COMMAND = "tests/test_similarity_tree_phase5b.py tests/test_similarity_tree_phase5b_v2.py"


def build_prompt(question_text: str) -> str:
    return f"{FIXED_INSTRUCTION}\n\nQuestion: {question_text}"


def predict_context(processor, model, question_text: str):
    """Choose the highest-logit allowed category; unrelated text is impossible."""
    import time

    import torch

    for label, expected_id in LABEL_TOKEN_IDS.items():
        observed = processor.tokenizer.encode(label, add_special_tokens=False)
        if observed != [expected_id]:
            raise RuntimeError(
                f"Frozen label token changed for {label}: expected {[expected_id]}, got {observed}"
            )
    prompt = base.build_prompt(question_text)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }
    started = time.perf_counter()
    with torch.inference_mode():
        next_token_logits = model(**inputs).logits[0, -1]
    elapsed = time.perf_counter() - started
    labels = list(base.CONTEXT_LABELS)
    candidate_logits = torch.stack(
        [next_token_logits[LABEL_TOKEN_IDS[label]] for label in labels]
    )
    selected = labels[int(torch.argmax(candidate_logits).item())]
    parsed, status = base.parse_context_label(selected)
    if parsed != selected or status != "valid":
        raise RuntimeError("Restricted Qwen decision did not produce a valid label")
    return selected, parsed, status, elapsed


def configure_base() -> None:
    """Apply only the isolated v2 identity and frozen prompt to shared code."""
    base.PHASE = PHASE
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.FORMULATION_VERSION = FORMULATION_VERSION
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.REPORT_PATH = REPORT_PATH
    base.DOC_PATH = DOC_PATH
    base.ENTRYPOINT_SCRIPT = ENTRYPOINT_SCRIPT
    base.FIXED_INSTRUCTION = FIXED_INSTRUCTION
    base.DECISION_CONFIG = DECISION_CONFIG
    base.PROCEDURE_DEVELOPMENT_NOTE = PROCEDURE_DEVELOPMENT_NOTE
    base.TEST_COMMAND = TEST_COMMAND
    base.predict_context = predict_context


def main() -> int:
    configure_base()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
