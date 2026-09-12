from __future__ import annotations

import similarity_tree_phase5b as base
import similarity_tree_phase5b_v2 as phase5b_v2


def test_repaired_prompt_forbids_answering_and_preserves_question_only_input() -> None:
    question = "Is the LSTM bidirectional?"
    prompt = phase5b_v2.build_prompt(question)
    assert prompt == f"{phase5b_v2.FIXED_INSTRUCTION}\n\nQuestion: {question}"
    assert "not being asked to answer" in prompt
    assert "do not output yes, no" in prompt.casefold()
    assert "short, medium, long" in prompt
    assert prompt.endswith(question)


def test_v2_changes_only_experiment_identity_and_prompt() -> None:
    original_model = base.MODEL_ID
    original_revision = base.MODEL_REVISION
    original_transformers = base.TRANSFORMERS_COMMIT
    original_labels = base.CONTEXT_LABELS
    original_max_tokens = base.MAX_NEW_TOKENS
    phase5b_v2.configure_base()
    assert base.MODEL_ID == original_model
    assert base.MODEL_REVISION == original_revision
    assert base.TRANSFORMERS_COMMIT == original_transformers
    assert base.CONTEXT_LABELS == original_labels
    assert base.MAX_NEW_TOKENS == original_max_tokens
    assert base.OUTPUT_ROOT == phase5b_v2.OUTPUT_ROOT
    assert base.FIXED_INSTRUCTION == phase5b_v2.FIXED_INSTRUCTION
    assert base.predict_context is phase5b_v2.predict_context
    assert base.DECISION_CONFIG == phase5b_v2.DECISION_CONFIG


def test_restricted_labels_have_frozen_single_token_ids() -> None:
    assert phase5b_v2.LABEL_TOKEN_IDS == {
        "short": 8412,
        "medium": 25252,
        "long": 4670,
    }
    assert tuple(phase5b_v2.LABEL_TOKEN_IDS) == base.CONTEXT_LABELS
    assert phase5b_v2.DECISION_CONFIG["candidate_tokens_are_single_token"] is True
