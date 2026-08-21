from __future__ import annotations

import numpy as np

import qwen_phase3c_fusion as phase3c


def rows(count: int = 3) -> list[dict]:
    names = [f"feature_{index:03d}" for index in range(173)]
    return [
        {
            "question_id": f"q{row}",
            "document_id": f"p{row}",
            "question_text": f"Question {row}?",
            "oracle_label": phase3c.CLASS_TOKENS[row % 5],
            "tree_features": {name: float(row + index) for index, name in enumerate(names)},
        }
        for row in range(count)
    ]


def arrays(source: list[dict]) -> dict[str, np.ndarray]:
    return {
        "question_ids": np.asarray([row["question_id"] for row in source]),
        "oracle_labels": np.asarray([row["oracle_label"] for row in source]),
        "logits": np.arange(len(source) * 5, dtype=np.float32).reshape(len(source), 5),
        "hidden": np.arange(len(source) * 8, dtype=np.float32).reshape(len(source), 8),
        "token_counts": np.full(len(source), 64, dtype=np.int64),
    }


def test_classifier_text_is_exact_phase2d_plain_prompt() -> None:
    observed = phase3c.classifier_text("How many examples?")
    assert observed == (
        phase3c.SUPERVISOR_INSTRUCTION + "\n\nQuestion: How many examples?"
    )


def test_fusion_variants_have_expected_dimensions_and_names() -> None:
    source = rows()
    qwen = arrays(source)
    logits_matrix, logits_names = phase3c.fusion_matrix(
        source, qwen, "qwen_logits_tree"
    )
    hidden_matrix, hidden_names = phase3c.fusion_matrix(
        source, qwen, "qwen_hidden_tree"
    )
    assert logits_matrix.shape == (3, 178)
    assert hidden_matrix.shape == (3, 181)
    assert logits_names[:5] == [f"qwen_logit_{value}" for value in phase3c.CLASS_TOKENS]
    assert hidden_names[:2] == ["qwen_hidden_0000", "qwen_hidden_0001"]
    assert all(name.startswith("tree__") for name in logits_names[5:])


def test_qwen_alignment_fails_closed() -> None:
    source = rows()
    qwen = arrays(source)
    qwen["question_ids"] = qwen["question_ids"][::-1]
    try:
        phase3c.validate_qwen_arrays(source, qwen)
    except RuntimeError as error:
        assert "do not align" in str(error)
    else:
        raise AssertionError("Misaligned Qwen features were accepted")


def test_variant_tie_prefers_hidden_fusion_only_on_exact_metric_tie() -> None:
    result = {
        "selected_candidate": {
            "oof_metrics": {"macro_f1": 0.3, "balanced_accuracy": 0.4, "accuracy": 0.5}
        }
    }
    assert phase3c.variant_key("qwen_hidden_tree", result) > phase3c.variant_key(
        "qwen_logits_tree", result
    )
