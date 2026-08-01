"""Focused tests for the evidence-length Oracle and Qwen output parser."""

import pytest

from qwen_phase1 import (
    CLASS_TOKENS,
    _fixed_classification_metrics,
    clean_deduplicate_combine_evidence,
    closest_chunk_size,
    parse_qwen_class,
)


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (0, 10),
        (9, 10),
        (10, 10),
        (14, 10),
        (15, 10),
        (16, 20),
        (30, 20),
        (31, 40),
        (60, 40),
        (61, 80),
        (120, 80),
        (121, 160),
        (160, 160),
        (161, 160),
        (1000, 160),
    ],
)
def test_closest_chunk_size_boundaries(length, expected):
    assert closest_chunk_size(length) == expected


def test_closest_chunk_size_rejects_negative_length():
    with pytest.raises(ValueError):
        closest_chunk_size(-1)


def test_evidence_cleaning_deduplication_and_combination():
    cleaned, combined = clean_deduplicate_combine_evidence(
        [" beta ", "", "alpha", "alpha", "  "]
    )
    assert cleaned == ["alpha", "beta"]
    assert combined == "alpha\nbeta"


@pytest.mark.parametrize("value", CLASS_TOKENS)
def test_parser_accepts_each_exact_class(value):
    assert parse_qwen_class(str(value)) == (value, "valid")


def test_parser_accepts_class_in_sentence():
    assert parse_qwen_class("The selected size is 40.") == (40, "valid")


def test_parser_rejects_no_valid_class():
    assert parse_qwen_class("No class selected.") == (
        None,
        "invalid_no_valid_class",
    )


def test_parser_rejects_multiple_different_classes():
    assert parse_qwen_class("Use 20 or 40.") == (
        None,
        "invalid_multiple_classes",
    )


def test_parser_accepts_repeated_same_class():
    assert parse_qwen_class("40; chunk size 40.") == (40, "valid")


def test_parser_distinguishes_10_from_160():
    assert parse_qwen_class("160") == (160, "valid")
    assert parse_qwen_class("10") == (10, "valid")


def test_parser_ignores_unrelated_numbers():
    assert parse_qwen_class("Use 30 tokens in 2026.") == (
        None,
        "invalid_no_valid_class",
    )


@pytest.mark.parametrize("text", ["110", "2040", "801", "1600", "99910"])
def test_parser_rejects_valid_digits_embedded_in_larger_number(text):
    assert parse_qwen_class(text) == (None, "invalid_no_valid_class")


def test_classification_metrics_count_invalid_as_incorrect_without_defaulting():
    rows = [
        {"oracle_label": 10, "parsed_prediction": 10},
        {"oracle_label": 20, "parsed_prediction": None},
    ]
    metrics = _fixed_classification_metrics(rows, "parsed_prediction")
    assert metrics["accuracy"] == 0.5
    assert metrics["valid_predictions"] == 1
    assert metrics["invalid_predictions"] == 1
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 1
    assert metrics["per_class"]["20"]["support"] == 1
    assert metrics["per_class"]["20"]["recall"] == 0.0


def test_top2_accuracy_is_explicitly_unavailable_for_generation():
    metrics = _fixed_classification_metrics(
        [{"oracle_label": 40, "parsed_prediction": 40}],
        "parsed_prediction",
    )
    assert metrics["top_2_accuracy"] is None
    assert metrics["top_2_accuracy_status"] == (
        "unavailable_no_comparable_class_scores"
    )
