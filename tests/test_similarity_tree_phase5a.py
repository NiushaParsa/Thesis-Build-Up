from __future__ import annotations

import numpy as np

import similarity_tree_phase5a as phase5a


def complete_scores(root_count: int = 2) -> dict[int, np.ndarray]:
    return {
        tokens: np.linspace(0.1, 0.9, root_count * (160 // tokens), dtype=np.float64)
        for tokens in phase5a.CLASS_TOKENS
    }


def test_local_tree_has_aligned_descendant_counts() -> None:
    local = phase5a.local_score_tree(complete_scores(), 1)
    assert {tokens: len(values) for tokens, values in local.items()} == {
        10: 16,
        20: 8,
        40: 4,
        80: 2,
        160: 1,
    }


def test_last_partial_tree_remains_hierarchically_aligned() -> None:
    scores = {
        10: np.arange(19, dtype=float),
        20: np.arange(10, dtype=float),
        40: np.arange(5, dtype=float),
        80: np.arange(3, dtype=float),
        160: np.arange(2, dtype=float),
    }
    local = phase5a.local_score_tree(scores, 1)
    assert {tokens: len(values) for tokens, values in local.items()} == {
        10: 3,
        20: 2,
        40: 1,
        80: 1,
        160: 1,
    }
    assert len(phase5a.extract_local_features(local)) == 173


def test_tree_score_is_average_of_five_level_means() -> None:
    local = {tokens: np.asarray([index, index + 2], dtype=float) for index, tokens in enumerate(phase5a.CLASS_TOKENS)}
    expected = np.mean([np.mean(local[tokens]) for tokens in phase5a.CLASS_TOKENS])
    assert phase5a.tree_score(local) == expected


def test_root_ranking_is_descending_and_stable() -> None:
    scores = complete_scores(3)
    ranked = phase5a.rank_roots(scores, top_n=3)
    assert [row["root_index"] for row in ranked] == [2, 1, 0]
    tied = {tokens: np.ones(3 * (160 // tokens)) for tokens in phase5a.CLASS_TOKENS}
    assert [row["root_index"] for row in phase5a.rank_roots(tied, 3)] == [0, 1, 2]


def test_midpoint_ties_choose_smaller_class_and_extremes_clamp() -> None:
    expected = {
        0: 10,
        15: 10,
        16: 20,
        30: 20,
        31: 40,
        60: 40,
        61: 80,
        120: 80,
        121: 160,
        1000: 160,
    }
    assert {length: phase5a.closest_chunk_size(length) for length in expected} == expected


def test_clipping_merges_overlapping_and_adjacent_evidence() -> None:
    intervals = [(5, 15), (12, 25), (25, 30), (40, 60)]
    assert phase5a.clipped_overlap_intervals(intervals, 10, 50) == [(10, 30), (40, 50)]


def test_span_recovery_accepts_only_unique_matches() -> None:
    paper = "Alpha\n\nBeta gamma."
    assert phase5a.recover_span(paper, "beta   GAMMA.") == (7, 18, "unique_normalized_recovery")
    assert phase5a.recover_span("same and same", "same") is None


def test_single_chunk_choice_maps_local_to_global_index() -> None:
    local = {tokens: np.zeros(160 // tokens) for tokens in phase5a.CLASS_TOKENS}
    local[40] = np.asarray([0.2, 0.8, 0.7, 0.1])
    choice = phase5a.choose_single_chunk(local, root_index=3, tokens=40)
    assert choice["local_chunk_index"] == 1
    assert choice["global_chunk_index"] == 13
    assert choice["tokens"] == 40


def test_all_chunks_variant_keeps_every_descendant_in_similarity_order() -> None:
    local = {tokens: np.zeros(160 // tokens) for tokens in phase5a.CLASS_TOKENS}
    local[40] = np.asarray([0.2, 0.8, 0.7, 0.1])
    choices = phase5a.all_chunks_at_level(local, root_index=3, tokens=40)
    assert [row["local_chunk_index"] for row in choices] == [1, 2, 0, 3]
    assert [row["global_chunk_index"] for row in choices] == [13, 14, 12, 15]


def test_feature_schema_is_exactly_173_and_inference_safe() -> None:
    features = phase5a.extract_local_features(phase5a.local_score_tree(complete_scores(), 0))
    assert len(features) == 173
    phase5a.assert_inference_safe_feature_names(sorted(features))
    assert not any("evidence" in name or "label" in name for name in features)


def test_inference_rows_contain_no_gold_fields() -> None:
    row = {
        "split": "validation",
        "question_id": "q",
        "document_id": "d",
        "question_text": "question",
        "scores_by_tokens": {str(tokens): values.tolist() for tokens, values in complete_scores(1).items()},
    }
    inference = phase5a.inference_tree_rows([row])
    assert len(inference) == 1
    prohibited = {"local_oracle_label", "local_evidence_token_length", "local_evidence_intervals"}
    assert not prohibited.intersection(inference[0])
