import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import qwen_phase1 as phase1
import qwen_phase2 as phase2
import qwen_phase2b as phase2b
import qwen_phase2c_posttraining as post
import qwen_phase2c_sequence_classifier as phase2c


class FakeClient:
    def __init__(self):
        self.requested_ids = []

    def retrieve(self, *, ids, **kwargs):
        del kwargs
        self.requested_ids.extend(ids)
        return [SimpleNamespace(vector=[0.1, 0.2, 0.3])]


def fake_phase1(client=None):
    client = client or FakeClient()

    def evaluate_question(**kwargs):
        level = kwargs["granularity_levels"][0]
        tokens = kwargs["chunk_sizes"][level - 1]
        question_id = kwargs["question_point_id"]
        score = 0.2 if question_id == "q1" else 0.4
        return [
            {
                "question_id": question_id,
                "document_id": kwargs["document_id"],
                "split": kwargs["split"],
                "granularity_tokens": tokens,
                "granularity_level": level,
                "k_requested": kwargs["top_k"],
                "embedding_model": kwargs["embedding_model"],
                "embedding_dimension": kwargs["embedding_dimension"],
                "tokenizer_identity": kwargs["tokenizer_name"],
                "evaluation_run_id": kwargs["evaluation_run_id"],
                "evaluation_config_hash": kwargs["evaluation_config_hash"],
                "schema_version": post.RETRIEVAL_SCHEMA_VERSION,
                "metric_version": post.RETRIEVAL_METRIC_VERSION,
                "normalization_version": post.RETRIEVAL_NORMALIZATION_VERSION,
                "f1_joined_topk": score,
            }
        ]

    return SimpleNamespace(
        OPENAI_EMBEDDING_MODEL="embedding",
        EMBEDDING_DIM=3,
        TOKENIZER_NAME="gpt2",
        PAPER_QUESTION_COLLECTION="PaperQuestion",
        qdrant_client=lambda: client,
        evaluate_question=evaluate_question,
        client=client,
    )


def prediction(question_id, document_id, class_id, checkpoint="step-000213"):
    label = phase2c.CHUNK_SIZES[class_id]
    return {
        "question_id": question_id,
        "document_id": document_id,
        "question_text": f"Question {question_id}?",
        "oracle_label": label,
        "predicted_class_id": class_id,
        "predicted_label": label,
        "parsed_prediction": label,
        "prediction_status": getattr(
            phase2c, "PREDICTION_STATUS", "valid_classifier_argmax"
        ),
        "formulation_version": phase2c.FORMULATION_VERSION,
        "experiment_fingerprint": "fingerprint",
        "class_logits": [0.1, 0.2, 0.3, 0.4, 0.5],
        "class_probabilities": [0.1, 0.1, 0.2, 0.2, 0.4],
        "selected_checkpoint_id": checkpoint,
        "selected_checkpoint_path": f"checkpoints/{checkpoint}",
    }


def create_phase2c_root(tmp_path):
    root = tmp_path / "phase2c"
    run_id = "run-phase2c"
    rows = [prediction("q1", "d1", 0), prediction("q2", "d2", 4)]
    phase2.atomic_jsonl(root / "validation" / "predictions.jsonl", rows)
    phase2.atomic_json(
        root / "final_summary.json",
        {
            "status": "classification_complete_retrieval_pending",
            "experiment_status": "classification_complete_retrieval_pending",
            "phase": "Phase 2C",
            "formulation_version": phase2c.FORMULATION_VERSION,
            "run_id": run_id,
            "model_id": phase2c.MODEL_ID,
            "model_revision": phase2c.MODEL_REVISION,
            "evaluated_examples": 2,
            "valid_output_count": 2,
            "invalid_output_count": 0,
            "valid_output_rate": 1.0,
            "selected_checkpoint_id": "step-000213",
            "selected_checkpoint": "checkpoints/step-000213",
            "classification": {
                "accuracy": 1.0,
                "macro_f1": 0.4,
                "weighted_f1": 1.0,
                "balanced_accuracy": 1.0,
                "top_2_accuracy": 1.0,
                "top_2_accuracy_status": "available_classifier_logits",
            },
            "oracle_distribution": {
                "10": 1,
                "20": 0,
                "40": 0,
                "80": 0,
                "160": 1,
            },
            "predicted_distribution": {
                "10": 1,
                "20": 0,
                "40": 0,
                "80": 0,
                "160": 1,
            },
            "training": {
                "elapsed_seconds": 10.0,
                "experiment_fingerprint": "fingerprint",
            },
            "runtime": {
                "model_load_seconds": 2.0,
                "isolated_inference_wall_seconds": 3.0,
            },
            "retrieval": None,
            "artifacts": {"classification_metrics": "classification/metrics.json"},
        },
    )
    return root, run_id, rows


def test_phase1_evaluate_question_accepts_frozen_retrieval_contract():
    parameters = inspect.signature(phase1.evaluate_question).parameters
    assert {
        "client",
        "question_point_id",
        "question_vector",
        "document_id",
        "question_text",
        "split",
        "top_k",
        "granularity_levels",
        "store_retrieved_text",
        "chunk_sizes",
        "embedding_model",
        "embedding_dimension",
        "tokenizer_name",
        "evaluation_run_id",
        "evaluation_config_hash",
    }.issubset(parameters)


def test_posttrainer_contains_no_qdrant_mutation_calls():
    source = inspect.getsource(post)
    for forbidden in (
        ".upsert(",
        ".delete(",
        ".create_collection(",
        ".recreate_collection(",
        ".update_collection(",
    ):
        assert forbidden not in source


def test_resolve_output_root_rejects_prior_experiment_roots():
    forbidden = [
        phase2.PHASE1_ROOT,
        phase2.DEFAULT_OUTPUT_ROOT,
        *phase2b.DEFAULT_OUTPUT_ROOTS.values(),
    ]
    for root in forbidden:
        with pytest.raises(RuntimeError, match="isolated root"):
            post.resolve_output_root(root)
        with pytest.raises(RuntimeError, match="isolated root"):
            post.resolve_output_root(root / "phase2c")


def test_context_rejects_class_id_mapping_mismatch(tmp_path):
    root, run_id, rows = create_phase2c_root(tmp_path)
    rows[0]["predicted_label"] = 160
    rows[0]["parsed_prediction"] = 160
    phase2.atomic_jsonl(root / "validation" / "predictions.jsonl", rows)
    with pytest.raises(RuntimeError, match="ID-to-chunk mapping mismatch"):
        post.load_retrieval_context(root, run_id, expected_examples=2)


def test_context_rejects_any_default_or_fallback_prediction(tmp_path):
    root, run_id, rows = create_phase2c_root(tmp_path)
    rows[0]["default_applied"] = True
    phase2.atomic_jsonl(root / "validation" / "predictions.jsonl", rows)
    with pytest.raises(RuntimeError, match="default/fallback"):
        post.load_retrieval_context(root, run_id, expected_examples=2)


def test_retrieval_is_resumable_and_updates_only_phase2c_root(tmp_path):
    root, run_id, _ = create_phase2c_root(tmp_path)
    fake = fake_phase1()
    summary = post.evaluate_retrieval(
        root, run_id, phase1_module=fake, expected_examples=2
    )
    assert fake.client.requested_ids == ["q1", "q2"]
    assert summary["retrieval_coverage"] == 1.0
    assert summary["valid_only_mean_joined_retrieval_f1"] == pytest.approx(0.3)
    assert summary["valid_only_median_joined_retrieval_f1"] == pytest.approx(0.3)
    assert summary["coverage_adjusted_full_set_mean_joined_retrieval_f1"] == pytest.approx(
        0.3
    )
    assert summary["top_k"] == 5
    assert summary["paper_restricted"] is True
    assert summary["method_name"] == post.METHOD_NAME

    final = json.loads((root / "final_summary.json").read_text(encoding="utf-8"))
    assert final["status"] == "complete"
    assert final["experiment_status"] == "complete"
    assert final["classification"]["accuracy"] == 1.0
    assert final["runtime"]["known_training_plus_final_validation_wall_seconds"] == 15.0
    assert final["retrieval"] == summary

    second_fake = fake_phase1()
    repeated = post.evaluate_retrieval(
        root, run_id, phase1_module=second_fake, expected_examples=2
    )
    assert repeated == summary
    assert second_fake.client.requested_ids == []


def _first_durable_record(root, run_id, rows, fake):
    context = post.load_retrieval_context(root, run_id, expected_examples=2)
    first = fake.evaluate_question(
        client=fake.client,
        question_point_id="q1",
        question_vector=[0.1, 0.2, 0.3],
        document_id="d1",
        question_text="Question q1?",
        split="validation",
        top_k=5,
        granularity_levels=[1],
        store_retrieved_text=False,
        chunk_sizes=list(phase2c.CHUNK_SIZES),
        embedding_model=fake.OPENAI_EMBEDDING_MODEL,
        embedding_dimension=fake.EMBEDDING_DIM,
        tokenizer_name=fake.TOKENIZER_NAME,
        evaluation_run_id=post.retrieval_run_id(run_id),
        evaluation_config_hash=post.RETRIEVAL_CONFIG_HASH,
    )[0]
    first.update(
        {
            "method_name": post.METHOD_NAME,
            "phase2c_run_id": run_id,
            "formulation_version": phase2c.FORMULATION_VERSION,
            "experiment_fingerprint": "fingerprint",
            "predicted_class_id": 0,
            "predicted_granularity_tokens": 10,
            "predicted_granularity_level": 1,
            "classifier_prediction_status": rows[0]["prediction_status"],
            "evidence_length_oracle": 10,
            "oracle_label_version": phase2c.ORACLE_VERSION,
            "selected_checkpoint_id": "step-000213",
            "selected_checkpoint_path": "checkpoints/step-000213",
            "model_id": phase2c.MODEL_ID,
            "model_revision": phase2c.MODEL_REVISION,
            "top_k": 5,
            "paper_restricted": True,
            "phase2c_retrieval_wall_seconds": 0.25,
        }
    )
    post.validate_retrieval_record(first, rows[0], context, fake)
    return first


def test_partial_jsonl_resume_retrieves_only_missing_question(tmp_path):
    root, run_id, rows = create_phase2c_root(tmp_path)
    fake = fake_phase1()
    first = _first_durable_record(root, run_id, rows, fake)
    phase2.atomic_jsonl(root / "retrieval" / "results.jsonl", [first])

    summary = post.evaluate_retrieval(
        root, run_id, phase1_module=fake, expected_examples=2
    )
    assert fake.client.requested_ids == ["q2"]
    saved = post.read_jsonl(root / "retrieval" / "results.jsonl")
    assert [row["question_id"] for row in saved] == ["q1", "q2"]
    assert summary["complete_uninterrupted_run_wall_seconds"] is None
    assert summary["reported_retrieval_wall_basis"].startswith("durable_sum")


def test_stale_cross_run_record_is_rejected(tmp_path):
    root, run_id, rows = create_phase2c_root(tmp_path)
    context = post.load_retrieval_context(root, run_id, expected_examples=2)
    fake = fake_phase1()
    stale = {
        "question_id": "q1",
        "phase2c_run_id": "different-run",
        "phase2c_retrieval_wall_seconds": 1.0,
    }
    with pytest.raises(RuntimeError, match="Stale or incompatible"):
        post.validate_retrieval_record(stale, rows[0], context, fake)


def test_completed_summary_is_recomputed_before_reuse(tmp_path):
    root, run_id, _ = create_phase2c_root(tmp_path)
    fake = fake_phase1()
    post.evaluate_retrieval(root, run_id, phase1_module=fake, expected_examples=2)
    path = root / "retrieval" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["valid_only_mean_joined_retrieval_f1"] = 0.99
    phase2.atomic_json(path, summary)
    with pytest.raises(RuntimeError, match="does not recompute"):
        post.evaluate_retrieval(
            root, run_id, phase1_module=fake_phase1(), expected_examples=2
        )


def summary_file(
    path,
    *,
    phase,
    variant=None,
    formulation=None,
    retrieval_schema="phase2",
):
    retrieval = {
        "top_k": 5,
        "paper_restricted": True,
        "coverage": 1.0,
        "valid_only_mean_joined_f1": 0.2,
        "valid_only_median_joined_f1": 0.19,
    }
    if retrieval_schema == "phase2":
        retrieval = {
            "top_k": 5,
            "paper_restricted": True,
            "retrieval_coverage": 1.0,
            "valid_only_mean_joined_retrieval_f1": 0.3,
            "valid_only_median_joined_retrieval_f1": 0.29,
        }
    value = {
        "status": "complete",
        "phase": phase,
        "variant": variant,
        "formulation_version": formulation,
        "run_id": f"run-{phase}",
        "model_id": phase2c.MODEL_ID,
        "model_revision": phase2c.MODEL_REVISION,
        "evaluated_examples": 2,
        "classification": {
            "accuracy": 0.4,
            "macro_f1": 0.3,
            "weighted_f1": 0.35,
            "balanced_accuracy": 0.31,
            "top_2_accuracy": 0.7 if (variant or formulation) else None,
            "top_2_accuracy_status": (
                "available" if (variant or formulation) else "unavailable"
            ),
        },
        "oracle_distribution": {"10": 1, "160": 1},
        "predicted_distribution": {"10": 1, "160": 1},
        "valid_output_rate": 1.0,
        "retrieval": retrieval,
        "runtime": {},
    }
    phase2.atomic_json(path, value)
    return path


def test_compare_normalizes_all_five_summary_schemas(tmp_path):
    phase1_path = summary_file(
        tmp_path / "p1.json", phase="phase1", retrieval_schema="phase1"
    )
    phase2_path = summary_file(tmp_path / "p2.json", phase="phase2")
    unweighted = summary_file(
        tmp_path / "u.json",
        phase="phase2b",
        variant=phase2b.VARIANT_UNWEIGHTED,
    )
    balanced = summary_file(
        tmp_path / "b.json",
        phase="phase2b",
        variant=phase2b.VARIANT_CLASSBALANCED,
    )
    classifier = summary_file(
        tmp_path / "c.json",
        phase="Phase 2C",
        formulation=phase2c.FORMULATION_VERSION,
    )
    result = post.compare_summaries(
        phase1_path, phase2_path, unweighted, balanced, classifier
    )
    assert [row["name"] for row in result["rows"]] == [
        "phase1_zero_shot",
        "phase2_numeric_sft",
        "phase2b_alias_unweighted",
        "phase2b_alias_classbalanced",
        "phase2c_base_sequence_classifier",
    ]
    assert result["rows"][0]["mean_joined_retrieval_f1"] == 0.2
    assert result["rows"][1]["mean_joined_retrieval_f1"] == 0.3
    assert result["rows"][4]["top_2_accuracy"] == 0.7
    assert "cannot be attributed" in result["comparability_note"]


def test_compare_rejects_different_oracle_distribution(tmp_path):
    paths = [
        summary_file(tmp_path / "p1.json", phase="p1"),
        summary_file(tmp_path / "p2.json", phase="p2"),
        summary_file(
            tmp_path / "a.json",
            phase="p2b",
            variant=phase2b.VARIANT_UNWEIGHTED,
        ),
        summary_file(
            tmp_path / "b.json",
            phase="p2b",
            variant=phase2b.VARIANT_CLASSBALANCED,
        ),
        summary_file(
            tmp_path / "c.json",
            phase="p2c",
            formulation=phase2c.FORMULATION_VERSION,
        ),
    ]
    changed = json.loads(paths[-1].read_text(encoding="utf-8"))
    changed["oracle_distribution"] = {"10": 2, "160": 0}
    phase2.atomic_json(paths[-1], changed)
    with pytest.raises(RuntimeError, match="same validation set"):
        post.compare_summaries(*paths)
