import inspect
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import qwen_phase2 as phase2
import qwen_phase2d_posttraining as phase2d_post
import qwen_phase2d_sequence_classifier as phase2d
import qwen_phase2e_posttraining as post
import qwen_phase2e_sequence_classifier_lr_grid as phase2e


class FakeClient:
    def __init__(self):
        self.requested_ids = []
        self.retrieve_calls = []

    def retrieve(self, *, ids, **kwargs):
        self.requested_ids.extend(ids)
        self.retrieve_calls.append({"ids": list(ids), **kwargs})
        return [SimpleNamespace(vector=[0.1, 0.2, 0.3])]


def fake_phase1(client=None):
    client = client or FakeClient()
    evaluate_calls = []

    def evaluate_question(**kwargs):
        evaluate_calls.append(dict(kwargs))
        level = kwargs["granularity_levels"][0]
        tokens = kwargs["chunk_sizes"][level - 1]
        score = 0.2 if kwargs["question_point_id"] == "q1" else 0.4
        return [
            {
                "question_id": kwargs["question_point_id"],
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
        evaluate_calls=evaluate_calls,
        client=client,
    )


def _candidate(variant, step, *, winner=False):
    lr = phase2e.LEARNING_RATES[variant]
    score = 0.8 if winner else 0.1 + step / 10000
    return {
        "phase": phase2e.PHASE,
        "study_id": phase2e.STUDY_ID,
        "variant": variant,
        "learning_rate": lr,
        "run_id": phase2e.RUN_IDS[variant],
        "checkpoint_id": f"step-{step:06d}",
        "checkpoint_path": f"checkpoints/step-{step:06d}",
        "epoch": phase2e.EXPECTED_STEPS.index(step) + 1,
        "global_step": step,
        "validation_loss": 0.1 if winner else 1.0,
        "classification_metrics": {
            "accuracy": score,
            "macro_f1": score,
            "weighted_f1": score,
            "balanced_accuracy": score,
            "top_2_accuracy": min(1.0, score + 0.1),
        },
        "predicted_distribution": {
            "10": 1,
            "20": 0,
            "40": 0,
            "80": 0,
            "160": 1,
        },
    }


def _prediction(question_id, document_id, class_id, checkpoint_id):
    label = phase2d.CHUNK_SIZES[class_id]
    return {
        "question_id": question_id,
        "document_id": document_id,
        "question_text": f"Question {question_id}?",
        "oracle_label": label,
        "predicted_class_id": class_id,
        "predicted_label": label,
        "parsed_prediction": label,
        "prediction_status": getattr(
            phase2d, "PREDICTION_STATUS", "valid_classifier_argmax"
        ),
        "formulation_version": phase2e.FORMULATION_VERSION,
        "experiment_fingerprint": "trial-fingerprint",
        "class_logits": [0.1, 0.2, 0.3, 0.4, 0.5],
        "class_probabilities": [0.1, 0.1, 0.2, 0.2, 0.4],
        "selected_checkpoint_id": checkpoint_id,
        "selected_checkpoint_path": f"checkpoints/{checkpoint_id}",
    }


def create_phase2e_study(tmp_path):
    root = tmp_path / "phase2e"
    winner_variant = "lr1e-5"
    candidates = []
    for variant in phase2e.VARIANT_ORDER:
        for step in phase2e.EXPECTED_STEPS:
            candidates.append(
                _candidate(
                    variant,
                    step,
                    winner=variant == winner_variant and step == 213,
                )
            )
    winner = phase2e.select_candidate_rows(candidates)
    run_id = phase2e.RUN_IDS[winner_variant]
    checkpoint_id = winner["checkpoint_id"]
    trial_root = phase2e.trial_root(root, winner_variant)

    protocol = phase2e._stable_grid_protocol()
    phase2.atomic_json(
        root / "configuration" / "grid_experiment.json",
        {
            **protocol,
            "status": "predeclared_before_training",
            "grid_fingerprint": phase2e.grid_fingerprint(),
            "created_at": "2026-08-08T00:00:00Z",
        },
    )
    phase2.atomic_json(
        root / "comparison" / "selected_trial.json",
        {
            "status": "classification_winner_locked_before_retrieval",
            "phase": phase2e.PHASE,
            "study_id": phase2e.STUDY_ID,
            "grid_fingerprint": phase2e.grid_fingerprint(),
            "selection_order": list(phase2e.SELECTION_ORDER),
            "candidate_count": 15,
            "trial_count": 3,
            "winner": winner,
            "all_epoch_candidates": candidates,
            "per_variant": {},
            "retrieval_was_not_used_for_selection": True,
            "locked_at": "2026-08-08T01:00:00Z",
        },
    )
    selected = {
        "status": "selected_checkpoint_final_validation_complete",
        "phase": "Phase 2E Base sequence-classification LR-grid fine-tuning",
        "study_id": phase2e.STUDY_ID,
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "variant": winner_variant,
        "learning_rate": phase2e.LEARNING_RATES[winner_variant],
        "epochs": 5,
        "run_id": run_id,
        "selected_checkpoint_id": checkpoint_id,
        "classification": {"accuracy": 1.0},
        "retrieval_was_not_used_for_selection": True,
        "global_grid_winner": True,
    }
    phase2.atomic_json(
        root / "comparison" / "selected_final_summary.json", selected
    )
    phase2.atomic_json(
        trial_root / "configuration" / "experiment.json",
        {
            "phase": phase2e.PHASE,
            "study_id": phase2e.STUDY_ID,
            "formulation_version": phase2e.FORMULATION_VERSION,
            "grid_fingerprint": phase2e.grid_fingerprint(),
            "variant": winner_variant,
            "learning_rate": phase2e.LEARNING_RATES[winner_variant],
            "epochs": 5,
            "run_id": run_id,
            "model_id": phase2d.MODEL_ID,
            "model_revision": phase2d.MODEL_REVISION,
            "architecture": "AutoModelForSequenceClassification",
            "instruction": phase2d.SUPERVISOR_INSTRUCTION,
            "instruction_sha256": phase2e.PROMPT_SHA256,
            "input_template": "{instruction}\\n\\nQuestion: {original_question_text}",
            "objective": "uniform_five_class_cross_entropy",
        },
    )
    rows = [
        _prediction("q1", "d1", 0, checkpoint_id),
        _prediction("q2", "d2", 4, checkpoint_id),
    ]
    phase2.atomic_jsonl(trial_root / "validation" / "predictions.jsonl", rows)
    phase2.atomic_json(
        trial_root / "final_summary.json",
        {
            "status": "classification_complete_retrieval_pending",
            "experiment_status": "classification_complete_retrieval_pending",
            "phase": "Phase 2E Base sequence-classification LR-grid fine-tuning",
            "formulation_version": phase2e.FORMULATION_VERSION,
            "study_id": phase2e.STUDY_ID,
            "grid_fingerprint": phase2e.grid_fingerprint(),
            "variant": winner_variant,
            "learning_rate": phase2e.LEARNING_RATES[winner_variant],
            "epochs": 5,
            "run_id": run_id,
            "model_id": phase2d.MODEL_ID,
            "model_revision": phase2d.MODEL_REVISION,
            "evaluated_examples": 2,
            "valid_output_count": 2,
            "invalid_output_count": 0,
            "valid_output_rate": 1.0,
            "selected_checkpoint_id": checkpoint_id,
            "classification": {"accuracy": 1.0, "macro_f1": 1.0},
            "training": {
                "elapsed_seconds": 10.0,
                "experiment_fingerprint": "trial-fingerprint",
            },
            "runtime": {
                "model_load_seconds": 2.0,
                "isolated_inference_wall_seconds": 3.0,
            },
            "retrieval": None,
            "retrieval_was_not_used_for_selection": True,
            "global_grid_winner": True,
            "artifacts": {},
        },
    )
    return root, winner_variant, run_id, checkpoint_id, rows


def _write_manifest(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in entries),
        encoding="utf-8",
    )


def add_transfer_artifacts(root):
    """Create a tiny but structurally exact three-checkpoint transfer record."""

    selection_path = root / "comparison" / "selected_trial.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winner_metrics = selection["winner"]["classification_metrics"]
    selected_path = root / "comparison" / "selected_final_summary.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected["classification"] = winner_metrics
    phase2.atomic_json(selected_path, selected)
    winner_root = phase2e.trial_root(root, selection["winner"]["variant"])
    final_path = winner_root / "final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["classification"] = winner_metrics
    phase2.atomic_json(final_path, final)

    transfer_root = root / "integrity" / "transfer_manifests"
    manifests = transfer_root / "manifests"
    checkpoint_files = (
        "model/chat_template.jinja",
        "model/config.json",
        "model/model.safetensors",
        "model/tokenizer.json",
        "model/tokenizer_config.json",
        "optimizer.pt",
        "random_states.pt",
        "scheduler.pt",
        "training_state.json",
    )
    inventory = []
    for variant in phase2e.VARIANT_ORDER:
        candidates = [
            row
            for row in selection["all_epoch_candidates"]
            if row["variant"] == variant
        ]
        best = phase2e.select_candidate_rows(candidates)
        checkpoint_id = best["checkpoint_id"]
        run_id = phase2e.RUN_IDS[variant]
        run_root = root / "trials" / variant / "runs" / run_id
        phase2.atomic_json(
            run_root / "best_checkpoint.json",
            {"checkpoint_id": checkpoint_id},
        )
        phase2.atomic_json(
            run_root / "summary.json",
            {
                "status": "complete",
                "run_id": run_id,
                "selected_checkpoint_id": checkpoint_id,
            },
        )
        checkpoint_root = run_root / "checkpoints" / checkpoint_id
        checkpoint_entries = []
        for relative in checkpoint_files:
            path = checkpoint_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{variant}:{checkpoint_id}:{relative}".encode())
            manifest_relative = (
                f"{root.name}/trials/{variant}/runs/{run_id}/checkpoints/"
                f"{checkpoint_id}/{relative}"
            )
            checkpoint_entries.append(
                (phase2.sha256_file(path), manifest_relative)
            )
        _write_manifest(
            manifests / f"{variant}_selected_checkpoint_files.sha256",
            checkpoint_entries,
        )
        archive_hash = hashlib.sha256(f"archive:{variant}".encode()).hexdigest()
        archive_relative = (
            f"archives/phase2e-{variant}-{checkpoint_id}.tar.zst"
        )
        _write_manifest(
            manifests / f"{variant}_archive.sha256",
            [(archive_hash, archive_relative)],
        )
        chunk_entries = []
        for index in range(2):
            relative = (
                f"{variant}_chunks/phase2e-{variant}-{checkpoint_id}.tar.zst."
                f"part-{index:03d}"
            )
            chunk_entries.append(
                (hashlib.sha256(relative.encode()).hexdigest(), relative)
            )
        _write_manifest(manifests / f"{variant}_chunks.sha256", chunk_entries)
        inventory.append(
            {
                "variant": variant,
                "run_id": run_id,
                "checkpoint_id": checkpoint_id,
                "relative_path": (
                    f"{root.name}/trials/{variant}/runs/{run_id}/checkpoints/"
                    f"{checkpoint_id}"
                ),
                "archive_bytes": "1234",
                "archive_sha256": archive_hash,
                "chunk_count": "2",
            }
        )

    metadata_paths = (
        root / "configuration" / "grid_experiment.json",
        selection_path,
        selected_path,
        final_path,
    )
    _write_manifest(
        manifests / "metadata_files.sha256",
        [
            (
                phase2.sha256_file(path),
                f"{root.name}/{path.relative_to(root).as_posix()}",
            )
            for path in metadata_paths
        ],
    )
    _write_manifest(
        manifests / "metadata_archive.sha256",
        [
            (
                hashlib.sha256(b"metadata-archive").hexdigest(),
                "archives/phase2e-metadata.tar.zst",
            )
        ],
    )
    _write_manifest(
        manifests / "metadata_chunks.sha256",
        [
            (
                hashlib.sha256(b"metadata-chunk").hexdigest(),
                "metadata_chunks/phase2e-metadata.tar.zst.part-000",
            )
        ],
    )
    inventory_path = manifests / "transfer_inventory.tsv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "variant",
        "run_id",
        "checkpoint_id",
        "relative_path",
        "archive_bytes",
        "archive_sha256",
        "chunk_count",
    )
    inventory_path.write_text(
        "\t".join(columns)
        + "\n"
        + "".join(
            "\t".join(row[column] for column in columns) + "\n"
            for row in inventory
        ),
        encoding="utf-8",
    )
    manifest_files = sorted(manifests.iterdir(), key=lambda path: path.name)
    assert len(manifest_files) == 13
    _write_manifest(
        transfer_root / "manifest_bundle.sha256",
        [
            (phase2.sha256_file(path), f"manifests/{path.name}")
            for path in manifest_files
        ],
    )
    phase2.atomic_json(
        root / "integrity" / "selected_checkpoints_transfer_verification.json",
        {
            "status": "passed",
            "instance_id": 46617164,
            "study": root.name,
            "variants": inventory,
            "metadata_and_all_three_selected_checkpoints_verified": True,
            "all_remote_and_local_hashes_match": True,
            "remote_originals_retained": True,
            "verified_at": "2026-08-08T21:25:36Z",
        },
    )
    return inventory


def test_frozen_retrieval_contract_is_exact_phase2d_identity():
    assert post.TOP_K == phase2d_post.TOP_K == 5
    assert post.RETRIEVAL_CONFIG_HASH == phase2d_post.RETRIEVAL_CONFIG_HASH
    assert post.RETRIEVAL_SCHEMA_VERSION == phase2d_post.RETRIEVAL_SCHEMA_VERSION
    assert post.RETRIEVAL_METRIC_VERSION == phase2d_post.RETRIEVAL_METRIC_VERSION
    assert (
        post.RETRIEVAL_NORMALIZATION_VERSION
        == phase2d_post.RETRIEVAL_NORMALIZATION_VERSION
    )
    assert post.FROZEN_RETRIEVAL_IDENTITY == phase2d_post.FROZEN_RETRIEVAL_IDENTITY


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


def test_locked_winner_is_recomputed_before_retrieval(tmp_path):
    root, variant, run_id, checkpoint_id, _ = create_phase2e_study(tmp_path)
    winner = post.load_locked_winner(root)
    assert winner.variant == variant
    assert winner.run_id == run_id
    assert winner.checkpoint_id == checkpoint_id
    assert winner.trial_root == phase2e.trial_root(root, variant)

    selection_path = root / "comparison" / "selected_trial.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["winner"] = next(
        row
        for row in selection["all_epoch_candidates"]
        if row["variant"] == "lr5e-6" and row["global_step"] == 71
    )
    phase2.atomic_json(selection_path, selection)
    with pytest.raises(RuntimeError, match="does not recompute"):
        post.load_locked_winner(root)


def test_selected_final_validation_is_required(tmp_path):
    root, *_ = create_phase2e_study(tmp_path)
    (root / "comparison" / "selected_final_summary.json").unlink()
    with pytest.raises(RuntimeError, match="requires saved artifact"):
        post.load_locked_winner(root)


def test_retrieval_for_nonwinner_is_rejected(tmp_path):
    root, variant, *_ = create_phase2e_study(tmp_path)
    other = next(value for value in phase2e.VARIANT_ORDER if value != variant)
    phase2.atomic_jsonl(
        phase2e.trial_root(root, other) / "retrieval" / "results.jsonl",
        [{"question_id": "wrong"}],
    )
    with pytest.raises(RuntimeError, match="non-winning"):
        post.load_locked_winner(root)


def test_retrieval_runs_only_for_locked_winner_and_is_resumable(tmp_path):
    root, variant, run_id, checkpoint_id, _ = create_phase2e_study(tmp_path)
    selection_path = root / "comparison" / "selected_trial.json"
    selection_before = selection_path.read_bytes()
    fake = fake_phase1()
    summary = post.evaluate_selected_retrieval(
        root, phase1_module=fake, expected_examples=2
    )

    assert selection_path.read_bytes() == selection_before
    assert summary["variant"] == variant
    assert summary["phase2e_run_id"] == run_id
    assert summary["selected_checkpoint_id"] == checkpoint_id
    assert summary["classification_winner_was_locked_before_retrieval"] is True
    assert summary["retrieval_was_not_used_for_selection"] is True
    assert summary["retrieval_coverage"] == 1.0
    assert summary["valid_only_mean_joined_retrieval_f1"] == pytest.approx(0.3)
    assert summary["valid_only_median_joined_retrieval_f1"] == pytest.approx(0.3)
    assert summary["top_k"] == 5
    assert summary["paper_restricted"] is True
    assert fake.client.requested_ids == ["q1", "q2"]
    assert fake.client.retrieve_calls == [
        {
            "ids": ["q1"],
            "collection_name": "PaperQuestion",
            "with_payload": True,
            "with_vectors": True,
        },
        {
            "ids": ["q2"],
            "collection_name": "PaperQuestion",
            "with_payload": True,
            "with_vectors": True,
        },
    ]
    for call, question_id, document_id, level in zip(
        fake.evaluate_calls, ("q1", "q2"), ("d1", "d2"), (1, 5)
    ):
        assert call["client"] is fake.client
        assert call["question_point_id"] == question_id
        assert call["document_id"] == document_id
        assert call["split"] == "validation"
        assert call["top_k"] == 5
        assert call["granularity_levels"] == [level]
        assert call["store_retrieved_text"] is False
        assert call["chunk_sizes"] == [10, 20, 40, 80, 160]
        assert call["embedding_model"] == "embedding"
        assert call["embedding_dimension"] == 3
        assert call["tokenizer_name"] == "gpt2"
        assert call["evaluation_config_hash"] == post.RETRIEVAL_CONFIG_HASH
        assert call["evaluation_run_id"] == post.retrieval_run_id(run_id)

    winning_root = phase2e.trial_root(root, variant)
    final = json.loads((winning_root / "final_summary.json").read_text())
    selected = json.loads(
        (root / "comparison" / "selected_final_summary.json").read_text()
    )
    assert final["status"] == "complete"
    assert selected["status"] == "complete"
    assert final["classification"]["accuracy"] == 1.0
    assert final["retrieval"] == summary
    assert selected["retrieval"] == summary
    assert final["runtime"]["known_training_plus_final_validation_wall_seconds"] == 15
    for other in phase2e.VARIANT_ORDER:
        if other != variant:
            assert not (phase2e.trial_root(root, other) / "retrieval").exists()

    repeated_fake = fake_phase1()
    repeated = post.evaluate_selected_retrieval(
        root, phase1_module=repeated_fake, expected_examples=2
    )
    assert repeated == summary
    assert repeated_fake.client.requested_ids == []


def test_prediction_must_match_locked_checkpoint(tmp_path):
    root, variant, _, _, rows = create_phase2e_study(tmp_path)
    rows[0]["selected_checkpoint_id"] = "step-000999"
    phase2.atomic_jsonl(
        phase2e.trial_root(root, variant) / "validation" / "predictions.jsonl",
        rows,
    )
    with pytest.raises(RuntimeError, match="selected_checkpoint_id mismatch"):
        post.load_retrieval_context(root, expected_examples=2)


def test_completed_summary_is_recomputed_before_reuse(tmp_path):
    root, variant, *_ = create_phase2e_study(tmp_path)
    fake = fake_phase1()
    post.evaluate_selected_retrieval(root, phase1_module=fake, expected_examples=2)
    path = phase2e.trial_root(root, variant) / "retrieval" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["valid_only_mean_joined_retrieval_f1"] = 0.99
    phase2.atomic_json(path, summary)
    with pytest.raises(RuntimeError, match="does not recompute"):
        post.evaluate_selected_retrieval(
            root, phase1_module=fake_phase1(), expected_examples=2
        )


def test_study_root_cannot_overlap_prior_phase_root():
    with pytest.raises(RuntimeError, match="prior experiment root"):
        post.load_locked_winner(phase2d.DEFAULT_OUTPUT_ROOT)


def test_final_post_retrieval_audit_verifies_transfer_and_recomputation(tmp_path):
    root, variant, run_id, checkpoint_id, _ = create_phase2e_study(tmp_path)
    add_transfer_artifacts(root)
    fake = fake_phase1()
    post.evaluate_selected_retrieval(
        root, phase1_module=fake, expected_examples=2
    )

    audit = post.audit_final_post_retrieval(
        root,
        phase1_module=fake,
        expected_examples=2,
        expected_metadata_files=4,
    )

    assert audit["status"] == "passed"
    assert audit["selection"]["variant"] == variant
    assert audit["selection"]["run_id"] == run_id
    assert audit["selection"]["selected_checkpoint_id"] == checkpoint_id
    assert audit["retrieval"]["records"] == 2
    assert audit["retrieval"]["unique_question_ids"] == 2
    assert audit["retrieval"]["retrieval_coverage"] == 1.0
    assert audit["retrieval"]["mean_joined_retrieval_f1"] == pytest.approx(0.3)
    assert audit["retrieval"]["median_joined_retrieval_f1"] == pytest.approx(0.3)
    assert audit["retrieval"]["top_k"] == 5
    assert audit["retrieval"]["paper_restricted"] is True
    assert audit["retrieval"]["evaluation_config_hash"] == post.RETRIEVAL_CONFIG_HASH
    assert audit["transfer"]["verification_status"] == "passed"
    assert audit["transfer"]["manifest_files_verified"] == 13
    assert audit["transfer"]["checkpoint_files_verified"] == 27
    assert [row["verified_file_count"] for row in audit["transfer"]["checkpoints"]] == [9, 9, 9]
    assert audit["metadata"]["manifest_entries"] == 4
    assert audit["metadata"]["unchanged_entries_verified"] == 2
    assert audit["metadata"]["authorized_changed_entries"] == 2
    assert {
        row["reason"] for row in audit["metadata"]["authorized_changes"]
    } == {"authorized_post_retrieval_summary_update"}
    assert audit["experiment_rerun"] is False
    assert audit["retrieval_rerun"] is False
    assert audit["qdrant_mutation"] is False
    saved = json.loads(
        (root / "integrity" / "final_post_retrieval_audit.json").read_text()
    )
    assert saved == audit


def test_final_audit_rejects_checkpoint_hash_drift(tmp_path):
    root, *_ = create_phase2e_study(tmp_path)
    inventory = add_transfer_artifacts(root)
    fake = fake_phase1()
    post.evaluate_selected_retrieval(
        root, phase1_module=fake, expected_examples=2
    )
    checkpoint = root.parent / inventory[0]["relative_path"] / "optimizer.pt"
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        post.audit_final_post_retrieval(
            root,
            phase1_module=fake,
            expected_examples=2,
            expected_metadata_files=4,
        )
    assert not (root / "integrity" / "final_post_retrieval_audit.json").exists()


def test_final_audit_rejects_unexpected_metadata_drift(tmp_path):
    root, *_ = create_phase2e_study(tmp_path)
    add_transfer_artifacts(root)
    fake = fake_phase1()
    post.evaluate_selected_retrieval(
        root, phase1_module=fake, expected_examples=2
    )
    grid_path = root / "configuration" / "grid_experiment.json"
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    grid["unexpected_post_transfer_change"] = True
    phase2.atomic_json(grid_path, grid)

    with pytest.raises(RuntimeError, match="Unexpected post-transfer metadata drift"):
        post.audit_final_post_retrieval(
            root,
            phase1_module=fake,
            expected_examples=2,
            expected_metadata_files=4,
        )


def test_final_audit_rejects_duplicate_retrieval_ids(tmp_path):
    root, variant, *_ = create_phase2e_study(tmp_path)
    add_transfer_artifacts(root)
    fake = fake_phase1()
    post.evaluate_selected_retrieval(
        root, phase1_module=fake, expected_examples=2
    )
    results_path = phase2e.trial_root(root, variant) / "retrieval" / "results.jsonl"
    rows = post.read_jsonl(results_path)
    rows[1]["question_id"] = rows[0]["question_id"]
    phase2.atomic_jsonl(results_path, rows)

    with pytest.raises(RuntimeError, match="canonical unique question IDs"):
        post.audit_final_post_retrieval(
            root,
            phase1_module=fake,
            expected_examples=2,
            expected_metadata_files=4,
        )


def test_final_audit_rejects_forbidden_tensorboard_payload(tmp_path):
    root, *_ = create_phase2e_study(tmp_path)
    add_transfer_artifacts(root)
    fake = fake_phase1()
    post.evaluate_selected_retrieval(
        root, phase1_module=fake, expected_examples=2
    )
    tensorboard = root / "trials" / "lr1e-5" / "tensorboard"
    tensorboard.mkdir(parents=True)
    (tensorboard / "events").write_text("forbidden", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Forbidden transfer/TensorBoard"):
        post.audit_final_post_retrieval(
            root,
            phase1_module=fake,
            expected_examples=2,
            expected_metadata_files=4,
        )
