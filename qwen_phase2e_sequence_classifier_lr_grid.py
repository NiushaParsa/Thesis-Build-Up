#!/usr/bin/env python
"""Phase 2E learning-rate grid for the frozen Phase 2D classifier protocol.

Phase 2E is an explicitly development-set-selected hyperparameter study.  It
reuses the completed Phase 2D computational implementation without modifying
its source or artifacts.  The only optimization changes are a five-epoch
training horizon and one of three predeclared peak learning rates.  Every trial
starts from the same pretrained Base revision and seed-42 classifier-head
initialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import qwen_phase2 as phase2
import qwen_phase2b as phase2b
import qwen_phase2c_sequence_classifier as phase2c
import qwen_phase2d_sequence_classifier as phase2d


PHASE = "Phase 2E"
STUDY_ID = "qwen-phase2e-lr-grid-token-count-prompt-5epochs-seed42-v1"
FORMULATION_VERSION = (
    "qwen-phase2e-base-sequence-classifier-token-count-prompt-lr-grid-v1"
)
DEFAULT_STUDY_ROOT = Path(
    "outputs/qwen_phase2e_lr_grid_token_count_prompt_5epochs_"
    "evidence_length_oracle"
)
VARIANT_ORDER = ("lr5e-6", "lr1e-5", "lr2e-5")
LEARNING_RATES = {
    "lr5e-6": 5e-6,
    "lr1e-5": 1e-5,
    "lr2e-5": 2e-5,
}
RUN_IDS = {
    "lr5e-6": (
        "qwen-phase2e-base-sequence-classifier-token-count-prompt-"
        "lr5e-6-5epochs-full-parameter-20260808-seed42-v1"
    ),
    "lr1e-5": (
        "qwen-phase2e-base-sequence-classifier-token-count-prompt-"
        "lr1e-5-5epochs-full-parameter-20260808-seed42-v1"
    ),
    "lr2e-5": (
        "qwen-phase2e-base-sequence-classifier-token-count-prompt-"
        "lr2e-5-5epochs-full-parameter-20260808-seed42-v1"
    ),
}
EXPECTED_STEPS = (71, 142, 213, 284, 355)
EXPECTED_TOTAL_STEPS = 355
EXPECTED_WARMUP_STEPS = 18
EXPECTED_INITIAL_HEAD_SHA256 = (
    "09826669f451891218742ea86926e0b484d1696e57999276889d97b5ccdcbda5"
)
PROMPT_SHA256 = (
    "b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5"
)
TRAIN_ORACLE_SHA256 = (
    "64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88"
)
VALIDATION_ORACLE_SHA256 = (
    "ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d"
)
SELECTION_ORDER = (
    "higher_macro_f1",
    "higher_accuracy",
    "higher_weighted_f1",
    "higher_balanced_accuracy",
    "lower_validation_cross_entropy",
    "earlier_optimizer_step",
    "lower_numeric_learning_rate_exact_final_tie",
)

_PHASE2D_CONFIG_CLASS = phase2d.TrainingConfig
_PHASE2D_EXPERIMENT_FINGERPRINT = phase2d.experiment_fingerprint
_PHASE2D_DEFAULT_OUTPUT_ROOT = Path(phase2d.DEFAULT_OUTPUT_ROOT)
_PHASE2D_SOURCE = Path(phase2d.__file__).resolve()


def execution_dependency_hashes() -> dict[str, str]:
    """Hash every Python module in the remote Phase 2E execution snapshot."""

    paths = {
        "qwen_phase2.py": Path(phase2.__file__).resolve(),
        "qwen_phase2b.py": Path(phase2b.__file__).resolve(),
        "qwen_phase2c_sequence_classifier.py": Path(phase2c.__file__).resolve(),
        "qwen_phase2d_sequence_classifier.py": _PHASE2D_SOURCE,
        "qwen_phase2e_sequence_classifier_lr_grid.py": Path(__file__).resolve(),
    }
    return {name: phase2.sha256_file(path) for name, path in paths.items()}


@dataclass(frozen=True)
class TrainingConfig(_PHASE2D_CONFIG_CLASS):
    """Phase 2D configuration with only the predeclared Phase 2E changes."""

    formulation_version: str = FORMULATION_VERSION
    epochs: int = 5
    early_stopping: str = "none_fixed_five_epochs"


def _validate_variant(variant: str) -> str:
    if variant not in LEARNING_RATES:
        raise ValueError(
            f"Unknown Phase 2E variant {variant!r}; expected one of {VARIANT_ORDER}"
        )
    return variant


def config_for_variant(variant: str) -> TrainingConfig:
    variant = _validate_variant(variant)
    value = float(LEARNING_RATES[variant])
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"Invalid predeclared learning rate for {variant}: {value}")
    return replace(TrainingConfig(), learning_rate=value)


def _stable_grid_protocol() -> dict[str, Any]:
    base = asdict(_PHASE2D_CONFIG_CLASS())
    frozen_config = {
        key: value
        for key, value in base.items()
        if key not in {"formulation_version", "learning_rate", "epochs", "early_stopping"}
    }
    return {
        "phase": PHASE,
        "study_id": STUDY_ID,
        "formulation_version": FORMULATION_VERSION,
        "model_id": phase2d.MODEL_ID,
        "model_revision": phase2d.MODEL_REVISION,
        "transformers_commit": phase2d.TRANSFORMERS_COMMIT,
        "architecture": "AutoModelForSequenceClassification",
        "instruction": phase2d.SUPERVISOR_INSTRUCTION,
        "instruction_sha256": PROMPT_SHA256,
        "input_template": "{instruction}\\n\\nQuestion: {original_question_text}",
        "id2label": {
            str(index): str(label) for index, label in phase2d.ID_TO_LABEL.items()
        },
        "label2id": {
            str(label): index for label, index in phase2d.LABEL_TO_ID.items()
        },
        "objective": "uniform_five_class_cross_entropy",
        "ordered_variants": list(VARIANT_ORDER),
        "learning_rates": {
            variant: LEARNING_RATES[variant] for variant in VARIANT_ORDER
        },
        "epochs": 5,
        "seed": 42,
        "steps_per_epoch": 71,
        "total_optimizer_steps": EXPECTED_TOTAL_STEPS,
        "warmup_steps": EXPECTED_WARMUP_STEPS,
        "validation_checkpoint_steps": list(EXPECTED_STEPS),
        "selection_order": list(SELECTION_ORDER),
        "run_ids": {variant: RUN_IDS[variant] for variant in VARIANT_ORDER},
        "train_oracle_sha256": TRAIN_ORACLE_SHA256,
        "validation_oracle_sha256": VALIDATION_ORACLE_SHA256,
        "expected_initial_classifier_head_sha256_float32": (
            EXPECTED_INITIAL_HEAD_SHA256
        ),
        "frozen_phase2d_configuration": frozen_config,
        "allowed_changes_from_phase2d": [
            "phase/formulation/artifact identity",
            "learning_rate selected from the predeclared grid",
            "epochs increased from 3 to 5",
            "derived cosine-schedule horizon and warmup steps",
        ],
        "validation_role": (
            "development/model-selection set; repeatedly observed and not an "
            "unbiased final generalization estimate"
        ),
        "retrieval_selection_rule": (
            "lock the classification winner before unchanged retrieval; retrieval "
            "F1 cannot select or revise the winning trial"
        ),
    }


def grid_fingerprint() -> str:
    return phase2d.canonical_json_sha256(_stable_grid_protocol())


def trial_fingerprint(
    data_manifest: Mapping[str, Any], pad_token_id: int, variant: str
) -> str:
    variant = _validate_variant(variant)
    if data_manifest.get("train_oracle_sha256") != TRAIN_ORACLE_SHA256:
        raise RuntimeError("Phase 2E train Oracle hash drifted")
    if data_manifest.get("validation_oracle_sha256") != VALIDATION_ORACLE_SHA256:
        raise RuntimeError("Phase 2E validation Oracle hash drifted")
    return phase2d.canonical_json_sha256(
        {
            "grid_fingerprint": grid_fingerprint(),
            "variant": variant,
            "learning_rate": LEARNING_RATES[variant],
            "pad_token_id": int(pad_token_id),
            "train_oracle_sha256": data_manifest["train_oracle_sha256"],
            "validation_oracle_sha256": data_manifest[
                "validation_oracle_sha256"
            ],
        }
    )


def _prior_output_roots() -> tuple[Path, ...]:
    return (
        phase2.PHASE1_ROOT,
        phase2.DEFAULT_OUTPUT_ROOT,
        *phase2b.DEFAULT_OUTPUT_ROOTS.values(),
        phase2c.DEFAULT_OUTPUT_ROOT,
        _PHASE2D_DEFAULT_OUTPUT_ROOT,
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _validate_study_root(study_root: Path) -> Path:
    root = Path(study_root)
    for protected in _prior_output_roots():
        if _paths_overlap(root, Path(protected)):
            raise RuntimeError(
                "Phase 2E study root must not equal, contain, or be contained by "
                f"a prior experiment root: {protected}"
            )
    return root


def trial_root(study_root: Path, variant: str) -> Path:
    return _validate_study_root(study_root) / "trials" / _validate_variant(variant)


def _verify_or_write_json(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"expected": value, "actual": existing.get(key)}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Immutable Phase 2E metadata mismatch: {mismatches}")
        return existing
    value = {**expected, "created_at": phase2.utc_now()}
    phase2.atomic_json(path, value)
    return value


def prepare_study(study_root: Path) -> dict[str, Any]:
    root = _validate_study_root(study_root)
    protocol = _stable_grid_protocol()
    marker = {
        **protocol,
        "status": "predeclared_before_training",
        "grid_fingerprint": grid_fingerprint(),
        "repository_base_commit": os.getenv(
            "PHASE2E_REPOSITORY_BASE_COMMIT", "unavailable"
        ),
        "source_snapshot_status": (
            "content-addressed Phase 2E worktree snapshot; the base commit does "
            "not by itself identify the new Phase 2E files"
        ),
        "execution_dependency_sha256": execution_dependency_hashes(),
        "remote_runner_sha256": os.getenv(
            "PHASE2E_REMOTE_RUNNER_SHA256", "unavailable"
        ),
        "phase2d_execution_module": str(_PHASE2D_SOURCE),
        "phase2d_execution_module_sha256": phase2.sha256_file(_PHASE2D_SOURCE),
        "phase2e_orchestrator_sha256": phase2.sha256_file(Path(__file__)),
    }
    path = root / "configuration" / "grid_experiment.json"
    return _verify_or_write_json(path, marker)


def _ensure_trial_root_factory(study_root: Path, variant: str):
    expected_root = trial_root(study_root, variant).resolve()
    config = config_for_variant(variant)

    def ensure_output_root(output_root: Path) -> dict[str, Any]:
        actual = Path(output_root).resolve()
        if actual != expected_root:
            raise RuntimeError(
                f"Phase 2E {variant} must write exactly to {expected_root}, got {actual}"
            )
        prepare_study(study_root)
        marker = {
            "phase": PHASE,
            "study_id": STUDY_ID,
            "formulation_version": FORMULATION_VERSION,
            "grid_fingerprint": grid_fingerprint(),
            "trial_fingerprint_static": phase2d.canonical_json_sha256(
                {
                    "grid_fingerprint": grid_fingerprint(),
                    "variant": variant,
                    "learning_rate": config.learning_rate,
                }
            ),
            "variant": variant,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "run_id": RUN_IDS[variant],
            "model_id": phase2d.MODEL_ID,
            "model_revision": phase2d.MODEL_REVISION,
            "architecture": "AutoModelForSequenceClassification",
            "instruction": phase2d.SUPERVISOR_INSTRUCTION,
            "instruction_sha256": PROMPT_SHA256,
            "input_template": "{instruction}\\n\\nQuestion: {original_question_text}",
            "id2label": {
                str(index): str(label)
                for index, label in phase2d.ID_TO_LABEL.items()
            },
            "label2id": {
                str(label): index
                for label, index in phase2d.LABEL_TO_ID.items()
            },
            "objective": "uniform_five_class_cross_entropy",
        }
        return _verify_or_write_json(
            expected_root / "configuration" / "experiment.json", marker
        )

    return ensure_output_root


def activate_variant(study_root: Path, variant: str) -> tuple[TrainingConfig, Path]:
    """Bind the frozen Phase 2D implementation to one isolated Phase 2E trial."""

    variant = _validate_variant(variant)
    config = config_for_variant(variant)
    output_root = trial_root(study_root, variant)
    phase2d.FORMULATION_VERSION = FORMULATION_VERSION
    phase2d.DEFAULT_OUTPUT_ROOT = output_root
    phase2d.TrainingConfig = lambda: config
    phase2d.ensure_output_root = _ensure_trial_root_factory(study_root, variant)
    phase2d.experiment_fingerprint = (
        lambda manifest, pad_id: trial_fingerprint(manifest, pad_id, variant)
    )
    phase2d._run_id = lambda mode: RUN_IDS[variant]
    os.environ["PHASE2D_REPOSITORY_COMMIT"] = (
        "not_applicable_content_addressed_phase2e_snapshot"
    )
    return config, output_root


def _augment_run_artifacts(study_root: Path, variant: str) -> dict[str, Any]:
    config, output_root = activate_variant(study_root, variant)
    run_id = RUN_IDS[variant]
    run_dir = output_root / "runs" / run_id
    metadata = {
        "phase": PHASE,
        "study_id": STUDY_ID,
        "grid_fingerprint": grid_fingerprint(),
        "variant": variant,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "phase2e_orchestrator_sha256": phase2.sha256_file(Path(__file__)),
        "phase2d_execution_module_sha256": phase2.sha256_file(_PHASE2D_SOURCE),
        "execution_dependency_sha256": execution_dependency_hashes(),
        "repository_base_commit": os.getenv(
            "PHASE2E_REPOSITORY_BASE_COMMIT", "unavailable"
        ),
    }
    names = (
        "training_config.json",
        "dataset_manifest.json",
        "best_checkpoint.json",
        "summary.json",
    )
    originals: dict[str, dict[str, Any]] = {}
    for name in names:
        path = run_dir / name
        if not path.exists():
            raise RuntimeError(f"Completed Phase 2E run lacks {path}")
        originals[name] = json.loads(path.read_text(encoding="utf-8"))

    training = originals["training_config.json"]
    data = originals["dataset_manifest.json"]
    best = originals["best_checkpoint.json"]
    summary = originals["summary.json"]
    head_hash = training["initial_model_loading_audit"]["classifier_head"][
        "head_weight_sha256_float32"
    ]
    expected_config = asdict(config)
    config_fields_match = all(
        training.get(key) == value for key, value in expected_config.items()
    )
    data_fields_match = all(
        data.get(key) == value
        for key, value in {
            "train_oracle_sha256": TRAIN_ORACLE_SHA256,
            "validation_oracle_sha256": VALIDATION_ORACLE_SHA256,
            "active_train_examples": 2245,
            "active_validation_examples": 924,
        }.items()
    )
    expected_fingerprint = trial_fingerprint(
        data, int(training["pad_token_id"]), variant
    )
    fingerprint_fields_match = all(
        document.get("experiment_fingerprint") == expected_fingerprint
        for document in (training, data, best, summary)
    )
    prior_metadata_matches = True
    prior_identity_matches = True
    for value in originals.values():
        prior = value.get("phase2e_metadata")
        if prior is not None and prior != metadata:
            prior_metadata_matches = False
        for key in ("study_id", "grid_fingerprint", "variant"):
            if key in value and value[key] != metadata[key]:
                prior_identity_matches = False
    checks = {
        "status_complete": summary.get("status") == "complete",
        "run_id": summary.get("run_id") == run_id
        and training.get("run_id") == run_id,
        "global_steps": int(summary["global_step"]) == EXPECTED_TOTAL_STEPS,
        "validation_events": int(summary["validation_events"]) == 5,
        "complete_frozen_training_config": config_fields_match,
        "data_identity": data_fields_match,
        "experiment_fingerprint": fingerprint_fields_match,
        "selected_checkpoint": best.get("checkpoint_id")
        == summary.get("selected_checkpoint_id")
        and int(best.get("global_step", -1)) in EXPECTED_STEPS,
        "learning_rate": training.get("learning_rate") == config.learning_rate,
        "epochs": training.get("epochs") == 5,
        "total_optimizer_steps": training.get("total_optimizer_steps")
        == EXPECTED_TOTAL_STEPS,
        "warmup_steps": int(training["warmup_steps"]) == EXPECTED_WARMUP_STEPS,
        "initial_head_hash": head_hash == EXPECTED_INITIAL_HEAD_SHA256,
        "prior_phase2e_metadata": prior_metadata_matches,
        "prior_phase2e_identity": prior_identity_matches,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 2E completed-run audit failed: {checks}")

    for name, original in originals.items():
        value = dict(original)
        value["phase2e_metadata"] = metadata
        if name == "summary.json":
            original_phase = value.get("phase")
            if original_phase not in {"Phase 2D", PHASE}:
                raise RuntimeError(
                    f"Unexpected execution phase in completed summary: {original_phase}"
                )
            value["execution_implementation_phase"] = value.get(
                "execution_implementation_phase", original_phase
            )
            value["phase"] = PHASE
        phase2.atomic_json(run_dir / name, value)

    audit = {
        "status": "passed",
        "checks": checks,
        "variant": variant,
        "run_id": run_id,
        "initial_classifier_head_sha256_float32": head_hash,
        "verified_at": phase2.utc_now(),
    }
    phase2.atomic_json(run_dir / "phase2e_completed_run_audit.json", audit)
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def inspect_variant(study_root: Path, phase1_root: Path, variant: str) -> dict[str, Any]:
    config, output_root = activate_variant(study_root, variant)
    args = argparse.Namespace(output_root=output_root, phase1_root=phase1_root)
    result = phase2d.inspect_phase2d(args)
    result.update(
        {
            "phase": PHASE,
            "study_id": STUDY_ID,
            "grid_fingerprint": grid_fingerprint(),
            "variant": variant,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "total_optimizer_steps": EXPECTED_TOTAL_STEPS,
            "warmup_steps": EXPECTED_WARMUP_STEPS,
        }
    )
    phase2.atomic_json(
        output_root / "configuration" / "preflight_manifest.json", result
    )
    return result


def validate_resume_checkpoint(
    output_root: Path, variant: str, resume: Path
) -> Path:
    """Accept only a complete, manifest-backed checkpoint for this exact trial."""

    run_dir = (Path(output_root) / "runs" / RUN_IDS[_validate_variant(variant)]).resolve()
    expected_parent = (run_dir / "checkpoints").resolve()
    checkpoint = Path(resume).resolve()
    if (
        checkpoint.parent != expected_parent
        or re.fullmatch(r"step-\d{6}", checkpoint.name) is None
        or not checkpoint.is_dir()
    ):
        raise RuntimeError(
            "Phase 2E resume checkpoint must be an existing checkpoint directory "
            f"inside {expected_parent}, got {checkpoint}"
        )
    required = (
        checkpoint / "model",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        checkpoint / "random_states.pt",
        checkpoint / "training_state.json",
    )
    if not all(path.is_dir() if path.name == "model" else path.is_file() for path in required):
        raise RuntimeError(f"Phase 2E resume checkpoint is incomplete: {checkpoint}")

    manifest_path = run_dir / "checkpoint_manifest.json"
    training_path = run_dir / "training_config.json"
    if not manifest_path.is_file() or not training_path.is_file():
        raise RuntimeError("Phase 2E resume requires run manifest and training config")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matching = [
        row for row in manifest if row.get("checkpoint_id") == checkpoint.name
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"Phase 2E resume checkpoint is not uniquely manifest-backed: {checkpoint}"
        )
    expected_step = int(checkpoint.name.removeprefix("step-"))
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    record = matching[0]
    if (
        int(record.get("global_step", -1)) != expected_step
        or int(state.get("global_step", -1)) != expected_step
        or state.get("resume_contract_sha256")
        != training.get("resume_contract_sha256")
        or record.get("experiment_fingerprint")
        != training.get("experiment_fingerprint")
    ):
        raise RuntimeError(
            f"Phase 2E manifest/checkpoint state mismatch at {checkpoint}"
        )
    return checkpoint


def latest_manifest_backed_checkpoint(output_root: Path, variant: str) -> Path:
    run_dir = Path(output_root) / "runs" / RUN_IDS[_validate_variant(variant)]
    manifest_path = run_dir / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Phase 2E run lacks checkpoint manifest: {manifest_path}")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for row in sorted(rows, key=lambda item: int(item["global_step"]), reverse=True):
        candidate = run_dir / "checkpoints" / str(row["checkpoint_id"])
        if not candidate.exists():
            continue
        try:
            return validate_resume_checkpoint(output_root, variant, candidate)
        except RuntimeError as error:
            failures.append(str(error))
    raise RuntimeError(
        "Phase 2E run has no complete manifest-backed checkpoint; "
        f"validation failures: {failures}"
    )


def _quarantine_unmanifested_checkpoints(output_root: Path, variant: str) -> list[str]:
    run_dir = Path(output_root) / "runs" / RUN_IDS[_validate_variant(variant)]
    manifest = json.loads(
        (run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    known = {str(row["checkpoint_id"]) for row in manifest}
    checkpoint_root = run_dir / "checkpoints"
    candidates = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name not in known
    ] if checkpoint_root.is_dir() else []
    if not candidates:
        return []
    stamp = re.sub(r"[^0-9A-Za-z]", "", phase2.utc_now())
    destination = (
        Path(output_root) / "recovery" / "orphan_checkpoints" / stamp
    )
    if destination.exists():
        raise RuntimeError(f"Phase 2E recovery destination already exists: {destination}")
    destination.mkdir(parents=True)
    moved = []
    for candidate in candidates:
        target = destination / candidate.name
        shutil.move(str(candidate), str(target))
        moved.append(str(target))
    phase2.atomic_json(
        destination / "quarantine.json",
        {
            "status": "recoverably_moved_unmanifested_checkpoints",
            "variant": variant,
            "moved": moved,
            "created_at": phase2.utc_now(),
        },
    )
    return moved


def train_variant(
    study_root: Path,
    phase1_root: Path,
    variant: str,
    resume: Path | None = None,
) -> dict[str, Any]:
    _, output_root = activate_variant(study_root, variant)
    if resume is not None:
        resume = validate_resume_checkpoint(output_root, variant, resume)
    args = argparse.Namespace(
        output_root=output_root,
        phase1_root=phase1_root,
        mode="full",
        run_id=RUN_IDS[variant],
        resume=resume,
        max_steps=None,
        per_class=2,
    )
    summary = phase2d.run_training(args)
    _augment_run_artifacts(study_root, variant)
    return summary


def resume_latest_variant(
    study_root: Path, phase1_root: Path, variant: str
) -> dict[str, Any]:
    _, output_root = activate_variant(study_root, variant)
    checkpoint = latest_manifest_backed_checkpoint(output_root, variant)
    quarantined = _quarantine_unmanifested_checkpoints(output_root, variant)
    result = train_variant(study_root, phase1_root, variant, checkpoint)
    result["phase2e_resume_checkpoint"] = str(checkpoint)
    result["phase2e_quarantined_unmanifested_checkpoints"] = quarantined
    return result


def _candidate_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = row["classification_metrics"]
    values = (
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
        float(metrics["weighted_f1"]),
        float(metrics["balanced_accuracy"]),
    )
    loss = float(row["validation_loss"])
    step = float(row["global_step"])
    learning_rate = float(row["learning_rate"])
    if (
        not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
        or not math.isfinite(loss)
        or loss < 0.0
        or not math.isfinite(step)
        or step <= 0.0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
    ):
        raise RuntimeError(f"Invalid Phase 2E selection candidate: {row}")
    return (*values, -loss, -step, -learning_rate)


def select_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Phase 2E selection requires candidate rows")
    return dict(max(rows, key=_candidate_key))


def _read_completed_candidates(study_root: Path) -> tuple[list[dict[str, Any]], dict]:
    all_rows: list[dict[str, Any]] = []
    per_variant: dict[str, Any] = {}
    head_hashes: dict[str, str] = {}
    for variant in VARIANT_ORDER:
        config, output_root = activate_variant(study_root, variant)
        run_id = RUN_IDS[variant]
        run_dir = output_root / "runs" / run_id
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "checkpoint_manifest.json"
        training_path = run_dir / "training_config.json"
        data_path = run_dir / "dataset_manifest.json"
        best_path = run_dir / "best_checkpoint.json"
        audit_path = run_dir / "phase2e_completed_run_audit.json"
        for path in (
            summary_path,
            manifest_path,
            training_path,
            data_path,
            best_path,
            audit_path,
        ):
            if not path.exists():
                raise RuntimeError(f"Phase 2E grid is incomplete: missing {path}")
        if (output_root / "retrieval" / "results.jsonl").exists():
            raise RuntimeError(
                "Phase 2E winner must be locked before any retrieval result exists"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        training = json.loads(training_path.read_text(encoding="utf-8"))
        data = json.loads(data_path.read_text(encoding="utf-8"))
        best = json.loads(best_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "complete"
            or summary.get("phase") != PHASE
            or audit.get("status") != "passed"
            or len(rows) != 5
        ):
            raise RuntimeError(f"Phase 2E {variant} is not a complete five-epoch run")
        steps = tuple(int(row["global_step"]) for row in rows)
        if steps != EXPECTED_STEPS:
            raise RuntimeError(f"Phase 2E {variant} checkpoint steps drifted: {steps}")
        epochs = tuple(int(row["epoch"]) for row in rows)
        if epochs != (1, 2, 3, 4, 5):
            raise RuntimeError(f"Phase 2E {variant} checkpoint epochs drifted: {epochs}")
        expected_data = {
            "train_oracle_sha256": TRAIN_ORACLE_SHA256,
            "validation_oracle_sha256": VALIDATION_ORACLE_SHA256,
            "active_train_examples": 2245,
            "active_validation_examples": 924,
        }
        for key, value in expected_data.items():
            if data.get(key) != value:
                raise RuntimeError(f"Phase 2E {variant} dataset mismatch at {key}")
        config_mismatches = {
            key: {"expected": value, "actual": training.get(key)}
            for key, value in asdict(config).items()
            if training.get(key) != value
        }
        runtime_expected = {
            "run_mode": "full",
            "active_per_device_batch_size": 4,
            "active_gradient_accumulation_steps": 8,
            "active_effective_batch_size": 32,
            "maximum_optimizer_steps": None,
            "total_optimizer_steps": EXPECTED_TOTAL_STEPS,
            "warmup_steps": EXPECTED_WARMUP_STEPS,
            "run_id": run_id,
            "transformers_version": "5.15.0.dev0",
            "transformers_commit": phase2d.TRANSFORMERS_COMMIT,
            "instruction_sha256": PROMPT_SHA256,
        }
        config_mismatches.update(
            {
                key: {"expected": value, "actual": training.get(key)}
                for key, value in runtime_expected.items()
                if training.get(key) != value
            }
        )
        if config_mismatches:
            raise RuntimeError(
                f"Phase 2E {variant} training configuration mismatch: "
                f"{config_mismatches}"
            )
        expected_fingerprint = trial_fingerprint(
            data, int(training["pad_token_id"]), variant
        )
        fingerprint_documents = [training, data, summary, best, *rows]
        if any(
            item.get("experiment_fingerprint") != expected_fingerprint
            for item in fingerprint_documents
        ):
            raise RuntimeError(f"Phase 2E {variant} fingerprint mismatch")
        head_hash = training["initial_model_loading_audit"]["classifier_head"][
            "head_weight_sha256_float32"
        ]
        head_hashes[variant] = head_hash
        augmented = []
        for row in rows:
            item = {
                **row,
                "phase": PHASE,
                "study_id": STUDY_ID,
                "variant": variant,
                "learning_rate": config.learning_rate,
                "run_id": run_id,
                "trial_root": str(output_root),
            }
            augmented.append(item)
            all_rows.append(item)
        selected = select_candidate_rows(augmented)
        selected_checkpoint = run_dir / "checkpoints" / selected["checkpoint_id"]
        if (
            selected["checkpoint_id"] != summary["selected_checkpoint_id"]
            or selected["checkpoint_id"] != best.get("checkpoint_id")
            or not selected_checkpoint.is_dir()
        ):
            raise RuntimeError(f"Phase 2E {variant} per-trial selection mismatch")
        per_variant[variant] = {
            "learning_rate": config.learning_rate,
            "run_id": run_id,
            "selected": selected,
            "all_epoch_candidates": augmented,
            "training_elapsed_seconds": summary["elapsed_seconds"],
            "initial_classifier_head_sha256_float32": head_hash,
        }
    if set(head_hashes.values()) != {EXPECTED_INITIAL_HEAD_SHA256}:
        raise RuntimeError(f"Phase 2E initial classifier-head hashes differ: {head_hashes}")
    return all_rows, per_variant


def select_grid_winner(study_root: Path) -> dict[str, Any]:
    root = _validate_study_root(study_root)
    prepare_study(root)
    candidates, per_variant = _read_completed_candidates(root)
    selected = select_candidate_rows(candidates)
    stable = {
        "status": "classification_winner_locked_before_retrieval",
        "phase": PHASE,
        "study_id": STUDY_ID,
        "grid_fingerprint": grid_fingerprint(),
        "selection_order": list(SELECTION_ORDER),
        "candidate_count": len(candidates),
        "trial_count": len(per_variant),
        "winner": selected,
        "per_variant": per_variant,
        "all_epoch_candidates": candidates,
        "initial_classifier_head_sha256_float32": EXPECTED_INITIAL_HEAD_SHA256,
        "validation_role": _stable_grid_protocol()["validation_role"],
        "retrieval_was_not_used_for_selection": True,
    }
    path = root / "comparison" / "selected_trial.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in stable}
        if comparable != stable:
            raise RuntimeError("Locked Phase 2E selection no longer recomputes exactly")
        result = existing
    else:
        result = {**stable, "locked_at": phase2.utc_now()}
        phase2.atomic_json(path, result)

    table_path = root / "comparison" / "lr_grid_metrics.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "learning_rate",
                "epoch",
                "global_step",
                "validation_ce",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy",
                "top_2_accuracy",
                "predicted_10",
                "predicted_20",
                "predicted_40",
                "predicted_80",
                "predicted_160",
                "selected_global",
            ]
        )
        for row in candidates:
            metrics = row["classification_metrics"]
            distribution = row["predicted_distribution"]
            writer.writerow(
                [
                    row["variant"],
                    row["learning_rate"],
                    row["epoch"],
                    row["global_step"],
                    row["validation_loss"],
                    metrics["accuracy"],
                    metrics["macro_f1"],
                    metrics["weighted_f1"],
                    metrics["balanced_accuracy"],
                    metrics["top_2_accuracy"],
                    *[distribution[str(label)] for label in phase2d.CHUNK_SIZES],
                    row["variant"] == selected["variant"]
                    and row["checkpoint_id"] == selected["checkpoint_id"],
                ]
            )
    return result


def _patch_final_artifacts(study_root: Path, variant: str) -> dict[str, Any]:
    config, output_root = activate_variant(study_root, variant)
    selection = json.loads(
        (Path(study_root) / "comparison" / "selected_trial.json").read_text(
            encoding="utf-8"
        )
    )
    winner = selection.get("winner", {})
    if (
        selection.get("status") != "classification_winner_locked_before_retrieval"
        or selection.get("phase") != PHASE
        or selection.get("study_id") != STUDY_ID
        or selection.get("grid_fingerprint") != grid_fingerprint()
        or winner.get("variant") != variant
    ):
        raise RuntimeError("Phase 2E locked selection identity mismatch")
    final_path = output_root / "final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    metrics_path = output_root / "classification" / "metrics.json"
    runtime_path = output_root / "validation" / "runtime_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if (
        final.get("selected_checkpoint_id") != winner.get("checkpoint_id")
        or final.get("experiment_fingerprint")
        != winner.get("experiment_fingerprint")
        or final.get("classification") != winner.get("classification_metrics")
        or metrics.get("selected_checkpoint_id") != winner.get("checkpoint_id")
        or runtime.get("selected_checkpoint_id") != winner.get("checkpoint_id")
    ):
        raise RuntimeError(
            "Phase 2E final validation artifacts do not match the locked winner"
        )
    fields = {
        "phase": "Phase 2E Base sequence-classification LR-grid fine-tuning",
        "study_id": STUDY_ID,
        "grid_fingerprint": grid_fingerprint(),
        "variant": variant,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "validation_role": _stable_grid_protocol()["validation_role"],
        "retrieval_was_not_used_for_selection": True,
        "global_grid_winner": True,
    }
    patched_final = {**final, **fields}
    patched_metrics = {**metrics, **fields}
    patched_runtime = {**runtime, **fields}
    result = {
        "status": "selected_checkpoint_final_validation_complete",
        **fields,
        "run_id": RUN_IDS[variant],
        "selected_checkpoint_id": patched_final["selected_checkpoint_id"],
        "classification": patched_final["classification"],
        "predicted_distribution": patched_final["predicted_distribution"],
        "oracle_distribution": patched_final["oracle_distribution"],
        "runtime": patched_runtime,
        "trial_final_summary": str(final_path),
        "selection_artifact": str(
            Path(study_root) / "comparison" / "selected_trial.json"
        ),
        "completed_at": phase2.utc_now(),
    }
    phase2.atomic_json(final_path, patched_final)
    phase2.atomic_json(metrics_path, patched_metrics)
    phase2.atomic_json(runtime_path, patched_runtime)
    phase2.atomic_json(
        Path(study_root) / "comparison" / "selected_final_summary.json", result
    )
    return result


def final_validate_selected(study_root: Path, phase1_root: Path) -> dict[str, Any]:
    selection = select_grid_winner(study_root)
    variant = _validate_variant(selection["winner"]["variant"])
    _, output_root = activate_variant(study_root, variant)
    args = argparse.Namespace(
        output_root=output_root,
        phase1_root=phase1_root,
        run_id=RUN_IDS[variant],
    )
    phase2d.final_validation(args)
    return _patch_final_artifacts(study_root, variant)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--phase1-root", type=Path, default=phase2.PHASE1_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--variant", choices=VARIANT_ORDER, required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--variant", choices=VARIANT_ORDER, required=True)
    train_parser.add_argument("--resume", type=Path)
    resume_parser = subparsers.add_parser("resume-latest")
    resume_parser.add_argument("--variant", choices=VARIANT_ORDER, required=True)
    audit_parser = subparsers.add_parser("audit-completed")
    audit_parser.add_argument("--variant", choices=VARIANT_ORDER, required=True)
    subparsers.add_parser("select")
    subparsers.add_parser("final-selected")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_study(args.study_root)
    elif args.command == "inspect":
        result = inspect_variant(
            args.study_root, args.phase1_root, args.variant
        )
    elif args.command == "train":
        result = train_variant(
            args.study_root, args.phase1_root, args.variant, args.resume
        )
    elif args.command == "resume-latest":
        result = resume_latest_variant(
            args.study_root, args.phase1_root, args.variant
        )
    elif args.command == "audit-completed":
        result = _augment_run_artifacts(args.study_root, args.variant)
    elif args.command == "select":
        result = select_grid_winner(args.study_root)
    elif args.command == "final-selected":
        result = final_validate_selected(args.study_root, args.phase1_root)
    else:  # pragma: no cover
        raise RuntimeError(f"Unsupported Phase 2E command: {args.command}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
