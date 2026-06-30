#!/usr/bin/env python
"""Run registered QASPER retrieval evaluators and persist schema-v2 results."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue
from tqdm import tqdm

from config import (
    CHUNK_SIZES,
    EMBEDDING_DIM,
    EVALUATION_OUTPUT_DIR,
    EVALUATION_UPSERT_BATCH_SIZE,
    MIXED_DEDUP_CANDIDATE_MULTIPLIER,
    MIXED_DEDUP_OVERLAP_THRESHOLD,
    OPENAI_EMBEDDING_MODEL,
    PAPER_CHUNK_COLLECTION,
    PAPER_EVIDENCE_COLLECTION,
    PAPER_QUESTION_COLLECTION,
    PERSIST_EVALUATIONS,
    PERSIST_ROUTER_DATASET,
    RETRIEVAL_EVALUATION_COLLECTION,
    RETRIEVAL_TOP_K,
    ROUTER_DATASET_COLLECTION,
    ROUTER_LABEL_TIE_EPSILON,
    TOKENIZER_NAME,
)
from evaluation_utils import (
    BufferedQdrantUpserter,
    METHOD_NAME,
    build_evaluation_config,
    build_router_record,
    evaluation_config_hash,
    new_evaluation_run_id,
)
from fixed_sized_granularity_separate import evaluate_question
from mixed_granularity import (
    MIXED_DEDUPLICATED_METHOD,
    MIXED_FILTER_BEHAVIOR,
    MIXED_RAW_METHOD,
    OVERLAP_DEFINITION,
    evaluate_mixed_question,
)
from qdrant_schema import ensure_evaluation_collections, get_qdrant_client
from router_selected import (
    ROUTER_SELECTED_CLI_METHOD,
    ROUTER_SELECTED_FILTER_BEHAVIOR,
    ROUTER_SELECTED_METHOD,
    RouterPredictor,
    build_router_selected_config,
    evaluate_router_selected_question,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

METHODS = {
    "fixed-separate": METHOD_NAME,
    MIXED_RAW_METHOD: MIXED_RAW_METHOD,
    MIXED_DEDUPLICATED_METHOD: MIXED_DEDUPLICATED_METHOD,
    ROUTER_SELECTED_CLI_METHOD: ROUTER_SELECTED_METHOD,
}
OUTPUT_STEMS = {
    "fixed-separate": "RetrievalEvalFixedSeparate",
    MIXED_RAW_METHOD: "RetrievalEvalMixedRaw",
    MIXED_DEDUPLICATED_METHOD: "RetrievalEvalMixedDeduplicated",
    ROUTER_SELECTED_CLI_METHOD: "RetrievalEvalRouterSelected",
}


def load_questions(client, collection_name: str, split=None, limit=None, question_ids=None):
    """Load question payloads and vectors, preserving their stored split."""
    if question_ids:
        points = client.retrieve(
            collection_name=collection_name,
            ids=question_ids,
            with_payload=True,
            with_vectors=True,
        )
        questions = []
        for point in points:
            payload = point.payload or {}
            if split and payload.get("split") != split:
                continue
            questions.append(
                {
                    "point_id": str(point.id),
                    "vector": point.vector,
                    "document_id": payload.get("document_id", ""),
                    "question_text": payload.get("question_text", ""),
                    "split": payload.get("split", ""),
                }
            )
        return questions[:limit] if limit is not None else questions

    scroll_filter = None
    if split:
        scroll_filter = Filter(
            must=[FieldCondition(key="split", match=MatchValue(value=split))]
        )
    questions = []
    offset = None
    while True:
        remaining = None if limit is None else limit - len(questions)
        if remaining is not None and remaining <= 0:
            break
        batch_size = 100 if remaining is None else min(100, remaining)
        results, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in results:
            payload = point.payload or {}
            questions.append(
                {
                    "point_id": str(point.id),
                    "vector": point.vector,
                    "document_id": payload.get("document_id", ""),
                    "question_text": payload.get("question_text", ""),
                    "split": payload.get("split", ""),
                }
            )
        if next_offset is None or not results:
            break
        offset = next_offset
    return questions


def _add_boolean_switch(parser, name: str, destination: str, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=destination, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}",
        dest=destination,
        action="store_false",
        help=f"Disable {help_text.lower()}",
    )
    parser.set_defaults(**{destination: None})


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate registered QASPER retrieval methods")
    parser.add_argument("--method", choices=list(METHODS), default="fixed-separate")
    parser.add_argument("--split", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=None)
    parser.add_argument("--question-ids-file", type=Path, default=None)
    parser.add_argument("--store-text", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluation-collection", default=None)
    parser.add_argument("--router-dataset-collection", default=None)
    parser.add_argument("--upsert-batch-size", type=int, default=None)
    parser.add_argument("--tie-epsilon", type=float, default=None)
    parser.add_argument("--overlap-threshold", type=float, default=None)
    parser.add_argument("--dedup-candidate-multiplier", type=int, default=None)
    parser.add_argument("--evaluation-run-id", default=None)
    parser.add_argument(
        "--evaluation-config-hash",
        default=None,
        help=(
            "For router-selected evaluation, the frozen fixed-separate oracle "
            "configuration hash used to join RouterDataset labels."
        ),
    )
    parser.add_argument("--router-model", type=Path, default=None)
    parser.add_argument("--router-model-choice", default="primary")
    _add_boolean_switch(
        parser,
        "persist-evaluations",
        "persist_evaluations",
        "Persist granularity evaluation records to Qdrant",
    )
    _add_boolean_switch(
        parser,
        "persist-router-dataset",
        "persist_router_dataset",
        "Persist complete question-level router records to Qdrant",
    )
    args = parser.parse_args()
    args.top_k = RETRIEVAL_TOP_K if args.top_k is None else args.top_k
    args.output_dir = Path(EVALUATION_OUTPUT_DIR) if args.output_dir is None else args.output_dir
    args.evaluation_collection = (
        RETRIEVAL_EVALUATION_COLLECTION
        if args.evaluation_collection is None
        else args.evaluation_collection
    )
    args.router_dataset_collection = (
        ROUTER_DATASET_COLLECTION
        if args.router_dataset_collection is None
        else args.router_dataset_collection
    )
    args.upsert_batch_size = (
        EVALUATION_UPSERT_BATCH_SIZE
        if args.upsert_batch_size is None
        else args.upsert_batch_size
    )
    args.tie_epsilon = (
        ROUTER_LABEL_TIE_EPSILON if args.tie_epsilon is None else args.tie_epsilon
    )
    args.overlap_threshold = (
        MIXED_DEDUP_OVERLAP_THRESHOLD
        if args.overlap_threshold is None
        else args.overlap_threshold
    )
    args.dedup_candidate_multiplier = (
        MIXED_DEDUP_CANDIDATE_MULTIPLIER
        if args.dedup_candidate_multiplier is None
        else args.dedup_candidate_multiplier
    )
    args.persist_evaluations = (
        PERSIST_EVALUATIONS
        if args.persist_evaluations is None
        else args.persist_evaluations
    )
    args.persist_router_dataset = (
        PERSIST_ROUTER_DATASET
        if args.persist_router_dataset is None
        else args.persist_router_dataset
    )
    if (
        args.top_k < 1
        or args.upsert_batch_size < 1
        or args.log_every < 1
        or args.dedup_candidate_multiplier < 1
    ):
        parser.error(
            "top-k, upsert-batch-size, log-every, and dedup-candidate-multiplier "
            "must be positive"
        )
    if args.tie_epsilon < 0:
        parser.error("tie-epsilon cannot be negative")
    if not 0.0 < args.overlap_threshold <= 1.0:
        parser.error("overlap-threshold must be in (0, 1]")
    if args.method != "fixed-separate" and args.persist_router_dataset:
        parser.error("router-dataset persistence is available only for fixed-separate")
    if args.method == ROUTER_SELECTED_CLI_METHOD and not args.router_model:
        parser.error("router-selected evaluation requires --router-model")
    if args.method != ROUTER_SELECTED_CLI_METHOD and args.evaluation_config_hash:
        parser.error("--evaluation-config-hash is reserved for router-selected oracle joins")
    question_ids = []
    if args.question_id:
        question_ids.extend(args.question_id)
    if args.question_ids_file:
        if not args.question_ids_file.exists():
            parser.error(f"question IDs file does not exist: {args.question_ids_file}")
        question_ids.extend(
            line.strip()
            for line in args.question_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    args.question_ids = question_ids or None
    return args


def _make_upserter(client, collection: str, batch_size: int):
    return BufferedQdrantUpserter(client, collection, batch_size)


def main() -> None:
    args = parse_args()
    is_fixed_separate = args.method == "fixed-separate"
    is_mixed = args.method in {MIXED_RAW_METHOD, MIXED_DEDUPLICATED_METHOD}
    is_router_selected = args.method == ROUTER_SELECTED_CLI_METHOD
    method_name = METHODS[args.method]
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_timestamp = datetime.now(timezone.utc).isoformat()
    evaluation_run_id = args.evaluation_run_id or new_evaluation_run_id()
    router_predictor = None
    oracle_evaluation_config_hash = None
    if is_router_selected:
        router_predictor = RouterPredictor.from_path(
            args.router_model, model_choice=args.router_model_choice
        )
        oracle_evaluation_config_hash = (
            args.evaluation_config_hash
            or router_predictor.oracle_evaluation_config_hash
        )
        if not oracle_evaluation_config_hash:
            raise ValueError(
                "router-selected evaluation requires an oracle evaluation config hash "
                "via --evaluation-config-hash or router artifact metadata"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = args.output_dir / f"{OUTPUT_STEMS[args.method]}_{timestamp_tag}.jsonl"
    router_path = (
        args.output_dir / f"RouterDataset_{timestamp_tag}.jsonl"
        if is_fixed_separate
        else None
    )
    incomplete_path = args.output_dir / f"IncompleteEvaluation_{timestamp_tag}.jsonl"

    strategy_settings = None
    filter_behavior = None
    if is_mixed:
        filter_behavior = MIXED_FILTER_BEHAVIOR
        strategy_settings = {"variant": args.method}
        if args.method == MIXED_DEDUPLICATED_METHOD:
            strategy_settings.update(
                {
                    "overlap_threshold": args.overlap_threshold,
                    "overlap_definition": OVERLAP_DEFINITION,
                    "candidate_multiplier": args.dedup_candidate_multiplier,
                }
            )
    elif is_router_selected:
        filter_behavior = ROUTER_SELECTED_FILTER_BEHAVIOR
        strategy_settings = {
            "router_model_path": router_predictor.model_path,
            "router_model_hash": router_predictor.model_hash,
            "router_model_version": router_predictor.model_version,
            "router_model_choice": router_predictor.model_choice,
            "router_selected_model_type": router_predictor.selected_model_type,
            "oracle_evaluation_config_hash": oracle_evaluation_config_hash,
        }

    if is_router_selected:
        evaluation_config = build_router_selected_config(
            top_k=args.top_k,
            chunk_sizes=CHUNK_SIZES,
            router_predictor=router_predictor,
            oracle_evaluation_config_hash=oracle_evaluation_config_hash,
            store_text=args.store_text,
            chunk_collection=PAPER_CHUNK_COLLECTION,
            question_collection=PAPER_QUESTION_COLLECTION,
            evidence_collection=PAPER_EVIDENCE_COLLECTION,
            evaluation_collection=args.evaluation_collection,
            router_collection=args.router_dataset_collection,
            embedding_model=OPENAI_EMBEDDING_MODEL,
            embedding_dimension=EMBEDDING_DIM,
            tokenizer_name=TOKENIZER_NAME,
            tie_epsilon=args.tie_epsilon,
        )
    else:
        evaluation_config = build_evaluation_config(
            method=method_name,
            top_k=args.top_k,
            chunk_sizes=CHUNK_SIZES,
            embedding_model=OPENAI_EMBEDDING_MODEL,
            embedding_dimension=EMBEDDING_DIM,
            tokenizer=TOKENIZER_NAME,
            chunk_collection=PAPER_CHUNK_COLLECTION,
            question_collection=PAPER_QUESTION_COLLECTION,
            evidence_collection=PAPER_EVIDENCE_COLLECTION,
            evaluation_collection=args.evaluation_collection,
            router_collection=args.router_dataset_collection,
            router_label_tie_epsilon=args.tie_epsilon,
            store_text=args.store_text,
            **({"filter_behavior": filter_behavior} if filter_behavior else {}),
            strategy_settings=strategy_settings,
        )
    config_hash = evaluation_config_hash(evaluation_config)
    levels = list(range(1, len(CHUNK_SIZES) + 1))

    client = get_qdrant_client()
    evaluation_upserter = None
    router_upserter = None
    persistence_setup_errors = []
    try:
        if args.persist_evaluations or args.persist_router_dataset:
            try:
                ensure_evaluation_collections(
                    client,
                    evaluation_collection=args.evaluation_collection,
                    router_collection=args.router_dataset_collection,
                    create_evaluation=args.persist_evaluations,
                    create_router=is_fixed_separate and args.persist_router_dataset,
                )
                if args.persist_evaluations:
                    evaluation_upserter = _make_upserter(
                        client, args.evaluation_collection, args.upsert_batch_size
                    )
                if is_fixed_separate and args.persist_router_dataset:
                    router_upserter = _make_upserter(
                        client, args.router_dataset_collection, args.upsert_batch_size
                    )
            except Exception as exc:
                message = f"Persistence setup failed; JSONL evaluation will continue: {exc}"
                logger.exception(message)
                persistence_setup_errors.append(message)

        questions = load_questions(
            client,
            PAPER_QUESTION_COLLECTION,
            split=args.split,
            limit=args.limit,
            question_ids=args.question_ids,
        )
        logger.info(
            "Loaded %d questions; run=%s config=%s top_k=%d",
            len(questions),
            evaluation_run_id,
            config_hash,
            args.top_k,
        )

        per_gran_f1: Dict[int, list] = defaultdict(list)
        per_gran_similarity: Dict[int, list] = defaultdict(list)
        per_gran_count = Counter()
        best_f1_distribution = Counter()
        best_similarity_distribution = Counter()
        router_distribution = Counter()
        disagreements = 0
        skipped_no_evidence = 0
        incomplete_questions = 0
        evaluated_questions = 0
        mixed_f1_values = []
        mixed_similarity_values = []
        mixed_granularity_counts = Counter()
        router_selected_f1_values = []
        router_selected_regrets = []
        router_selected_latencies = []
        router_selected_predictions = Counter()
        router_oracle_matches = 0
        router_oracle_compared = 0
        router_missing_oracles = 0

        with ExitStack() as stack:
            evaluation_file = stack.enter_context(
                evaluation_path.open("w", encoding="utf-8")
            )
            incomplete_file = stack.enter_context(
                incomplete_path.open("w", encoding="utf-8")
            )
            router_file = (
                stack.enter_context(router_path.open("w", encoding="utf-8"))
                if router_path
                else None
            )
            for question_index, question in enumerate(
                tqdm(questions, desc="Evaluating"), start=1
            ):
                no_records_reason = "no_valid_evidence"
                try:
                    if is_fixed_separate:
                        records = list(
                            evaluate_question(
                                client=client,
                                question_point_id=question["point_id"],
                                question_vector=question["vector"],
                                document_id=question["document_id"],
                                question_text=question["question_text"],
                                split=question["split"],
                                top_k=args.top_k,
                                granularity_levels=levels,
                                store_retrieved_text=args.store_text,
                                chunk_sizes=CHUNK_SIZES,
                                chunk_collection=PAPER_CHUNK_COLLECTION,
                                question_collection=PAPER_QUESTION_COLLECTION,
                                evidence_collection=PAPER_EVIDENCE_COLLECTION,
                                embedding_model=OPENAI_EMBEDDING_MODEL,
                                embedding_dimension=EMBEDDING_DIM,
                                tokenizer_name=TOKENIZER_NAME,
                                evaluation_run_id=evaluation_run_id,
                                evaluation_config_hash=config_hash,
                                evaluation_timestamp=evaluation_timestamp,
                            )
                        )
                    elif is_mixed:
                        mixed_record = evaluate_mixed_question(
                            client=client,
                            question_point_id=question["point_id"],
                            question_vector=question["vector"],
                            document_id=question["document_id"],
                            question_text=question["question_text"],
                            split=question["split"],
                            top_k=args.top_k,
                            variant=args.method,
                            store_retrieved_text=args.store_text,
                            chunk_sizes=CHUNK_SIZES,
                            chunk_collection=PAPER_CHUNK_COLLECTION,
                            question_collection=PAPER_QUESTION_COLLECTION,
                            evidence_collection=PAPER_EVIDENCE_COLLECTION,
                            evaluation_collection=args.evaluation_collection,
                            router_collection=args.router_dataset_collection,
                            embedding_model=OPENAI_EMBEDDING_MODEL,
                            embedding_dimension=EMBEDDING_DIM,
                            tokenizer_name=TOKENIZER_NAME,
                            overlap_threshold=args.overlap_threshold,
                            candidate_multiplier=args.dedup_candidate_multiplier,
                            evaluation_run_id=evaluation_run_id,
                            evaluation_config_hash=config_hash,
                            evaluation_timestamp=evaluation_timestamp,
                        )
                        records = [mixed_record] if mixed_record else []
                    else:
                        router_record, incomplete_reason = evaluate_router_selected_question(
                            client=client,
                            question_point_id=question["point_id"],
                            question_vector=question["vector"],
                            document_id=question["document_id"],
                            question_text=question["question_text"],
                            split=question["split"],
                            router_predictor=router_predictor,
                            oracle_evaluation_config_hash=oracle_evaluation_config_hash,
                            top_k=args.top_k,
                            store_retrieved_text=args.store_text,
                            chunk_sizes=CHUNK_SIZES,
                            chunk_collection=PAPER_CHUNK_COLLECTION,
                            question_collection=PAPER_QUESTION_COLLECTION,
                            evidence_collection=PAPER_EVIDENCE_COLLECTION,
                            evaluation_collection=args.evaluation_collection,
                            router_collection=args.router_dataset_collection,
                            embedding_model=OPENAI_EMBEDDING_MODEL,
                            embedding_dimension=EMBEDDING_DIM,
                            tokenizer_name=TOKENIZER_NAME,
                            evaluation_run_id=evaluation_run_id,
                            evaluation_config_hash=config_hash,
                            evaluation_timestamp=evaluation_timestamp,
                        )
                        if router_record is None:
                            records = []
                            no_records_reason = (
                                incomplete_reason or "router_selected_incomplete"
                            )
                        else:
                            records = [router_record]
                except Exception as exc:
                    incomplete_questions += 1
                    logger.exception(
                        "Question %s evaluation failed", question["point_id"]
                    )
                    incomplete_file.write(
                        json.dumps(
                            {
                                "question_id": question["point_id"],
                                "document_id": question["document_id"],
                                "split": question["split"],
                                "reason": f"evaluation_error:{exc}",
                                "evaluation_run_id": evaluation_run_id,
                                "evaluation_config_hash": config_hash,
                            }
                        )
                        + "\n"
                    )
                    continue

                if not records:
                    if no_records_reason == "no_valid_evidence":
                        skipped_no_evidence += 1
                    else:
                        incomplete_questions += 1
                    incomplete_file.write(
                        json.dumps(
                            {
                                "question_id": question["point_id"],
                                "document_id": question["document_id"],
                                "split": question["split"],
                                "reason": no_records_reason,
                                "evaluation_run_id": evaluation_run_id,
                                "evaluation_config_hash": config_hash,
                            }
                        )
                        + "\n"
                    )
                    continue

                for record in records:
                    evaluation_file.write(json.dumps(record) + "\n")
                    if is_fixed_separate:
                        level = record["granularity_level"]
                        per_gran_count[level] += 1
                        per_gran_f1[level].append(record["f1_joined_topk"])
                        per_gran_similarity[level].append(
                            record["mean_max_evidence_similarity_topk"]
                        )
                    elif is_mixed:
                        mixed_f1_values.append(record["f1_joined_topk"])
                        mixed_similarity_values.append(
                            record["mean_max_evidence_similarity_topk"]
                        )
                        for item in record["granularity_composition"]:
                            mixed_granularity_counts[item["granularity_level"]] += item[
                                "count"
                            ]
                    elif is_router_selected:
                        router_selected_f1_values.append(record["f1_joined_topk"])
                        if record.get("regret_f1") is not None:
                            router_selected_regrets.append(record["regret_f1"])
                        router_selected_latencies.append(record.get("total_latency_ms", 0.0))
                        router_selected_predictions[
                            record["predicted_granularity_tokens"]
                        ] += 1
                        if record.get("router_oracle_match") is not None:
                            router_oracle_compared += 1
                            router_oracle_matches += int(record["router_oracle_match"])
                        if record.get("oracle_lookup_status") in {"missing", "duplicate"}:
                            router_missing_oracles += 1
                    if evaluation_upserter:
                        evaluation_upserter.add(
                            point_id=record["eval_id"], payload=record, vector={}
                        )

                if not is_fixed_separate:
                    evaluated_questions += 1
                    if question_index % args.log_every == 0:
                        logger.info(
                            "Processed %d/%d questions (%d evaluation records)",
                            question_index,
                            len(questions),
                            evaluated_questions,
                        )
                    continue

                router_record, incomplete_reason = build_router_record(
                    question=question,
                    records=records,
                    expected_levels=levels,
                    tie_epsilon=args.tie_epsilon,
                    evaluation_run_id=evaluation_run_id,
                    config_hash=config_hash,
                    embedding_model=OPENAI_EMBEDDING_MODEL,
                    embedding_dimension=EMBEDDING_DIM,
                )
                if router_record is None:
                    incomplete_questions += 1
                    incomplete_file.write(
                        json.dumps(
                            {
                                "question_id": question["point_id"],
                                "document_id": question["document_id"],
                                "split": question["split"],
                                "reason": incomplete_reason,
                                "evaluation_run_id": evaluation_run_id,
                                "evaluation_config_hash": config_hash,
                            }
                        )
                        + "\n"
                    )
                    continue

                evaluated_questions += 1
                router_file.write(json.dumps(router_record) + "\n")
                best_f1_distribution[router_record["best_granularity_by_f1"]] += 1
                best_similarity_distribution[
                    router_record["best_granularity_by_evidence_similarity"]
                ] += 1
                router_distribution[router_record["router_target_granularity"]] += 1
                disagreements += int(
                    router_record["best_granularity_by_f1"]
                    != router_record["best_granularity_by_evidence_similarity"]
                )
                if router_upserter:
                    router_upserter.add(
                        point_id=router_record["router_record_id"],
                        payload=router_record,
                        vector=question["vector"],
                    )

                if question_index % args.log_every == 0:
                    logger.info(
                        "Processed %d/%d questions (%d complete router examples)",
                        question_index,
                        len(questions),
                        evaluated_questions,
                    )

        if evaluation_upserter:
            evaluation_upserter.flush()
        if router_upserter:
            router_upserter.flush()

        logger.info("=" * 68)
        logger.info("EVALUATION COMPLETE")
        logger.info("evaluated_questions=%d", evaluated_questions)
        logger.info("skipped_questions_without_evidence=%d", skipped_no_evidence)
        logger.info("incomplete_questions=%d", incomplete_questions)
        if is_fixed_separate:
            for level in levels:
                f1_values = per_gran_f1[level]
                similarity_values = per_gran_similarity[level]
                logger.info(
                    "level=%d tokens=%d records=%d mean_joined_f1=%.6f "
                    "mean_evidence_similarity=%.6f",
                    level,
                    CHUNK_SIZES[level - 1],
                    per_gran_count[level],
                    sum(f1_values) / len(f1_values) if f1_values else 0.0,
                    sum(similarity_values) / len(similarity_values)
                    if similarity_values
                    else 0.0,
                )
            logger.info(
                "best_f1_granularity_distribution=%s", dict(best_f1_distribution)
            )
            logger.info(
                "best_similarity_granularity_distribution=%s",
                dict(best_similarity_distribution),
            )
            logger.info("router_target_distribution=%s", dict(router_distribution))
            logger.info("f1_similarity_label_disagreements=%d", disagreements)
        elif is_mixed:
            logger.info(
                "method=%s records=%d mean_joined_f1=%.6f "
                "mean_evidence_similarity=%.6f",
                method_name,
                len(mixed_f1_values),
                sum(mixed_f1_values) / len(mixed_f1_values)
                if mixed_f1_values
                else 0.0,
                sum(mixed_similarity_values) / len(mixed_similarity_values)
                if mixed_similarity_values
                else 0.0,
            )
            logger.info(
                "retrieved_granularity_composition=%s",
                {
                    CHUNK_SIZES[level - 1]: mixed_granularity_counts[level]
                    for level in levels
                },
            )
        else:
            logger.info(
                "method=%s records=%d mean_joined_f1=%.6f median_joined_f1=%.6f",
                method_name,
                len(router_selected_f1_values),
                sum(router_selected_f1_values) / len(router_selected_f1_values)
                if router_selected_f1_values
                else 0.0,
                median(router_selected_f1_values)
                if router_selected_f1_values
                else 0.0,
            )
            logger.info(
                "predicted_granularity_distribution=%s",
                dict(router_selected_predictions),
            )
            logger.info(
                "router_oracle_match_rate=%.6f compared=%d missing_oracle_records=%d",
                router_oracle_matches / router_oracle_compared
                if router_oracle_compared
                else 0.0,
                router_oracle_compared,
                router_missing_oracles,
            )
            logger.info(
                "mean_regret_f1=%.6f",
                sum(router_selected_regrets) / len(router_selected_regrets)
                if router_selected_regrets
                else 0.0,
            )
            logger.info(
                "latency_ms mean=%.2f median=%.2f max=%.2f",
                sum(router_selected_latencies) / len(router_selected_latencies)
                if router_selected_latencies
                else 0.0,
                median(router_selected_latencies)
                if router_selected_latencies
                else 0.0,
                max(router_selected_latencies) if router_selected_latencies else 0.0,
            )
        logger.info("evaluation_jsonl=%s", evaluation_path)
        if router_path:
            logger.info("router_jsonl=%s", router_path)
        logger.info("incomplete_jsonl=%s", incomplete_path)
        logger.info(
            "qdrant_evaluation_records_upserted=%d",
            evaluation_upserter.upserted if evaluation_upserter else 0,
        )
        logger.info(
            "qdrant_router_records_upserted=%d",
            router_upserter.upserted if router_upserter else 0,
        )
        persistence_errors = list(persistence_setup_errors)
        if evaluation_upserter:
            persistence_errors.extend(evaluation_upserter.errors)
        if router_upserter:
            persistence_errors.extend(router_upserter.errors)
        if persistence_errors:
            logger.error("Persistence completed with errors: %s", persistence_errors)
        logger.info("=" * 68)
    finally:
        client.close()


if __name__ == "__main__":
    main()
