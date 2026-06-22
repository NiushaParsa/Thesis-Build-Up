#!/usr/bin/env python
"""Run fixed-separate retrieval, persist optional records, and build oracle labels."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue
from tqdm import tqdm

from config import (
    CHUNK_SIZES,
    EMBEDDING_DIM,
    EVALUATION_OUTPUT_DIR,
    EVALUATION_UPSERT_BATCH_SIZE,
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
from qdrant_schema import ensure_evaluation_collections, get_qdrant_client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

METHODS = {"fixed-separate": METHOD_NAME}


def load_questions(client, collection_name: str, split=None, limit=None):
    """Load question payloads and vectors, preserving their stored split."""
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
    parser = argparse.ArgumentParser(description="Evaluate fixed-separate QASPER retrieval")
    parser.add_argument("--method", choices=list(METHODS), default="fixed-separate")
    parser.add_argument("--split", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--store-text", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluation-collection", default=None)
    parser.add_argument("--router-dataset-collection", default=None)
    parser.add_argument("--upsert-batch-size", type=int, default=None)
    parser.add_argument("--tie-epsilon", type=float, default=None)
    parser.add_argument("--evaluation-run-id", default=None)
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
    if args.top_k < 1 or args.upsert_batch_size < 1 or args.log_every < 1:
        parser.error("top-k, upsert-batch-size, and log-every must be positive")
    if args.tie_epsilon < 0:
        parser.error("tie-epsilon cannot be negative")
    return args


def _make_upserter(client, collection: str, batch_size: int):
    return BufferedQdrantUpserter(client, collection, batch_size)


def main() -> None:
    args = parse_args()
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_timestamp = datetime.now(timezone.utc).isoformat()
    evaluation_run_id = args.evaluation_run_id or new_evaluation_run_id()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = args.output_dir / f"RetrievalEvalFixedSeparate_{timestamp_tag}.jsonl"
    router_path = args.output_dir / f"RouterDataset_{timestamp_tag}.jsonl"
    incomplete_path = args.output_dir / f"IncompleteEvaluation_{timestamp_tag}.jsonl"

    evaluation_config = build_evaluation_config(
        method=METHOD_NAME,
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
                    create_router=args.persist_router_dataset,
                )
                if args.persist_evaluations:
                    evaluation_upserter = _make_upserter(
                        client, args.evaluation_collection, args.upsert_batch_size
                    )
                if args.persist_router_dataset:
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

        with (
            evaluation_path.open("w", encoding="utf-8") as evaluation_file,
            router_path.open("w", encoding="utf-8") as router_file,
            incomplete_path.open("w", encoding="utf-8") as incomplete_file,
        ):
            for question_index, question in enumerate(
                tqdm(questions, desc="Evaluating"), start=1
            ):
                try:
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
                    skipped_no_evidence += 1
                    incomplete_file.write(
                        json.dumps(
                            {
                                "question_id": question["point_id"],
                                "document_id": question["document_id"],
                                "split": question["split"],
                                "reason": "no_valid_evidence",
                                "evaluation_run_id": evaluation_run_id,
                                "evaluation_config_hash": config_hash,
                            }
                        )
                        + "\n"
                    )
                    continue

                for record in records:
                    evaluation_file.write(json.dumps(record) + "\n")
                    level = record["granularity_level"]
                    per_gran_count[level] += 1
                    per_gran_f1[level].append(record["f1_joined_topk"])
                    per_gran_similarity[level].append(
                        record["mean_max_evidence_similarity_topk"]
                    )
                    if evaluation_upserter:
                        evaluation_upserter.add(
                            point_id=record["eval_id"], payload=record, vector={}
                        )

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
        for level in levels:
            f1_values = per_gran_f1[level]
            similarity_values = per_gran_similarity[level]
            logger.info(
                "level=%d tokens=%d records=%d mean_joined_f1=%.6f mean_evidence_similarity=%.6f",
                level,
                CHUNK_SIZES[level - 1],
                per_gran_count[level],
                sum(f1_values) / len(f1_values) if f1_values else 0.0,
                sum(similarity_values) / len(similarity_values)
                if similarity_values
                else 0.0,
            )
        logger.info("best_f1_granularity_distribution=%s", dict(best_f1_distribution))
        logger.info(
            "best_similarity_granularity_distribution=%s",
            dict(best_similarity_distribution),
        )
        logger.info("router_target_distribution=%s", dict(router_distribution))
        logger.info("f1_similarity_label_disagreements=%d", disagreements)
        logger.info("evaluation_jsonl=%s", evaluation_path)
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
