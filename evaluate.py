#!/usr/bin/env python
"""
Retrieval Evaluation – Orchestrator
====================================
Loads questions from Qdrant's ``PaperQuestion`` collection, dispatches
them to one or more evaluation methods, and writes per-record JSONL
output plus an end-of-run summary.

Usage
-----
::

    python evaluate.py                              # default: all splits, K=5
    python evaluate.py --split test --top-k 10      # test split, K=10
    python evaluate.py --limit 50                   # quick test with 50 questions
    python evaluate.py --store-text                 # include full texts in output
    python evaluate.py --method fixed-separate      # explicit method selection

Output
------
``outputs/RetrievalEvalFixedSeparate_<YYYYMMDD_HHMMSS>.jsonl``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)
from tqdm import tqdm

from config import CHUNK_SIZES
from qdrant_schema import get_qdrant_client
from fixed_sized_granularity_separate import evaluate_question

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Method registry (easy to extend later)
METHODS = {
    "fixed-separate": "fixed-sized granularity - separate",
}


# ── Load questions from Qdrant ───────────────────────────
def load_questions(client, split=None, limit=None):
    """Scroll through PaperQuestion and return a list of dicts.

    Each dict: {point_id, vector, document_id, question_text, split}.
    """
    scroll_filter = None
    if split:
        scroll_filter = Filter(must=[
            FieldCondition(key="split", match=MatchValue(value=split)),
        ])

    questions = []
    offset = None
    batch_size = 100

    while True:
        results, next_offset = client.scroll(
            collection_name="PaperQuestion",
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in results:
            questions.append({
                "point_id":      str(point.id),
                "vector":        point.vector,
                "document_id":   (point.payload or {}).get("document_id", ""),
                "question_text": (point.payload or {}).get("question_text", ""),
                "split":         (point.payload or {}).get("split", ""),
            })
        if next_offset is None:
            break
        offset = next_offset

        if limit and len(questions) >= limit:
            questions = questions[:limit]
            break

    return questions


# ── Main ─────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval methods on QASPER (Qdrant)",
    )
    parser.add_argument(
        "--method", choices=list(METHODS.keys()), default="fixed-separate",
        help="Retrieval method to evaluate (default: fixed-separate).",
    )
    parser.add_argument(
        "--split", default=None,
        help="Evaluate only this split (train / validation / test).  Default: all.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve per granularity (default: 5).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max questions to evaluate (useful for quick testing).",
    )
    parser.add_argument(
        "--store-text", action="store_true",
        help="Store full retrieved_text and evidence_text in output records.",
    )
    parser.add_argument(
        "--log-every", type=int, default=50,
        help="Log progress every N questions (default: 50).",
    )
    args = parser.parse_args()

    # ── Output path ──────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = {
        "fixed-separate": "FixedSeparate",
    }[args.method]
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"RetrievalEval{method_tag}_{ts}.jsonl"

    # ── Connect to Qdrant ────────────────────────────────
    logger.info("Connecting to Qdrant …")
    client = get_qdrant_client()

    # ── Load questions ───────────────────────────────────
    logger.info(
        "Loading questions from PaperQuestion (split=%s, limit=%s) …",
        args.split or "ALL", args.limit or "ALL",
    )
    questions = load_questions(client, split=args.split, limit=args.limit)
    logger.info("Loaded %d questions.", len(questions))

    if not questions:
        logger.warning("No questions found – exiting.")
        client.close()
        return

    # ── Granularity levels ───────────────────────────────
    granularity_levels = list(range(1, len(CHUNK_SIZES) + 1))
    logger.info(
        "Granularity levels: %s  (tokens: %s)", granularity_levels, CHUNK_SIZES,
    )
    logger.info("top-k = %d", args.top_k)

    # ── Run evaluation ───────────────────────────────────
    # Accumulators for end-of-run summary
    per_gran_f1: Dict = defaultdict(list)         # level → [f1, …]
    per_gran_avg_score: Dict = defaultdict(list)   # level → [avg_score, …]
    per_gran_count: Dict = defaultdict(int)
    questions_skipped = 0
    total_records = 0

    logger.info("Writing results to %s", out_path)
    logger.info("=" * 60)

    with open(out_path, "w", encoding="utf-8") as fout:
        for q_idx, q in enumerate(tqdm(questions, desc="Evaluating")):
            records = list(evaluate_question(
                client=client,
                question_point_id=q["point_id"],
                question_vector=q["vector"],
                document_id=q["document_id"],
                question_text=q["question_text"],
                split=q["split"],
                top_k=args.top_k,
                granularity_levels=granularity_levels,
                store_retrieved_text=args.store_text,
            ))

            if not records:
                questions_skipped += 1
                continue

            for rec in records:
                fout.write(json.dumps(rec) + "\n")
                total_records += 1

                lv = rec["granularity_level"]
                per_gran_f1[lv].append(rec["f1_joined_topk"])
                per_gran_avg_score[lv].append(rec["avg_score_topk"])
                per_gran_count[lv] += 1

            # Periodic log
            if (q_idx + 1) % args.log_every == 0:
                logger.info(
                    "  … processed %d / %d questions  (%d records written)",
                    q_idx + 1, len(questions), total_records,
                )

    # ── Close connection ─────────────────────────────────
    client.close()

    # ── Summary ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info("  %-30s %s", "method", METHODS[args.method])
    logger.info("  %-30s %s", "split", args.split or "ALL")
    logger.info("  %-30s %d", "top_k", args.top_k)
    logger.info("  %-30s %d", "questions_evaluated", len(questions) - questions_skipped)
    logger.info("  %-30s %d", "questions_skipped (no evidence)", questions_skipped)
    logger.info("  %-30s %d", "total_records", total_records)
    logger.info("  %-30s %s", "output_file", out_path)
    logger.info("-" * 60)

    for lv in sorted(per_gran_f1.keys()):
        f1_vals = per_gran_f1[lv]
        sc_vals = per_gran_avg_score[lv]
        tok_size = CHUNK_SIZES[lv - 1] if lv <= len(CHUNK_SIZES) else "?"
        mean_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
        mean_sc = sum(sc_vals) / len(sc_vals) if sc_vals else 0.0
        logger.info(
            "  Level %d (%3s tok) │ mean_f1=%.4f  mean_avg_score=%.4f  n=%d",
            lv, tok_size, mean_f1, mean_sc, per_gran_count[lv],
        )

    logger.info("=" * 60)


# Typing import at module level causes issues under Python 3.9
# with forward-reference resolution; keep Dict here.
from typing import Dict  # noqa: E402

if __name__ == "__main__":
    main()
