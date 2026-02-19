#!/usr/bin/env python
"""
Phase 1 – Prepare QASPER Dataset
=================================
Downloads ``allenai/qasper`` from HuggingFace, embeds everything via the
OpenAI API, and stores three collections in Weaviate:

1. **PaperChunk**    – multi-granularity document chunks
2. **PaperQuestion** – research questions
3. **PaperEvidence** – highlighted evidence (un-chunked)

Usage
-----
::

    python prepare_dataset.py                      # full run
    python prepare_dataset.py --recreate           # drop & recreate schema first
    python prepare_dataset.py --limit 5            # process only 5 papers (testing)
    python prepare_dataset.py --splits train       # process only the train split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from weaviate.util import generate_uuid5

from config import CHUNK_SIZES, JSON_OUTPUT, JSON_OUTPUT_DIR
from weaviate_schema import get_weaviate_client, create_schema
from embedding_utils import get_embeddings, set_embedding_max_workers
from chunking_utils import chunk_text, build_full_text

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global stats (thread-safe) ───────────────────────────
_stats_lock = threading.Lock()
stats = {
    "papers_processed": 0,
    "chunks_inserted": 0,
    "questions_inserted": 0,
    "questions_skipped_no_evidence": 0,
    "evidence_inserted": 0,
    "embedding_api_calls_tokens": 0,
}


def _inc(key: str, n: int = 1) -> None:
    """Thread-safe stats increment."""
    with _stats_lock:
        stats[key] += n


# ═════════════════════════════════════════════════════════
#  Per-paper processing helpers
# ═════════════════════════════════════════════════════════

def process_paper_chunks(paper: dict, collection, json_file=None) -> None:
    """Chunk a single paper at every granularity level, embed, and insert."""
    document_id = paper["id"]
    document_title = paper.get("title", "")
    full_text = build_full_text(paper)

    if not full_text.strip():
        logger.warning("Paper %s has empty text – skipping chunks.", document_id)
        return

    all_records: list[tuple[dict, str]] = []   # (properties, uuid)
    all_contents: list[str] = []

    for level, chunk_size in enumerate(CHUNK_SIZES, start=1):
        chunks = chunk_text(full_text, chunk_size)
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            props = {
                "document_id":        document_id,
                "document_title":     document_title,
                "chunk_idx":          idx,
                "total_chunks":       total_chunks,
                "content":            chunk["content"],
                "chunk_size":         chunk["token_count"],
                "granularity_level":  level,
                "original_text_span": f"{chunk['span_start']}:{chunk['span_end']}",
            }
            uuid = generate_uuid5(f"{document_id}_g{level}_c{idx}")
            all_records.append((props, uuid))
            all_contents.append(chunk["content"])

    if not all_contents:
        return

    # Batch-embed (batches sent concurrently inside get_embeddings)
    embeddings = get_embeddings(all_contents)

    # Batch-insert into Weaviate
    with collection.batch.dynamic() as batch:
        for (props, uuid), vec in zip(all_records, embeddings):
            batch.add_object(properties=props, vector=vec, uuid=uuid)

    _inc("chunks_inserted", len(all_records))

    # Optional JSONL output (vectors omitted to save space)
    if json_file:
        for (props, uuid), _ in zip(all_records, embeddings):
            json_file.write(json.dumps({**props, "uuid": str(uuid)}) + "\n")


def _get_answerable_question_ids(paper: dict) -> set:
    """Return the set of question_ids that have at least one non-empty evidence."""
    qas = paper.get("qas", {})
    answers_list = qas.get("answers", [])
    question_ids = qas.get("question_id", [])
    answerable = set()

    for q_id, answers_data in zip(question_ids, answers_list):
        for ann in answers_data.get("answer", []):
            for ev in ann.get("highlighted_evidence", []):
                if ev and ev.strip():
                    answerable.add(q_id)
                    break
            if q_id in answerable:
                break
    return answerable


def process_questions(
    paper: dict,
    split_name: str,
    collection,
    json_file=None,
) -> dict[str, tuple[str, str]]:
    """Embed and insert questions.  Returns ``{qasper_qid: (uuid, text)}``.

    Questions with **no valid evidence** are skipped entirely.
    """
    document_id = paper["id"]
    qas = paper.get("qas", {})
    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [])

    if not questions:
        return {}

    answerable_ids = _get_answerable_question_ids(paper)

    q_records: list[tuple[dict, str, str]] = []   # (props, uuid, qid)
    q_texts: list[str] = []

    for q_text, q_id in zip(questions, question_ids):
        if q_id not in answerable_ids:
            _inc("questions_skipped_no_evidence")
            continue
        uuid = generate_uuid5(f"{document_id}_{q_id}")
        props = {
            "document_id":   document_id,
            "question_text": q_text,
            "split":         split_name,
        }
        q_records.append((props, uuid, q_id))
        q_texts.append(q_text)

    if not q_texts:
        return {}

    embeddings = get_embeddings(q_texts)

    with collection.batch.dynamic() as batch:
        for (props, uuid, _), vec in zip(q_records, embeddings):
            batch.add_object(properties=props, vector=vec, uuid=uuid)

    _inc("questions_inserted", len(q_records))

    # Build map for evidence linking
    question_map: dict[str, tuple[str, str]] = {}
    for (props, uuid, q_id), _ in zip(q_records, embeddings):
        question_map[q_id] = (str(uuid), props["question_text"])

    if json_file:
        for (props, uuid, q_id), _ in zip(q_records, embeddings):
            json_file.write(
                json.dumps({**props, "uuid": str(uuid), "original_question_id": q_id})
                + "\n"
            )

    return question_map


def process_evidence(
    paper: dict,
    question_map: dict[str, tuple[str, str]],
    collection,
    json_file=None,
) -> None:
    """Embed highlighted evidence (un-chunked) and insert into PaperEvidence."""
    document_id = paper["id"]
    qas = paper.get("qas", {})
    answers_list = qas.get("answers", [])
    question_ids = qas.get("question_id", [])

    ev_records: list[tuple[dict, str]] = []
    ev_texts: list[str] = []

    for q_id, answers_data in zip(question_ids, answers_list):
        if q_id not in question_map:
            continue

        q_uuid_str, q_text = question_map[q_id]

        # Collect unique evidence across all annotators
        unique_evidence: set[str] = set()
        annotator_answers = answers_data.get("answer", [])
        for ann in annotator_answers:
            for ev in ann.get("highlighted_evidence", []):
                if ev and ev.strip():
                    unique_evidence.add(ev.strip())

        evidence_list = sorted(unique_evidence)
        total_counts = len(evidence_list)

        for ev_text in evidence_list:
            ev_hash = hashlib.md5(ev_text.encode()).hexdigest()[:12]
            uuid = generate_uuid5(f"{q_uuid_str}_{ev_hash}")
            props = {
                "question_id":   q_uuid_str,
                "document_id":   document_id,
                "question_text": q_text,
                "evidence_text": ev_text,
                "total_counts":  total_counts,
            }
            ev_records.append((props, uuid))
            ev_texts.append(ev_text)

    if not ev_texts:
        return

    embeddings = get_embeddings(ev_texts)

    with collection.batch.dynamic() as batch:
        for (props, uuid), vec in zip(ev_records, embeddings):
            batch.add_object(properties=props, vector=vec, uuid=uuid)

    _inc("evidence_inserted", len(ev_records))

    if json_file:
        for (props, uuid), _ in zip(ev_records, embeddings):
            json_file.write(json.dumps({**props, "uuid": str(uuid)}) + "\n")


# ═════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 – Prepare QASPER dataset for Weaviate",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate Weaviate collections before ingestion.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max papers to process per split (useful for quick testing).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Which splits to ingest (e.g. --splits train validation).",
    )
    parser.add_argument(
        "--embedding-max-workers", type=int, default=8,
        help="Concurrent OpenAI embedding batch requests (default: 8).",
    )
    parser.add_argument(
        "--paper-max-workers", type=int, default=4,
        help="Papers processed in parallel (default: 4).",
    )
    args = parser.parse_args()

    paper_workers = args.paper_max_workers
    set_embedding_max_workers(args.embedding_max_workers)

    # 1. Load dataset ──────────────────────────────────────
    logger.info("Downloading / loading QASPER dataset …")
    dataset = load_dataset("allenai/qasper")
    available_splits = list(dataset.keys())
    logger.info("Available splits: %s", available_splits)

    splits = args.splits or available_splits

    # 2. Weaviate connection & schema ──────────────────────
    logger.info("Connecting to Weaviate …")
    client = get_weaviate_client()

    try:
        create_schema(client, recreate=args.recreate)

        chunk_col    = client.collections.get("PaperChunk")
        question_col = client.collections.get("PaperQuestion")
        evidence_col = client.collections.get("PaperEvidence")

        # 3. Optional JSONL writers ────────────────────────
        jf_chunks = jf_questions = jf_evidence = None
        if JSON_OUTPUT:
            out = Path(JSON_OUTPUT_DIR)
            out.mkdir(parents=True, exist_ok=True)
            jf_chunks    = open(out / "chunks.jsonl",    "w", encoding="utf-8")
            jf_questions = open(out / "questions.jsonl",  "w", encoding="utf-8")
            jf_evidence  = open(out / "evidence.jsonl",   "w", encoding="utf-8")
            logger.info("JSONL output → %s/", out)

        # 4. Iterate over splits & papers (concurrent) ────
        def _process_one_paper(paper, split_name):
            """Process a single paper (embedding + insert). Called from threads."""
            process_paper_chunks(paper, chunk_col, jf_chunks)
            question_map = process_questions(
                paper, split_name, question_col, jf_questions,
            )
            process_evidence(paper, question_map, evidence_col, jf_evidence)
            _inc("papers_processed")

        for split_name in splits:
            if split_name not in dataset:
                logger.warning("Split '%s' not in dataset – skipping.", split_name)
                continue

            split_data = dataset[split_name]
            n = min(args.limit, len(split_data)) if args.limit else len(split_data)

            logger.info("═" * 60)
            logger.info(
                "Split: %s  (%d papers, %d workers)",
                split_name, n, paper_workers,
            )
            logger.info("═" * 60)

            papers = [split_data[i] for i in range(n)]

            with ThreadPoolExecutor(max_workers=paper_workers) as pool:
                futures = {
                    pool.submit(_process_one_paper, paper, split_name): paper
                    for paper in papers
                }
                with tqdm(total=n, desc=f"[{split_name}]") as pbar:
                    for future in as_completed(futures):
                        paper = futures[future]
                        try:
                            future.result()
                        except Exception:
                            logger.exception(
                                "Error processing paper %s",
                                paper.get("id", "?"),
                            )
                        pbar.update(1)

        # 5. Cleanup & summary ─────────────────────────────
        if JSON_OUTPUT:
            for f in (jf_chunks, jf_questions, jf_evidence):
                if f:
                    f.close()

        logger.info("═" * 60)
        logger.info("  INGESTION COMPLETE")
        logger.info("═" * 60)
        for k, v in stats.items():
            logger.info("  %-30s %s", k, f"{v:,}")
        logger.info("  %-30s %s", "chunk_sizes", CHUNK_SIZES)
        logger.info("  %-30s %d", "granularity_levels", len(CHUNK_SIZES))

    finally:
        client.close()
        logger.info("Weaviate connection closed.")


if __name__ == "__main__":
    main()
