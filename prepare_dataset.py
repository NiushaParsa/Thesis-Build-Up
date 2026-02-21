#!/usr/bin/env python
"""
Phase 1 – Prepare QASPER Dataset
=================================
Downloads ``allenai/qasper`` from HuggingFace, embeds everything via the
OpenAI API, and stores three collections in **Qdrant**:

1. **PaperChunk**    – multi-granularity document chunks
2. **PaperQuestion** – research questions
3. **PaperEvidence** – highlighted evidence (un-chunked)

Usage
-----
::

    python prepare_dataset.py                      # full run (resumes from checkpoint)
    python prepare_dataset.py --recreate           # drop schema + checkpoint, start fresh
    python prepare_dataset.py --limit 5            # process only 5 papers (testing)
    python prepare_dataset.py --splits train       # process only the train split

Checkpoint / Resume
-------------------
A ``checkpoint.json`` file is saved after every successfully processed
paper.  If the script crashes, simply re-run the **same command**
(without ``--recreate``) and it will automatically skip papers that were
already ingested.  Use ``--recreate`` to start fresh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from qdrant_client.models import PointStruct
from tqdm import tqdm

from config import CHUNK_SIZES, JSON_OUTPUT, JSON_OUTPUT_DIR
from qdrant_schema import get_qdrant_client, create_schema
from embedding_utils import get_embeddings, set_embedding_max_workers
from chunking_utils import chunk_text, build_full_text

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Checkpoint file ──────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoint.json")

# ── Global stats (thread-safe) ───────────────────────────
_stats_lock = threading.Lock()
stats = {
    "papers_processed": 0,
    "papers_skipped_checkpoint": 0,
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


# ── Deterministic UUID helper ────────────────────────────
def _make_uuid(seed: str) -> str:
    """Generate a deterministic UUID-5 from *seed* (idempotent upserts)."""
    return str(_uuid.uuid5(_uuid.NAMESPACE_DNS, seed))


# ═════════════════════════════════════════════════════════
#  Checkpoint helpers
#  Format:  { "doc_id": {"split": "train", "done": [10, 20, …]}, … }
#  Each value is an object with the split name and a list of completed
#  chunk_sizes (ints) plus the special markers "questions" / "evidence".
# ═════════════════════════════════════════════════════════
_checkpoint_lock = threading.Lock()
_checkpoint: dict = {}


def load_checkpoint() -> dict:
    """Load checkpoint from disk, or return empty dict."""
    global _checkpoint
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Migrate old format (plain list) → new format (dict with split)
            migrated = False
            for doc_id, val in raw.items():
                if isinstance(val, list):
                    raw[doc_id] = {"split": "unknown", "done": val}
                    migrated = True
            if migrated:
                logger.info("Migrated checkpoint to new format (added split info).")
            _checkpoint = raw
            total_docs = len(_checkpoint)
            total_items = sum(len(v["done"]) for v in _checkpoint.values())
            logger.info(
                "Loaded checkpoint – %d documents, %d completed items.",
                total_docs, total_items,
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt checkpoint file – starting fresh.")
            _checkpoint = {}
    else:
        _checkpoint = {}
    return _checkpoint


def save_checkpoint(document_id: str, item, split: str = "unknown") -> None:
    """Thread-safe: mark *item* (chunk_size int or stage str) as done for *document_id*."""
    with _checkpoint_lock:
        entry = _checkpoint.setdefault(document_id, {"split": split, "done": []})
        # Ensure split is set (may still be 'unknown' from migration)
        if entry["split"] == "unknown" and split != "unknown":
            entry["split"] = split
        done_list = entry["done"]
        if item not in done_list:
            done_list.append(item)

        tmp = CHECKPOINT_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_checkpoint, f, indent=2)
        tmp.replace(CHECKPOINT_PATH)


def clear_checkpoint() -> None:
    """Delete the checkpoint file (called on ``--recreate``)."""
    global _checkpoint
    _checkpoint = {}
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Cleared checkpoint file.")


def is_done(document_id: str, item) -> bool:
    """Check if *item* (chunk_size int or stage str) is already done for *document_id*."""
    entry = _checkpoint.get(document_id, {})
    return item in entry.get("done", [])


# ═════════════════════════════════════════════════════════
#  Per-paper processing helpers
# ═════════════════════════════════════════════════════════

def process_paper_chunks(paper: dict, client, chunk_size: int, level: int, json_file=None) -> None:
    """Chunk a single paper at ONE granularity level, embed, and upsert."""
    document_id = paper["id"]
    document_title = paper.get("title", "")
    full_text = build_full_text(paper)

    if not full_text.strip():
        logger.warning("Paper %s has empty text – skipping chunks.", document_id)
        return

    chunks = chunk_text(full_text, chunk_size)
    total_chunks = len(chunks)

    all_payloads: list[dict] = []
    all_ids: list[str] = []
    all_contents: list[str] = []

    for idx, chunk in enumerate(chunks):
        payload = {
            "document_id":        document_id,
            "document_title":     document_title,
            "chunk_idx":          idx,
            "total_chunks":       total_chunks,
            "content":            chunk["content"],
            "chunk_size":         chunk["token_count"],
            "granularity_level":  level,
            "original_text_span": f"{chunk['span_start']}:{chunk['span_end']}",
        }
        point_id = _make_uuid(f"{document_id}_g{level}_c{idx}")
        all_payloads.append(payload)
        all_ids.append(point_id)
        all_contents.append(chunk["content"])

    if not all_contents:
        return

    # Batch-embed (batches sent concurrently inside get_embeddings)
    embeddings = get_embeddings(all_contents)

    # Upsert in sub-batches of 100
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        points = [
            PointStruct(id=pid, vector=vec, payload=pay)
            for pid, vec, pay in zip(
                all_ids[i : i + batch_size],
                embeddings[i : i + batch_size],
                all_payloads[i : i + batch_size],
            )
        ]
        client.upsert(collection_name="PaperChunk", points=points)

    _inc("chunks_inserted", len(all_ids))

    # Optional JSONL output (vectors omitted to save space)
    if json_file:
        for pay, pid in zip(all_payloads, all_ids):
            json_file.write(json.dumps({**pay, "uuid": pid}) + "\n")


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


def _rebuild_question_map(paper: dict) -> dict[str, tuple[str, str]]:
    """Rebuild the question_map for a paper whose questions are already in Qdrant.

    Needed when questions are checkpointed but evidence is not yet done.
    """
    document_id = paper["id"]
    qas = paper.get("qas", {})
    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [])
    answerable_ids = _get_answerable_question_ids(paper)

    question_map: dict[str, tuple[str, str]] = {}
    for q_text, q_id in zip(questions, question_ids):
        if q_id not in answerable_ids:
            continue
        point_id = _make_uuid(f"{document_id}_{q_id}")
        question_map[q_id] = (point_id, q_text)
    return question_map


def process_questions(
    paper: dict,
    split_name: str,
    client,
    json_file=None,
) -> dict[str, tuple[str, str]]:
    """Embed and upsert questions.  Returns ``{qasper_qid: (uuid, text)}``.

    Questions with **no valid evidence** are skipped entirely.
    """
    document_id = paper["id"]
    qas = paper.get("qas", {})
    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [])

    if not questions:
        return {}

    answerable_ids = _get_answerable_question_ids(paper)

    q_payloads: list[dict] = []
    q_ids: list[str] = []
    q_texts: list[str] = []
    q_original_ids: list[str] = []

    for q_text, q_id in zip(questions, question_ids):
        if q_id not in answerable_ids:
            _inc("questions_skipped_no_evidence")
            continue
        point_id = _make_uuid(f"{document_id}_{q_id}")
        payload = {
            "document_id":   document_id,
            "question_text": q_text,
            "split":         split_name,
        }
        q_payloads.append(payload)
        q_ids.append(point_id)
        q_texts.append(q_text)
        q_original_ids.append(q_id)

    if not q_texts:
        return {}

    embeddings = get_embeddings(q_texts)

    points = [
        PointStruct(id=pid, vector=vec, payload=pay)
        for pid, vec, pay in zip(q_ids, embeddings, q_payloads)
    ]
    client.upsert(collection_name="PaperQuestion", points=points)

    _inc("questions_inserted", len(q_ids))

    # Build map for evidence linking
    question_map: dict[str, tuple[str, str]] = {}
    for pid, pay, q_id in zip(q_ids, q_payloads, q_original_ids):
        question_map[q_id] = (pid, pay["question_text"])

    if json_file:
        for pay, pid, q_id in zip(q_payloads, q_ids, q_original_ids):
            json_file.write(
                json.dumps({**pay, "uuid": pid, "original_question_id": q_id})
                + "\n"
            )

    return question_map


def process_evidence(
    paper: dict,
    question_map: dict[str, tuple[str, str]],
    client,
    json_file=None,
) -> None:
    """Embed highlighted evidence (un-chunked) and upsert into PaperEvidence."""
    document_id = paper["id"]
    qas = paper.get("qas", {})
    answers_list = qas.get("answers", [])
    question_ids = qas.get("question_id", [])

    ev_payloads: list[dict] = []
    ev_ids: list[str] = []
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
            point_id = _make_uuid(f"{q_uuid_str}_{ev_hash}")
            payload = {
                "question_id":   q_uuid_str,
                "document_id":   document_id,
                "question_text": q_text,
                "evidence_text": ev_text,
                "total_counts":  total_counts,
            }
            ev_payloads.append(payload)
            ev_ids.append(point_id)
            ev_texts.append(ev_text)

    if not ev_texts:
        return

    embeddings = get_embeddings(ev_texts)

    # Upsert in sub-batches of 100
    batch_size = 100
    for i in range(0, len(ev_ids), batch_size):
        points = [
            PointStruct(id=pid, vector=vec, payload=pay)
            for pid, vec, pay in zip(
                ev_ids[i : i + batch_size],
                embeddings[i : i + batch_size],
                ev_payloads[i : i + batch_size],
            )
        ]
        client.upsert(collection_name="PaperEvidence", points=points)

    _inc("evidence_inserted", len(ev_ids))

    if json_file:
        for pay, pid in zip(ev_payloads, ev_ids):
            json_file.write(json.dumps({**pay, "uuid": pid}) + "\n")


# ═════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 – Prepare QASPER dataset for Qdrant",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate Qdrant collections AND clear checkpoint.",
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

    # ── Checkpoint handling ───────────────────────────────
    if args.recreate:
        clear_checkpoint()
    checkpoint = load_checkpoint()

    resuming = bool(checkpoint)
    if resuming:
        logger.info(
            "RESUMING – %d documents with partial/full progress.",
            len(checkpoint),
        )

    # 1. Load dataset ──────────────────────────────────────
    logger.info("Downloading / loading QASPER dataset …")
    dataset = load_dataset("allenai/qasper")
    available_splits = list(dataset.keys())
    logger.info("Available splits: %s", available_splits)

    splits = args.splits or available_splits

    # 2. Qdrant connection & schema ────────────────────────
    logger.info("Connecting to Qdrant …")
    client = get_qdrant_client()

    create_schema(client, recreate=args.recreate)

    # 3. Optional JSONL writers ────────────────────────────
    file_mode = "a" if resuming else "w"
    jf_chunks = jf_questions = jf_evidence = None
    if JSON_OUTPUT:
        out = Path(JSON_OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        jf_chunks    = open(out / "chunks.jsonl",    file_mode, encoding="utf-8")
        jf_questions = open(out / "questions.jsonl",  file_mode, encoding="utf-8")
        jf_evidence  = open(out / "evidence.jsonl",   file_mode, encoding="utf-8")
        logger.info(
            "JSONL output → %s/ (mode: %s)",
            out, "append" if resuming else "write",
        )

    # 4. Iterate over splits & papers (concurrent) ────────
    def _process_one_paper(paper, split_name):
        """Process a single paper (embedding + upsert). Called from threads."""
        paper_id = paper["id"]

        # ── Chunks: one chunk_size at a time ──────────────
        for level, chunk_size in enumerate(CHUNK_SIZES, start=1):
            if is_done(paper_id, chunk_size):
                continue
            process_paper_chunks(paper, client, chunk_size, level, jf_chunks)
            save_checkpoint(paper_id, chunk_size, split=split_name)
            _inc("chunks_inserted")   # per-granularity count

        # ── Questions ─────────────────────────────────────
        if not is_done(paper_id, "questions"):
            question_map = process_questions(paper, split_name, client, jf_questions)
            save_checkpoint(paper_id, "questions", split=split_name)
        else:
            # Still need the map to process evidence
            question_map = _rebuild_question_map(paper)

        # ── Evidence ──────────────────────────────────────
        if not is_done(paper_id, "evidence"):
            process_evidence(paper, question_map, client, jf_evidence)
            save_checkpoint(paper_id, "evidence", split=split_name)

        _inc("papers_processed")

    try:
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
                                "Error processing paper %s – will be retried on next run.",
                                paper.get("id", "?"),
                            )
                        pbar.update(1)

        # 5. Cleanup & summary ─────────────────────────────
        logger.info("═" * 60)
        logger.info("  INGESTION COMPLETE")
        logger.info("═" * 60)
        for k, v in stats.items():
            logger.info("  %-30s %s", k, f"{v:,}")
        logger.info("  %-30s %s", "chunk_sizes", CHUNK_SIZES)
        logger.info("  %-30s %d", "granularity_levels", len(CHUNK_SIZES))

        # Expected items per fully-done paper: len(CHUNK_SIZES) + "questions" + "evidence"
        expected_per_paper = len(CHUNK_SIZES) + 2
        cp = load_checkpoint()
        fully_done = sum(1 for v in cp.values() if len(v.get("done", [])) >= expected_per_paper)
        logger.info("  %-30s %d", "fully_checkpointed_papers", fully_done)
        logger.info("  %-30s %d", "total_docs_in_checkpoint", len(cp))
        for s in ("train", "validation", "test"):
            cnt = sum(1 for v in cp.values() if v.get("split") == s)
            if cnt:
                logger.info("  %-30s %d", f"  └─ {s}", cnt)

    finally:
        if JSON_OUTPUT:
            for f in (jf_chunks, jf_questions, jf_evidence):
                if f:
                    f.close()
        client.close()
        logger.info("Qdrant connection closed.")


if __name__ == "__main__":
    main()
