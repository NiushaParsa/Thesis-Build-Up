#!/usr/bin/env python
"""
Retrieval Example
=================
Demonstrates how to query the three QASPER collections stored in Qdrant:

* Semantic search on **PaperChunk** (with optional granularity filter)
* Semantic search on **PaperQuestion** (with optional split filter)
* Evidence look-up for a given question UUID

Run after ``prepare_dataset.py`` has finished ingestion.

Usage::

    python retrieval_example.py
"""

from __future__ import annotations

import logging

from qdrant_client.models import Filter, FieldCondition, MatchValue

from qdrant_schema import get_qdrant_client
from embedding_utils import get_single_embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
#  Query helpers
# ═════════════════════════════════════════════════════════

def search_chunks(
    client,
    query: str,
    limit: int = 5,
    granularity_level: int = None,
):
    """Semantic search over PaperChunk, optionally filtered by granularity."""
    vec = get_single_embedding(query)

    query_filter = None
    if granularity_level is not None:
        query_filter = Filter(
            must=[FieldCondition(key="granularity_level", match=MatchValue(value=granularity_level))]
        )

    results = client.search(
        collection_name="PaperChunk",
        query_vector=vec,
        limit=limit,
        query_filter=query_filter,
    )

    print(f"\n{'─'*60}")
    print(f"Chunk search: \"{query}\"  (granularity={granularity_level})")
    print(f"{'─'*60}")
    for hit in results:
        p = hit.payload
        print(
            f"  [{p['document_title'][:50]}]  "
            f"G{p['granularity_level']}  "
            f"chunk {p['chunk_idx']}/{p['total_chunks']}  "
            f"({p['chunk_size']} tok)  "
            f"score={hit.score:.4f}"
        )
        print(f"    {p['content'][:150]}…")
    return results


def search_questions(
    client,
    query: str,
    limit: int = 5,
    split: str = None,
):
    """Semantic search over PaperQuestion."""
    vec = get_single_embedding(query)

    query_filter = None
    if split:
        query_filter = Filter(
            must=[FieldCondition(key="split", match=MatchValue(value=split))]
        )

    results = client.search(
        collection_name="PaperQuestion",
        query_vector=vec,
        limit=limit,
        query_filter=query_filter,
    )

    print(f"\n{'─'*60}")
    print(f"Question search: \"{query}\"  (split={split})")
    print(f"{'─'*60}")
    for hit in results:
        p = hit.payload
        print(
            f"  [{p['document_id'][:20]}]  "
            f"score={hit.score:.4f}"
        )
        print(f"    Q: {p['question_text'][:150]}")
    return results


def get_evidence_for_question(client, question_id: str):
    """Fetch all evidence objects linked to a PaperQuestion UUID."""
    query_filter = Filter(
        must=[FieldCondition(key="question_id", match=MatchValue(value=question_id))]
    )

    results, _ = client.scroll(
        collection_name="PaperEvidence",
        scroll_filter=query_filter,
        limit=50,
    )

    print(f"\n{'─'*60}")
    print(f"Evidence for question  {question_id}")
    print(f"{'─'*60}")
    for point in results:
        p = point.payload
        print(f"  Q: {p['question_text'][:100]}")
        print(f"  E: {p['evidence_text'][:200]}")
        print(f"  (total evidence pieces for this question: {p['total_counts']})")
        print()
    return results


def collection_stats(client):
    """Print object counts for every QASPER collection."""
    print("\n╔══════════════════════════════════════╗")
    print("║        Collection Statistics         ║")
    print("╠══════════════════════════════════════╣")
    for name in ("PaperChunk", "PaperQuestion", "PaperEvidence"):
        info = client.get_collection(name)
        cnt = info.points_count
        print(f"║  {name:<22s}  {cnt:>8,}  ║")
    print("╚══════════════════════════════════════╝")


# ═════════════════════════════════════════════════════════
#  Demo
# ═════════════════════════════════════════════════════════

def main():
    client = get_qdrant_client()
    try:
        collection_stats(client)

        # ── Example 1 – chunk search at granularity 3 (40 tokens) ────
        search_chunks(
            client,
            "machine learning model performance evaluation",
            limit=3,
            granularity_level=3,
        )

        # ── Example 2 – question search (train split) ────────────────
        search_questions(
            client,
            "How does the model handle out-of-domain data?",
            limit=3,
            split="train",
        )

        # ── Example 3 – find a question then look up its evidence ────
        results = search_questions(
            client,
            "What evaluation metrics were used?",
            limit=1,
        )
        if results:
            q_uuid = str(results[0].id)
            get_evidence_for_question(client, q_uuid)

    finally:
        client.close()


if __name__ == "__main__":
    main()
