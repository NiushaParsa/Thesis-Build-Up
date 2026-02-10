#!/usr/bin/env python
"""
Retrieval Example
=================
Demonstrates how to query the three QASPER collections stored in Weaviate:

* Semantic search on **PaperChunk** (with optional granularity filter)
* Semantic search on **PaperQuestion** (with optional split filter)
* Evidence look-up for a given question UUID

Run after ``prepare_dataset.py`` has finished ingestion.

Usage::

    python retrieval_example.py
"""

from __future__ import annotations

import logging

from weaviate.classes.query import MetadataQuery, Filter

from weaviate_schema import get_weaviate_client
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
    collection = client.collections.get("PaperChunk")
    vec = get_single_embedding(query)

    filters = None
    if granularity_level is not None:
        filters = Filter.by_property("granularity_level").equal(granularity_level)

    response = collection.query.near_vector(
        near_vector=vec,
        limit=limit,
        filters=filters,
        return_metadata=MetadataQuery(distance=True),
    )

    print(f"\n{'─'*60}")
    print(f"Chunk search: \"{query}\"  (granularity={granularity_level})")
    print(f"{'─'*60}")
    for obj in response.objects:
        p = obj.properties
        print(
            f"  [{p['document_title'][:50]}]  "
            f"G{p['granularity_level']}  "
            f"chunk {p['chunk_idx']}/{p['total_chunks']}  "
            f"({p['chunk_size']} tok)  "
            f"dist={obj.metadata.distance:.4f}"
        )
        print(f"    {p['content'][:150]}…")
    return response


def search_questions(
    client,
    query: str,
    limit: int = 5,
    split: str = None,
):
    """Semantic search over PaperQuestion."""
    collection = client.collections.get("PaperQuestion")
    vec = get_single_embedding(query)

    filters = None
    if split:
        filters = Filter.by_property("split").equal(split)

    response = collection.query.near_vector(
        near_vector=vec,
        limit=limit,
        filters=filters,
        return_metadata=MetadataQuery(distance=True),
    )

    print(f"\n{'─'*60}")
    print(f"Question search: \"{query}\"  (split={split})")
    print(f"{'─'*60}")
    for obj in response.objects:
        p = obj.properties
        print(
            f"  [{p['document_id'][:20]}]  "
            f"dist={obj.metadata.distance:.4f}"
        )
        print(f"    Q: {p['question_text'][:150]}")
    return response


def get_evidence_for_question(client, question_id: str):
    """Fetch all evidence objects linked to a PaperQuestion UUID."""
    collection = client.collections.get("PaperEvidence")

    response = collection.query.fetch_objects(
        filters=Filter.by_property("question_id").equal(question_id),
        limit=50,
    )

    print(f"\n{'─'*60}")
    print(f"Evidence for question  {question_id}")
    print(f"{'─'*60}")
    for obj in response.objects:
        p = obj.properties
        print(f"  Q: {p['question_text'][:100]}")
        print(f"  E: {p['evidence_text'][:200]}")
        print(f"  (total evidence pieces for this question: {p['total_counts']})")
        print()
    return response


def collection_stats(client):
    """Print object counts for every QASPER collection."""
    print("\n╔══════════════════════════════════════╗")
    print("║        Collection Statistics         ║")
    print("╠══════════════════════════════════════╣")
    for name in ("PaperChunk", "PaperQuestion", "PaperEvidence"):
        col = client.collections.get(name)
        cnt = col.aggregate.over_all(total_count=True).total_count
        print(f"║  {name:<22s}  {cnt:>8,}  ║")
    print("╚══════════════════════════════════════╝")


# ═════════════════════════════════════════════════════════
#  Demo
# ═════════════════════════════════════════════════════════

def main():
    client = get_weaviate_client()
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
        if results.objects:
            q_uuid = str(results.objects[0].uuid)
            get_evidence_for_question(client, q_uuid)

    finally:
        client.close()


if __name__ == "__main__":
    main()
