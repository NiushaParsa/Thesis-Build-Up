"""
Qdrant Schema Definition & Management
======================================
Defines and creates three collections for the QASPER dataset:

- **PaperChunk**     – document chunks at multiple granularity levels
- **PaperQuestion**  – research questions with embeddings
- **PaperEvidence**  – highlighted evidence for each question

All collections use on-disk vector storage + on-disk HNSW to stay
within low-RAM environments.  Vectors are computed externally via
the OpenAI API (``text-embedding-3-small``, 1 536 dimensions).
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    VectorParams,
)

from config import (
    EMBEDDING_DIM,
    PAPER_CHUNK_COLLECTION,
    PAPER_EVIDENCE_COLLECTION,
    PAPER_QUESTION_COLLECTION,
    QDRANT_API_KEY,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    QDRANT_URL,
    RETRIEVAL_EVALUATION_COLLECTION,
    ROUTER_DATASET_COLLECTION,
)

logger = logging.getLogger(__name__)

COLLECTIONS = [PAPER_CHUNK_COLLECTION, PAPER_QUESTION_COLLECTION, PAPER_EVIDENCE_COLLECTION]


# ── Connection ───────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client."""
    api_key: Optional[str] = QDRANT_API_KEY if QDRANT_API_KEY else None

    if QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL, api_key=api_key, prefer_grpc=False)
        logger.info("Connected to Qdrant at %s", QDRANT_URL)
    else:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_HTTP_PORT,
            grpc_port=QDRANT_GRPC_PORT,
            prefer_grpc=True,
            api_key=api_key,
        )
        logger.info("Connected to Qdrant at %s:%s", QDRANT_HOST, QDRANT_HTTP_PORT)
    return client


# ── Shared vector params ─────────────────────────────────
def _vector_params() -> VectorParams:
    """Return vector configuration shared by all collections."""
    return VectorParams(
        size=EMBEDDING_DIM,
        distance=Distance.COSINE,
        on_disk=True,              # ← vectors stored on disk, loaded via mmap
        hnsw_config=HnswConfigDiff(
            on_disk=True,          # ← HNSW graph also on disk
        ),
    )


# ── Schema Creation ──────────────────────────────────────
def create_schema(client: QdrantClient, recreate: bool = False):
    """Create all three QASPER collections (skip existing unless *recreate*).

    Payload indices are created on filterable fields so that
    ``near_vector`` queries with filters stay efficient.
    """

    if recreate:
        delete_schema(client)

    existing = {c.name for c in client.get_collections().collections}

    # ── PaperChunk ────────────────────────────────────────
    if PAPER_CHUNK_COLLECTION not in existing:
        client.create_collection(
            collection_name=PAPER_CHUNK_COLLECTION,
            vectors_config=_vector_params(),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=10_000,   # use mmap after 10 k vectors
            ),
        )
        # Payload indices for common filters
        client.create_payload_index(PAPER_CHUNK_COLLECTION, "document_id", PayloadSchemaType.KEYWORD)
        client.create_payload_index(PAPER_CHUNK_COLLECTION, "granularity_level", PayloadSchemaType.INTEGER)
        logger.info("Created collection: %s", PAPER_CHUNK_COLLECTION)
    else:
        logger.info("Collection %s already exists – skipping.", PAPER_CHUNK_COLLECTION)

    # ── PaperQuestion ─────────────────────────────────────
    if PAPER_QUESTION_COLLECTION not in existing:
        client.create_collection(
            collection_name=PAPER_QUESTION_COLLECTION,
            vectors_config=_vector_params(),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=10_000,
            ),
        )
        client.create_payload_index(PAPER_QUESTION_COLLECTION, "document_id", PayloadSchemaType.KEYWORD)
        client.create_payload_index(PAPER_QUESTION_COLLECTION, "split", PayloadSchemaType.KEYWORD)
        logger.info("Created collection: %s", PAPER_QUESTION_COLLECTION)
    else:
        logger.info("Collection %s already exists – skipping.", PAPER_QUESTION_COLLECTION)

    # ── PaperEvidence ─────────────────────────────────────
    if PAPER_EVIDENCE_COLLECTION not in existing:
        client.create_collection(
            collection_name=PAPER_EVIDENCE_COLLECTION,
            vectors_config=_vector_params(),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=10_000,
            ),
        )
        client.create_payload_index(PAPER_EVIDENCE_COLLECTION, "question_id", PayloadSchemaType.KEYWORD)
        client.create_payload_index(PAPER_EVIDENCE_COLLECTION, "document_id", PayloadSchemaType.KEYWORD)
        logger.info("Created collection: %s", PAPER_EVIDENCE_COLLECTION)
    else:
        logger.info("Collection %s already exists – skipping.", PAPER_EVIDENCE_COLLECTION)


# ── Schema Deletion ──────────────────────────────────────
def delete_schema(client: QdrantClient):
    """Drop every QASPER-related collection."""
    existing = {c.name for c in client.get_collections().collections}
    for name in COLLECTIONS:
        if name in existing:
            client.delete_collection(name)
            logger.info("Deleted collection: %s", name)


def ensure_evaluation_collections(
    client: QdrantClient,
    evaluation_collection: str = RETRIEVAL_EVALUATION_COLLECTION,
    router_collection: str = ROUTER_DATASET_COLLECTION,
    create_evaluation: bool = True,
    create_router: bool = True,
) -> None:
    """Create optional evaluation/router collections without deleting data."""
    existing = {c.name for c in client.get_collections().collections}
    if create_evaluation and evaluation_collection not in existing:
        client.create_collection(collection_name=evaluation_collection, vectors_config={})
        client.create_payload_index(evaluation_collection, "question_id", PayloadSchemaType.KEYWORD)
        client.create_payload_index(evaluation_collection, "split", PayloadSchemaType.KEYWORD)
        client.create_payload_index(evaluation_collection, "evaluation_config_hash", PayloadSchemaType.KEYWORD)
        client.create_payload_index(evaluation_collection, "granularity_level", PayloadSchemaType.INTEGER)
        logger.info("Created payload-only collection: %s", evaluation_collection)
    if create_router and router_collection not in existing:
        client.create_collection(collection_name=router_collection, vectors_config=_vector_params())
        client.create_payload_index(router_collection, "question_id", PayloadSchemaType.KEYWORD)
        client.create_payload_index(router_collection, "split", PayloadSchemaType.KEYWORD)
        client.create_payload_index(router_collection, "evaluation_config_hash", PayloadSchemaType.KEYWORD)
        client.create_payload_index(router_collection, "router_target_granularity", PayloadSchemaType.INTEGER)
        logger.info("Created router collection: %s", router_collection)


# ── Standalone entry-point ───────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the QASPER Qdrant schema")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Explicitly delete and recreate all QASPER collections.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
    )
    client = get_qdrant_client()
    try:
        create_schema(client, recreate=args.recreate)
        logger.info("Schema checked successfully (recreate=%s).", args.recreate)
    finally:
        client.close()
