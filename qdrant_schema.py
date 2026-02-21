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

from config import QDRANT_HOST, QDRANT_HTTP_PORT, QDRANT_GRPC_PORT, QDRANT_API_KEY

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536   # text-embedding-3-small default dimension
COLLECTIONS = ["PaperChunk", "PaperQuestion", "PaperEvidence"]


# ── Connection ───────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client."""
    api_key: Optional[str] = QDRANT_API_KEY if QDRANT_API_KEY else None

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
    if "PaperChunk" not in existing:
        client.create_collection(
            collection_name="PaperChunk",
            vectors_config=_vector_params(),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=10_000,   # use mmap after 10 k vectors
            ),
        )
        # Payload indices for common filters
        client.create_payload_index("PaperChunk", "document_id",       PayloadSchemaType.KEYWORD)
        client.create_payload_index("PaperChunk", "granularity_level", PayloadSchemaType.INTEGER)
        logger.info("Created collection: PaperChunk")
    else:
        logger.info("Collection PaperChunk already exists – skipping.")

    # ── PaperQuestion ─────────────────────────────────────
    if "PaperQuestion" not in existing:
        client.create_collection(
            collection_name="PaperQuestion",
            vectors_config=_vector_params(),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=10_000,
            ),
        )
        client.create_payload_index("PaperQuestion", "document_id", PayloadSchemaType.KEYWORD)
        client.create_payload_index("PaperQuestion", "split",       PayloadSchemaType.KEYWORD)
        logger.info("Created collection: PaperQuestion")
    else:
        logger.info("Collection PaperQuestion already exists – skipping.")

    # ── PaperEvidence ─────────────────────────────────────
    if "PaperEvidence" not in existing:
        client.create_collection(
            collection_name="PaperEvidence",
            vectors_config=_vector_params(),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=10_000,
            ),
        )
        client.create_payload_index("PaperEvidence", "question_id", PayloadSchemaType.KEYWORD)
        client.create_payload_index("PaperEvidence", "document_id", PayloadSchemaType.KEYWORD)
        logger.info("Created collection: PaperEvidence")
    else:
        logger.info("Collection PaperEvidence already exists – skipping.")


# ── Schema Deletion ──────────────────────────────────────
def delete_schema(client: QdrantClient):
    """Drop every QASPER-related collection."""
    existing = {c.name for c in client.get_collections().collections}
    for name in COLLECTIONS:
        if name in existing:
            client.delete_collection(name)
            logger.info("Deleted collection: %s", name)


# ── Standalone entry-point ───────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
    )
    client = get_qdrant_client()
    create_schema(client, recreate=True)
    logger.info("Schema (re)created successfully.")
