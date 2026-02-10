"""
Weaviate Schema Definition & Management
========================================
Defines and creates three collections for the QASPER dataset:

- **PaperChunk**     – document chunks at multiple granularity levels
- **PaperQuestion**  – research questions with embeddings
- **PaperEvidence**  – highlighted evidence for each question

All collections use ``vectorizer: none`` because embeddings are
computed externally via the OpenAI API.
"""

import logging
import weaviate
from weaviate.classes.config import Configure, Property, DataType

from config import WEAVIATE_URL, WEAVIATE_PORT, WEAVIATE_GRPC_PORT, WEAVIATE_API_KEY

logger = logging.getLogger(__name__)

COLLECTIONS = ["PaperChunk", "PaperQuestion", "PaperEvidence"]


# ── Connection ───────────────────────────────────────────
def get_weaviate_client() -> weaviate.WeaviateClient:
    """Create and return a Weaviate v4 client."""
    auth = (
        weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        if WEAVIATE_API_KEY
        else None
    )

    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_URL,
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_URL,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
        auth_credentials=auth,
    )
    logger.info("Connected to Weaviate at %s:%s", WEAVIATE_URL, WEAVIATE_PORT)
    return client


# ── Schema Creation ──────────────────────────────────────
def create_schema(client: weaviate.WeaviateClient, recreate: bool = False):
    """Create all three QASPER collections (skip existing unless *recreate*)."""

    if recreate:
        delete_schema(client)

    # ── PaperChunk ────────────────────────────────────────
    if not client.collections.exists("PaperChunk"):
        client.collections.create(
            name="PaperChunk",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="document_id",        data_type=DataType.TEXT),
                Property(name="document_title",     data_type=DataType.TEXT),
                Property(name="chunk_idx",          data_type=DataType.INT),
                Property(name="total_chunks",       data_type=DataType.INT),
                Property(name="content",            data_type=DataType.TEXT),
                Property(name="chunk_size",         data_type=DataType.INT),
                Property(name="granularity_level",  data_type=DataType.INT),
                Property(name="original_text_span", data_type=DataType.TEXT),
            ],
        )
        logger.info("Created collection: PaperChunk")
    else:
        logger.info("Collection PaperChunk already exists – skipping.")

    # ── PaperQuestion ─────────────────────────────────────
    if not client.collections.exists("PaperQuestion"):
        client.collections.create(
            name="PaperQuestion",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="document_id",    data_type=DataType.TEXT),
                Property(name="question_text",  data_type=DataType.TEXT),
                Property(name="split",          data_type=DataType.TEXT),
            ],
        )
        logger.info("Created collection: PaperQuestion")
    else:
        logger.info("Collection PaperQuestion already exists – skipping.")

    # ── PaperEvidence ─────────────────────────────────────
    if not client.collections.exists("PaperEvidence"):
        client.collections.create(
            name="PaperEvidence",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="question_id",    data_type=DataType.TEXT),
                Property(name="document_id",    data_type=DataType.TEXT),
                Property(name="question_text",  data_type=DataType.TEXT),
                Property(name="evidence_text",  data_type=DataType.TEXT),
                Property(name="total_counts",   data_type=DataType.INT),
            ],
        )
        logger.info("Created collection: PaperEvidence")
    else:
        logger.info("Collection PaperEvidence already exists – skipping.")


# ── Schema Deletion ──────────────────────────────────────
def delete_schema(client: weaviate.WeaviateClient):
    """Drop every QASPER-related collection."""
    for name in COLLECTIONS:
        if client.collections.exists(name):
            client.collections.delete(name)
            logger.info("Deleted collection: %s", name)


# ── Standalone entry-point ───────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
    )
    client = get_weaviate_client()
    try:
        create_schema(client, recreate=True)
        logger.info("Schema (re)created successfully.")
    finally:
        client.close()
