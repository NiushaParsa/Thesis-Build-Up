"""
Configuration Module
====================
Loads environment variables from .env and provides centralised
configuration for every module in the QASPER pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Chunk Sizes ──────────────────────────────────────────
CHUNK_SIZES = [
    int(x.strip())
    for x in os.getenv("CHUNK_SIZES", "10,20,40,80,160").split(",")
]
GRANULARITY_LEVEL_TO_TOKENS = {
    level: tokens for level, tokens in enumerate(CHUNK_SIZES, start=1)
}

# ── OpenAI ───────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM: int = int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "1536"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# ── Qdrant ───────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT: int = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

# ── Collections ─────────────────────────────────────────
PAPER_CHUNK_COLLECTION: str = os.getenv("PAPER_CHUNK_COLLECTION", "PaperChunk")
PAPER_QUESTION_COLLECTION: str = os.getenv("PAPER_QUESTION_COLLECTION", "PaperQuestion")
PAPER_EVIDENCE_COLLECTION: str = os.getenv("PAPER_EVIDENCE_COLLECTION", "PaperEvidence")
RETRIEVAL_EVALUATION_COLLECTION: str = os.getenv(
    "RETRIEVAL_EVALUATION_COLLECTION", "RetrievalEvaluation"
)
ROUTER_DATASET_COLLECTION: str = os.getenv("ROUTER_DATASET_COLLECTION", "RouterDataset")

# ── Tokenizer ────────────────────────────────────────────
TOKENIZER_NAME: str = os.getenv("TOKENIZER_NAME", "gpt2")

# ── Output ───────────────────────────────────────────────
JSON_OUTPUT: bool = os.getenv("JSON_OUTPUT", "false").lower() == "true"
JSON_OUTPUT_DIR: str = os.getenv("JSON_OUTPUT_DIR", "json_output")
EVALUATION_OUTPUT_DIR: str = os.getenv("EVALUATION_OUTPUT_DIR", "outputs")

# ── Evaluation ──────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MIXED_DEDUP_OVERLAP_THRESHOLD: float = float(
    os.getenv("MIXED_DEDUP_OVERLAP_THRESHOLD", "0.8")
)
MIXED_DEDUP_CANDIDATE_MULTIPLIER: int = int(
    os.getenv("MIXED_DEDUP_CANDIDATE_MULTIPLIER", "10")
)
ROUTER_LABEL_TIE_EPSILON: float = float(os.getenv("ROUTER_LABEL_TIE_EPSILON", "1e-6"))
EVALUATION_UPSERT_BATCH_SIZE: int = int(os.getenv("EVALUATION_UPSERT_BATCH_SIZE", "100"))
PERSIST_EVALUATIONS: bool = os.getenv("PERSIST_EVALUATIONS", "false").lower() == "true"
PERSIST_ROUTER_DATASET: bool = os.getenv("PERSIST_ROUTER_DATASET", "false").lower() == "true"
