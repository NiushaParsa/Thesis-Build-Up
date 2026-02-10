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

# ── OpenAI ───────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# ── Weaviate ─────────────────────────────────────────────
WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "localhost")
WEAVIATE_PORT: int = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_API_KEY: str = os.getenv("WEAVIATE_API_KEY", "")

# ── Tokenizer ────────────────────────────────────────────
TOKENIZER_NAME: str = os.getenv("TOKENIZER_NAME", "gpt2")

# ── Output ───────────────────────────────────────────────
JSON_OUTPUT: bool = os.getenv("JSON_OUTPUT", "false").lower() == "true"
JSON_OUTPUT_DIR: str = os.getenv("JSON_OUTPUT_DIR", "json_output")
