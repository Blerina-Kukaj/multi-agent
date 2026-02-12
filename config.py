"""
Central configuration – loads .env and exposes typed settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "healthcare_docs")

# ── Retrieval ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
