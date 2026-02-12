"""
Vector store wrapper – builds and queries a ChromaDB collection
backed by OpenAI embeddings.

Performance: uses module-level singletons to avoid re-initializing
the embedding model and Chroma client on every call.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config

# ── Singletons (created once, reused across calls) ─────────────────────────
_embeddings: OpenAIEmbeddings | None = None
_cached_store: Chroma | None = None


def get_embeddings() -> OpenAIEmbeddings:
    """Return the shared embedding model (cached)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small",
        )
    return _embeddings


def build_vector_store(docs: list[Document]) -> Chroma:
    """Create (or overwrite) a persisted Chroma collection from chunked docs."""
    global _cached_store
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    _cached_store = vector_store  # cache the freshly-built store
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing Chroma collection from disk (cached after first call)."""
    global _cached_store
    if _cached_store is None:
        embeddings = get_embeddings()
        _cached_store = Chroma(
            collection_name=config.CHROMA_COLLECTION,
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
    return _cached_store


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """Run a similarity search and return the top-k matching chunks."""
    top_k = top_k or config.TOP_K
    store = load_vector_store()
    results = store.similarity_search(query, k=top_k)
    return results


