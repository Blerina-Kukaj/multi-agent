"""
Vector store wrapper – builds and queries a ChromaDB collection
backed by OpenAI embeddings.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config


def get_embeddings() -> OpenAIEmbeddings:
    """Return the shared embedding model."""
    return OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY,
        model="text-embedding-3-small",
    )


def build_vector_store(docs: list[Document]) -> Chroma:
    """Create (or overwrite) a persisted Chroma collection from chunked docs."""
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing Chroma collection from disk."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """Run a similarity search and return the top-k matching chunks."""
    top_k = top_k or config.TOP_K
    store = load_vector_store()
    results = store.similarity_search(query, k=top_k)
    return results


def retrieve_with_scores(
    query: str, top_k: int | None = None
) -> list[tuple[Document, float]]:
    """Run a similarity search and return (document, score) pairs."""
    top_k = top_k or config.TOP_K
    store = load_vector_store()
    return store.similarity_search_with_relevance_scores(query, k=top_k)
