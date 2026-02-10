"""
Ingestion script – load documents, chunk them, and build the vector store.

Usage:  python -m retrieval.ingest
"""

from retrieval.loader import load_documents, chunk_documents
from retrieval.vector_store import build_vector_store


def main() -> None:
    print("📄 Loading documents …")
    raw_docs = load_documents()
    print(f"   Found {len(raw_docs)} documents")

    print("✂️  Chunking …")
    chunks = chunk_documents(raw_docs)
    print(f"   Created {len(chunks)} chunks")

    print("🧠 Building vector store …")
    store = build_vector_store(chunks)
    count = store._collection.count()
    print(f"   Stored {count} vectors in ChromaDB")

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    main()
