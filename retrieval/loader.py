"""
Document loader – reads .txt files from /data, chunks them, and returns
LangChain Document objects with metadata (source filename + chunk id).
"""

import os
import glob

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents(data_dir: str | None = None) -> list[Document]:
    """Load all .txt files from the data directory and return raw Documents."""
    data_dir = data_dir or config.DATA_DIR
    docs: list[Document] = []

    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(
            Document(page_content=text, metadata={"source": filename})
        )

    if not docs:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}. "
            "Please add sample documents to the /data folder."
        )
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into smaller chunks, preserving metadata + adding chunk_id."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked: list[Document] = []
    # Group by source to assign per-document chunk IDs
    source_chunks: dict[str, int] = {}

    for doc in docs:
        splits = splitter.split_documents([doc])
        source = doc.metadata.get("source", "unknown")
        start_idx = source_chunks.get(source, 0)
        for i, chunk in enumerate(splits):
            chunk_id = start_idx + i + 1  # 1-indexed
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["citation"] = f"[{source} | Chunk #{chunk_id}]"
            chunked.append(chunk)
        source_chunks[source] = start_idx + len(splits)

    return chunked


if __name__ == "__main__":
    raw = load_documents()
    chunks = chunk_documents(raw)
    print(f"Loaded {len(raw)} documents → {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"  {c.metadata['citation']}  ({len(c.page_content)} chars)")
