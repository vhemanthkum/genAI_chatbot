"""
vector_store.py
===============
Persistent ChromaDB wrapper for the Multi-Modal Assistant knowledge base.

Provides semantic search over ingested documents using local sentence-transformers
embeddings (all-MiniLM-L6-v2) — no extra API keys required for retrieval.
"""

from __future__ import annotations

import hashlib
import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "multimodal_assistant_kb"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL_NAME
)

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embed_fn,
    metadata={"hnsw:space": "cosine"},
)


def _doc_id(text: str, source: str) -> str:
    return hashlib.md5(f"{source}::{text}".encode()).hexdigest()


def add_documents(chunks: List[str], source: str = "unknown") -> int:
    """Embed and upsert text chunks. Returns number of chunks upserted."""
    if not chunks:
        return 0

    seen: dict[str, str] = {}
    for chunk in chunks:
        doc_id = _doc_id(chunk, source)
        if doc_id not in seen:
            seen[doc_id] = chunk

    ids = list(seen.keys())
    documents = list(seen.values())
    metadatas = [{"source": source} for _ in ids]

    _collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def query(text: str, n_results: int = 3) -> List[dict]:
    """Semantic nearest-neighbour search. Returns list of {text, source, distance}."""
    if _collection.count() == 0:
        return []

    results = _collection.query(
        query_texts=[text],
        n_results=min(n_results, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "distance": dist,
            }
        )
    return hits


def count() -> int:
    return _collection.count()


def clear() -> None:
    global _collection
    _client.delete_collection(COLLECTION_NAME)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
