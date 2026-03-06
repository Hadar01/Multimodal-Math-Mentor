"""
Memory module — stores and retrieves previously solved Q&A pairs
so the Solver can reuse solution patterns.
"""

import json
import hashlib

import streamlit as st
import chromadb

from config import CHROMA_DIR


@st.cache_resource(show_spinner="Initializing memory store …")
def get_memory_collection():
    """Return (or create) the ChromaDB collection for solved-problem memory."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="solved_memory",
        metadata={"hnsw:space": "cosine"},
    )


def memory_search(query: str, n_results: int = 2) -> str:
    """Search memory for previously solved problems similar to `query`."""
    collection = get_memory_collection()
    if collection.count() == 0:
        return ""
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    if docs:
        return "\n---\n".join(docs)
    return ""


def memory_save(record: dict) -> None:
    """Persist a full Q&A record into memory (upsert by question hash)."""
    collection = get_memory_collection()
    question = record.get("parsed_question", "")
    doc_id = hashlib.sha256(question.encode()).hexdigest()[:16]
    doc_text = json.dumps(record, ensure_ascii=False, indent=2)
    collection.upsert(ids=[doc_id], documents=[doc_text])
