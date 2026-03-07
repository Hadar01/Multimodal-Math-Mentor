"""
Memory module — stores and retrieves previously solved Q&A pairs
so the Solver can reuse solution patterns.
"""

import json
import hashlib

import streamlit as st

from config import CHROMA_DIR, ENABLE_CHROMA


def _truncate(text: str, limit: int = 280) -> str:
    """Trim long text for compact display/prompt context."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _format_memory_doc(doc: str) -> str:
    """Convert a stored JSON memory record into a readable summary block."""
    try:
        record = json.loads(doc)
    except json.JSONDecodeError:
        return _truncate(doc, limit=320)

    question = record.get("parsed_question") or record.get("original_input") or "Unknown question"
    final_answer = record.get("final_answer") or ""
    verifier = record.get("verifier_outcome") or {}
    confidence = verifier.get("confidence", "unknown")
    feedback = record.get("user_feedback")

    lines = [
        f"**Similar Question:** {_truncate(question, limit=180)}",
        f"**Previous Answer:** {_truncate(final_answer, limit=260)}",
        f"**Verifier Confidence:** {confidence}",
    ]

    if isinstance(feedback, dict) and feedback.get("label"):
        label = str(feedback.get("label", "")).capitalize()
        comment = _truncate(str(feedback.get("comment", "")), limit=120)
        if comment:
            lines.append(f"**User Feedback:** {label} - {comment}")
        else:
            lines.append(f"**User Feedback:** {label}")

    return "\n".join(lines)


@st.cache_resource(show_spinner="Initializing memory store …")
def get_memory_collection():
    """Return (or create) the ChromaDB collection for solved-problem memory."""
    if not ENABLE_CHROMA:
        return None

    try:
        import chromadb
    except Exception:
        st.warning("ChromaDB unavailable; long-term memory disabled.")
        return None

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="solved_memory",
        metadata={"hnsw:space": "cosine"},
    )


def memory_search(query: str, n_results: int = 2) -> str:
    """Search memory for previously solved problems similar to `query`."""
    if not ENABLE_CHROMA:
        return ""

    collection = get_memory_collection()
    if collection is None:
        return ""

    if collection.count() == 0:
        return ""
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    if docs:
        return "\n\n---\n\n".join(_format_memory_doc(doc) for doc in docs)
    return ""


def memory_save(record: dict) -> None:
    """Persist a full Q&A record into memory (upsert by question hash)."""
    if not ENABLE_CHROMA:
        return

    collection = get_memory_collection()
    if collection is None:
        return

    question = record.get("parsed_question", "")
    doc_id = hashlib.sha256(question.encode()).hexdigest()[:16]
    doc_text = json.dumps(record, ensure_ascii=False, indent=2)
    collection.upsert(ids=[doc_id], documents=[doc_text])
