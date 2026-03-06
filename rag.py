"""
RAG module — builds and queries the ChromaDB knowledge-base collection.

Sources:
  1. Structured entries from knowledge_base.json
  2. PDF chunks from the docs/ folder (auto-ingested via pdfplumber)
"""

import json
import re
from pathlib import Path

import streamlit as st
import chromadb

from config import KB_PATH, DOCS_DIR, CHROMA_DIR


# ── Chunking configuration ─────────────────────────────────────
PDF_CHUNK_SIZE = 700
PDF_CHUNK_OVERLAP = 120


# ── PDF ingestion helper ───────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentences using punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Build chunks from sentence units while preserving overlap between chunks.
    Overlap is applied in character terms (approximate, sentence-preserving).
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence) + 1
        if current and current_len + s_len > chunk_size:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Keep a trailing overlap window from the previous chunk
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                s_size = len(s) + 1
                if overlap_len + s_size > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += s_size

            current = overlap_sentences[:]
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += s_len

    final_chunk = " ".join(current).strip()
    if final_chunk:
        chunks.append(final_chunk)

    return chunks


def _looks_like_heading(line: str) -> bool:
    """Heuristic to detect likely section headings in extracted PDF text."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) <= 80 and stripped == stripped.upper():
        return True
    if re.match(r"^(Chapter|Unit|Exercise|Example|Theorem|Formula|Topic)\b", stripped, re.IGNORECASE):
        return True
    if re.match(r"^\d+(\.\d+)*\s+", stripped):
        return True
    return False


def _split_into_sections(full_text: str) -> list[str]:
    """
    Split text into sections using heading-like lines.
    Falls back to paragraph grouping if no headings are detected.
    """
    lines = [ln.rstrip() for ln in full_text.splitlines()]
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if _looks_like_heading(line) and current:
            sec = "\n".join(current).strip()
            if sec:
                sections.append(sec)
            current = [line]
        else:
            current.append(line)

    tail = "\n".join(current).strip()
    if tail:
        sections.append(tail)

    # Fallback: if we got one huge section, split by blank lines (paragraphs)
    if len(sections) <= 1:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]
        if paragraphs:
            return paragraphs

    return sections


def _load_pdf_chunks(pdf_path: Path, chunk_size: int = PDF_CHUNK_SIZE, overlap: int = PDF_CHUNK_OVERLAP) -> list[dict]:
    """
    Extract text from a PDF and split into section-aware chunks with overlap.
    Each chunk carries source metadata for traceability in the UI.
    """
    try:
        import pdfplumber
    except ImportError:
        st.warning(
            f"pdfplumber not installed — skipping {pdf_path.name}.  "
            "Run `pip install pdfplumber` to enable PDF ingestion."
        )
        return []

    chunks: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

    if not full_text.strip():
        return []

    sections = _split_into_sections(full_text)
    for idx, section in enumerate(sections):
        section_chunks = _chunk_with_overlap(
            text=section,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk_idx, chunk_text in enumerate(section_chunks):
            chunks.append({
                "text": chunk_text,
                "source": pdf_path.name,
                "section": idx,
                "chunk": chunk_idx,
            })

    return chunks


# ── ChromaDB collection ────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base into vector store …")
def get_rag_collection():
    """Build (or reload) the RAG collection from JSON + PDFs."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name="math_knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        ids, documents, metadatas = [], [], []

        # ── Structured JSON entries ──
        if KB_PATH.exists():
            with open(KB_PATH, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for i, entry in enumerate(entries):
                doc_text = f"{entry['formula']}  |  Example: {entry['example']}"
                ids.append(f"kb_{i}")
                documents.append(doc_text)
                metadatas.append({"topic": entry["topic"], "source": "knowledge_base.json"})

        # ── PDF documents ──
        if DOCS_DIR.exists():
            pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
            pdf_chunk_count = 0
            for pdf_path in pdf_files:
                chunks = _load_pdf_chunks(pdf_path)
                for j, chunk in enumerate(chunks):
                    ids.append(f"pdf_{pdf_path.stem}_{j}")
                    documents.append(chunk["text"])
                    metadatas.append({
                        "topic": "pdf",
                        "source": chunk["source"],
                        "section": chunk.get("section", 0),
                        "chunk": chunk.get("chunk", j),
                    })
                    pdf_chunk_count += 1
            if pdf_chunk_count:
                st.toast(f"Loaded {pdf_chunk_count} chunks from {len(pdf_files)} PDF(s) in docs/", icon="📄")

        if ids:
            for start in range(0, len(ids), 5000):
                end = start + 5000
                collection.add(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )

    return collection


def rag_retrieve(query: str, n_results: int = 3) -> list[str]:
    """Return top-N matching documents from the knowledge base."""
    collection = get_rag_collection()
    if collection.count() == 0:
        return ["(No knowledge-base entries found.)"]
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]
