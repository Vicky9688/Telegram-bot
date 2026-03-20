"""
rag/embedder.py
Chunks documents, generates embeddings, and stores them in SQLite.
Model: all-MiniLM-L6-v2 (fast, 384-dim, great for small knowledge bases)
"""

import os
import sqlite3
import json
import hashlib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "docs"
DB_PATH = Path(__file__).parent / "embeddings.db"

# Chosen for speed + accuracy tradeoff at small scale.
# 384-dim vectors are compact enough for SQLite without a vector DB.
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300       # characters per chunk
CHUNK_OVERLAP = 50     # overlap to preserve context across boundaries

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (only once per process)."""
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def init_db():
    """Create the embeddings table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT    NOT NULL,
            chunk    TEXT    NOT NULL,
            hash     TEXT    UNIQUE NOT NULL,
            embedding BLOB   NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def doc_already_embedded(doc_name: str, conn: sqlite3.Connection) -> bool:
    """Check if a document has already been embedded (skip re-embedding)."""
    cur = conn.execute("SELECT 1 FROM chunks WHERE doc_name = ? LIMIT 1", (doc_name,))
    return cur.fetchone() is not None


def embed_documents(force: bool = False):
    """
    Read all .md / .txt files in docs/, embed them, and store in SQLite.
    Skips documents already embedded unless force=True.
    """
    init_db()
    model = get_model()
    conn = sqlite3.connect(DB_PATH)

    doc_files = list(DOCS_DIR.glob("*.md")) + list(DOCS_DIR.glob("*.txt"))
    if not doc_files:
        print(f"[Embedder] No documents found in {DOCS_DIR}")
        conn.close()
        return

    for doc_path in doc_files:
        doc_name = doc_path.name

        if not force and doc_already_embedded(doc_name, conn):
            print(f"[Embedder] Skipping (already embedded): {doc_name}")
            continue

        print(f"[Embedder] Processing: {doc_name}")
        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for chunk in chunks:
            chunk_hash = hashlib.md5(f"{doc_name}:{chunk}".encode()).hexdigest()
            embedding = model.encode(chunk, normalize_embeddings=True)
            embedding_blob = json.dumps(embedding.tolist())

            try:
                conn.execute(
                    "INSERT OR IGNORE INTO chunks (doc_name, chunk, hash, embedding) VALUES (?, ?, ?, ?)",
                    (doc_name, chunk, chunk_hash, embedding_blob),
                )
            except sqlite3.IntegrityError:
                pass  # Duplicate chunk hash, skip

        conn.commit()
        print(f"[Embedder] Embedded {len(chunks)} chunks from {doc_name}")

    conn.close()
    print("[Embedder] Done.")


def get_all_chunks() -> list[dict]:
    """Load all chunks + embeddings from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT doc_name, chunk, embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()

    results = []
    for doc_name, chunk, embedding_blob in rows:
        embedding = np.array(json.loads(embedding_blob), dtype=np.float32)
        results.append({"doc_name": doc_name, "chunk": chunk, "embedding": embedding})

    return results


if __name__ == "__main__":
    embed_documents(force=False)
