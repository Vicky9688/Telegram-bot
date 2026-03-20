"""
rag/retriever.py
Retrieves top-k relevant chunks via cosine similarity,
then feeds them to Ollama (local LLM) to generate a grounded answer.
"""

import numpy as np
import requests
import json
from rag.embedder import get_model, get_all_chunks, embed_documents, DB_PATH

import os

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL   = f"{OLLAMA_HOST}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
TOP_K        = int(os.getenv("RAG_TOP_K", "3"))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))  # Already L2-normalized from embedder


def retrieve_top_k(query: str, k: int = TOP_K) -> list[dict]:
    """
    Embed the query, compare against all stored chunk embeddings,
    and return the top-k most relevant chunks.
    """
    # Ensure DB is populated
    if not DB_PATH.exists():
        print("[Retriever] No embeddings DB found — running embedder first...")
        embed_documents()

    chunks = get_all_chunks()
    if not chunks:
        return []

    model = get_model()
    query_embedding = model.encode(query, normalize_embeddings=True)

    # Score all chunks
    scored = []
    for item in chunks:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append({**item, "score": score})

    # Sort descending by similarity score
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Construct a RAG prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Document {i} — {chunk['doc_name']}]\n{chunk['chunk']}")

    context = "\n\n".join(context_parts)

    return f"""You are a helpful AI assistant. Answer the user's question using ONLY the provided context below.
If the answer is not in the context, say "I don't have information about that in my knowledge base."
Keep your answer concise and factual.

Context:
{context}

Question: {query}

Answer:"""


def call_ollama(prompt: str) -> str:
    """
    Send prompt to Ollama local API and return the generated text.
    Uses streaming=False for simplicity.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,      # Low temp for factual RAG answers
            "num_predict": 150,      # Max tokens in response
            "num_thread": 8,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it's running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama timed out. Try a smaller model.")


def answer_query(query: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve top-k chunks
    2. Build prompt
    3. Call Ollama
    4. Return answer + source metadata
    """
    top_chunks = retrieve_top_k(query)

    if not top_chunks:
        return {
            "answer": "I don't have any documents in my knowledge base yet.",
            "source": "N/A",
            "snippet": "",
        }

    # Best match for source attribution
    best_chunk = top_chunks[0]

    prompt = build_prompt(query, top_chunks)
    answer = call_ollama(prompt)

    # Truncate snippet for display
    snippet = best_chunk["chunk"][:200].replace("\n", " ") + "..."

    return {
        "answer": answer,
        "source": best_chunk["doc_name"],
        "snippet": snippet,
        "score": round(best_chunk["score"], 3),
    }
