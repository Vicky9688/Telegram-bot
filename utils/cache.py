"""
utils/cache.py
Simple query cache using SQLite.
Key: MD5 hash of the normalized query string.
Avoids re-embedding / re-querying LLM for identical questions.
"""

import sqlite3
import hashlib
import json
from pathlib import Path

CACHE_DB = Path(__file__).parent.parent / "cache.db"


def _init_cache():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash TEXT PRIMARY KEY,
            query      TEXT NOT NULL,
            result     TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _hash_query(query: str) -> str:
    """Normalize and hash query for cache key."""
    normalized = query.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()


def get_cached(query: str) -> dict | None:
    """Return cached result dict if it exists, else None."""
    _init_cache()
    query_hash = _hash_query(query)

    conn = sqlite3.connect(CACHE_DB)
    cur = conn.execute(
        "SELECT result FROM query_cache WHERE query_hash = ?", (query_hash,)
    )
    row = cur.fetchone()
    conn.close()

    if row:
        return json.loads(row[0])
    return None


def set_cache(query: str, result: dict):
    """Store a query result in the cache."""
    _init_cache()
    query_hash = _hash_query(query)

    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "INSERT OR REPLACE INTO query_cache (query_hash, query, result) VALUES (?, ?, ?)",
        (query_hash, query.strip().lower(), json.dumps(result)),
    )
    conn.commit()
    conn.close()


def clear_cache():
    """Wipe the entire cache (utility function)."""
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("DELETE FROM query_cache")
    conn.commit()
    conn.close()
    print("[Cache] Cleared.")
