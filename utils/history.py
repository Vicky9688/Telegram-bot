"""
utils/history.py
In-memory per-user conversation history (last 3 interactions).
No persistence — resets on bot restart (intentional for simplicity).
For persistence, swap the dict with a SQLite table.
"""

from collections import defaultdict, deque

# user_id -> deque of last 3 interactions
_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=3))


def add_to_history(user_id: str, interaction_type: str, query: str, answer: str):
    """
    Record an interaction for a user.
    type: 'ask' | 'image'
    """
    _history[user_id].append({
        "type": interaction_type,
        "query": query,
        "answer": answer,
    })


def get_history(user_id: str) -> list[dict]:
    """Return the last (up to 3) interactions for a user."""
    return list(_history.get(user_id, []))


def clear_history(user_id: str):
    """Clear history for a specific user."""
    if user_id in _history:
        _history[user_id].clear()
