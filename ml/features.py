"""Text feature extraction helpers (regex-based, no external NLP models)."""

from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "i", "my", "have", "has",
    "been", "be", "am", "it", "its", "this", "that",
}


def extract_features(text: str) -> dict:
    """Return simple token and keyword features from a symptom string.

    Args:
        text: Raw symptom description.

    Returns:
        Dict with ``tokens`` (lowercased content words) and ``entities``
        (empty list — placeholder kept for API compatibility).
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    tokens = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    return {
        "tokens": tokens,
        "entities": [],
    }
