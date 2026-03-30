"""
utils.py
--------
Shared helpers: sentence splitting and text preprocessing.
"""

import re


def split_sentences(text: str) -> list[str]:
    """
    Split *text* into sentences using a simple regex heuristic.
    Falls back gracefully for edge cases.
    """
    # Split on '. ', '! ', '? ' followed by a capital letter or end-of-string
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences if sentences else [text.strip()]


def preprocess_text(text: str) -> str:
    """Light cleaning: collapse whitespace, strip leading/trailing spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_entity_sentences(entity: str, sentences: list[str]) -> list[str]:
    """Return sentences that mention *entity* (case-insensitive)."""
    lower_entity = entity.lower()
    return [s for s in sentences if lower_entity in s.lower()]
