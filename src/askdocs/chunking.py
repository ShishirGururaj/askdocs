"""Split long text into overlapping word windows."""

from __future__ import annotations


def chunk_text(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """Split ``text`` into windows of ``size`` words, sharing ``overlap`` words.

    Overlap keeps sentences that straddle a boundary retrievable from both sides.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks