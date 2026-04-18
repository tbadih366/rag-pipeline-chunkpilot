from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    """
    Split text into overlapping word chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        piece = words[start:end]
        if not piece:
            continue
        chunks.append(" ".join(piece))
        if end >= len(words):
            break
    return chunks

