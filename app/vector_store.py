from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class StoredChunk:
    source: str
    chunk_id: int
    text: str


@dataclass
class SearchResult:
    source: str
    chunk_id: int
    text: str
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._chunks: List[StoredChunk] = []
        self._matrix = None

    @property
    def chunks(self) -> List[StoredChunk]:
        return self._chunks

    def reset(self) -> None:
        self._chunks = []
        self._matrix = None

    def add_chunks(self, chunks: List[StoredChunk]) -> None:
        self._chunks.extend(chunks)
        if self._chunks:
            corpus = [c.text for c in self._chunks]
            self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not self._chunks or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        ranked_indices = sims.argsort()[::-1][:top_k]
        results: List[SearchResult] = []
        for idx in ranked_indices:
            chunk = self._chunks[int(idx)]
            results.append(
                SearchResult(
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(sims[int(idx)]),
                )
            )
        return results

    def save(self, path: str) -> None:
        payload = {
            "chunks": self._chunks,
            "vectorizer": self._vectorizer,
            "matrix": self._matrix,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> bool:
        target = Path(path)
        if not target.exists():
            return False
        with target.open("rb") as f:
            payload = pickle.load(f)
        self._chunks = payload.get("chunks", [])
        self._vectorizer = payload.get("vectorizer", TfidfVectorizer(stop_words="english"))
        self._matrix = payload.get("matrix")
        return True
