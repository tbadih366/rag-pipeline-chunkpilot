from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from app.chunking import chunk_text
from app.config import settings
from app.vector_store import SearchResult, StoredChunk, VectorStore


class RAGPipeline:
    def __init__(self) -> None:
        self.store = VectorStore()

    def reset(self) -> None:
        self.store.reset()

    def ingest_folder(self, folder_path: str) -> Dict[str, int]:
        base = Path(folder_path)
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        text_files = sorted(base.glob("*.txt"))
        all_chunks: List[StoredChunk] = []
        for file_path in text_files:
            raw = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(
                raw,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            for idx, chunk in enumerate(chunks):
                all_chunks.append(
                    StoredChunk(source=file_path.name, chunk_id=idx, text=chunk)
                )

        self.store.add_chunks(all_chunks)
        return {"files_processed": len(text_files), "chunks_created": len(all_chunks)}

    def retrieve(self, question: str, top_k: int = 3) -> List[SearchResult]:
        return self.store.search(query=question, top_k=top_k)

    def answer(self, question: str, top_k: int = 3) -> Dict[str, object]:
        hits = self.retrieve(question=question, top_k=top_k)
        if not hits:
            return {
                "answer": "I do not have knowledge loaded yet. Ingest text files first.",
                "retrieved": [],
            }

        lines = [
            "Here is an evidence-based answer generated from retrieved chunks:",
            "",
        ]
        for idx, hit in enumerate(hits, start=1):
            lines.append(
                f"{idx}. [{hit.source} | chunk {hit.chunk_id} | score={hit.score:.3f}] {hit.text}"
            )
        lines.append("")
        lines.append(
            "Summary: the highest scoring chunks above are the context you should trust first."
        )
        return {"answer": "\n".join(lines), "retrieved": hits}

    def save_index(self, path: str | None = None) -> str:
        target = path or settings.index_path
        self.store.save(target)
        return target

    def load_index(self, path: str | None = None) -> bool:
        target = path or settings.index_path
        return self.store.load(target)
