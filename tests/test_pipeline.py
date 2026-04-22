from pathlib import Path

from app.pipeline import RAGPipeline


def test_pipeline_ingest_and_retrieve(tmp_path: Path) -> None:
    doc = tmp_path / "demo.txt"
    doc.write_text(
        "RAG retrieves context before generating an answer. "
        "Chunk overlap helps preserve continuity.",
        encoding="utf-8",
    )
    pipeline = RAGPipeline()
    stats = pipeline.ingest_folder(str(tmp_path))
    assert stats["files_processed"] == 1
    assert stats["chunks_created"] >= 1

    hits = pipeline.retrieve("What helps preserve continuity?", top_k=2)
    assert hits
    assert "continuity" in hits[0].text.lower()


def test_pipeline_save_and_load_index(tmp_path: Path) -> None:
    doc = tmp_path / "demo.txt"
    doc.write_text(
        "RAG uses retrieval before generation and can persist an index.",
        encoding="utf-8",
    )
    pipeline = RAGPipeline()
    pipeline.ingest_folder(str(tmp_path))

    index_path = tmp_path / "index.pkl"
    pipeline.save_index(str(index_path))

    fresh = RAGPipeline()
    loaded = fresh.load_index(str(index_path))
    assert loaded
    hits = fresh.retrieve("What can RAG persist?", top_k=1)
    assert hits
