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

