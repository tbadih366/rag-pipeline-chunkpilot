from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.models import (
    IngestRequest,
    IngestResponse,
    PersistRequest,
    PersistResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from app.pipeline import RAGPipeline

app = FastAPI(title="RAG Pipeline API", version="1.0.0")
pipeline = RAGPipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    try:
        result = pipeline.ingest_folder(req.folder_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = pipeline.answer(req.question, top_k=req.top_k)
    retrieved = [
        RetrievedChunk(
            source=r.source,
            chunk_id=r.chunk_id,
            score=r.score,
            text=r.text,
        )
        for r in result["retrieved"]
    ]
    return QueryResponse(question=req.question, answer=result["answer"], retrieved=retrieved)


@app.get("/chunks")
def list_chunks() -> dict:
    chunks = pipeline.store.chunks
    return {"count": len(chunks), "items": [c.__dict__ for c in chunks]}


@app.delete("/reset")
def reset() -> dict:
    pipeline.reset()
    return {"status": "reset"}


@app.post("/demo-load", response_model=IngestResponse)
def demo_load() -> IngestResponse:
    demo_folder = Path(__file__).resolve().parents[1] / "sample_data"
    pipeline.reset()
    result = pipeline.ingest_folder(str(demo_folder))
    return IngestResponse(**result)


@app.post("/persist/save", response_model=PersistResponse)
def persist_save(req: PersistRequest) -> PersistResponse:
    saved_path = pipeline.save_index(req.path)
    return PersistResponse(path=saved_path)


@app.post("/persist/load", response_model=PersistResponse)
def persist_load(req: PersistRequest) -> PersistResponse:
    target_path = req.path or settings.index_path
    loaded = pipeline.load_index(req.path)
    if not loaded:
        raise HTTPException(status_code=404, detail="Index file not found")
    return PersistResponse(path=target_path, loaded=True)
