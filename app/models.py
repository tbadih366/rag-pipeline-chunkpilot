from typing import List

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    folder_path: str = Field(..., description="Local folder containing .txt files")


class IngestResponse(BaseModel):
    files_processed: int
    chunks_created: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)


class RetrievedChunk(BaseModel):
    source: str
    chunk_id: int
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved: List[RetrievedChunk]

