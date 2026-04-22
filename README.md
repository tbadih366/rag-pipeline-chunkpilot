# ContextCraft-RAG

A compact RAG pipeline you can run locally in a few minutes.

This project does four things well:
- ingests `.txt` files from a folder
- chunks text with overlap
- retrieves relevant chunks with vector search
- serves everything through a simple FastAPI API

It is intentionally small, so you can understand every part without fighting framework complexity.

## Stack

- Python 3.11+
- FastAPI
- scikit-learn (TF-IDF + cosine similarity)
- pytest

## Project Layout

```text
app/
  api.py            # FastAPI routes
  chunking.py       # word-based chunking with overlap
  config.py         # runtime settings
  models.py         # request/response schemas
  pipeline.py       # ingest/retrieve/answer orchestration
  vector_store.py   # in-memory vector index + save/load
sample_data/
scripts/
  eval_retrieval.py # quick retrieval quality check
tests/
requirements.txt
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api:app --reload
```

Open:
- API docs: `http://127.0.0.1:8000/docs`

## API Endpoints

- `GET /health`
- `POST /ingest`
- `POST /query`
- `GET /chunks`
- `DELETE /reset`
- `POST /demo-load`
- `POST /persist/save`
- `POST /persist/load`

## Typical Flow

1. Load sample data:
```http
POST /demo-load
```

2. Ask a question:
```json
POST /query
{
  "question": "Why does RAG reduce hallucinations?",
  "top_k": 3
}
```

3. Save the index:
```json
POST /persist/save
{
  "path": "data/index.pkl"
}
```

4. Later, load it back:
```json
POST /persist/load
{
  "path": "data/index.pkl"
}
```

## Evaluation Script

Run a quick retrieval sanity check:

```bash
python scripts/eval_retrieval.py
```

It prints:
- ingest stats
- top-k hit behavior on sample prompts
- average retrieval latency

## Tests

```bash
python -m pytest -q
```

## Notes

- This is a retrieval-first baseline. It is fast, transparent, and easy to inspect.
- The answer generator is deliberately lightweight and returns cited chunk context.
- For production at scale, swap in dense embeddings and a persistent vector DB.

## Roadmap

- Add optional dense embeddings backend
- Add re-ranking step
- Add metrics endpoint (latency, recall proxy)
- Add UI client for interactive querying

