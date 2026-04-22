from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_demo_load_and_query() -> None:
    client.delete("/reset")
    load = client.post("/demo-load")
    assert load.status_code == 200
    assert load.json()["files_processed"] >= 1

    res = client.post(
        "/query",
        json={"question": "Why does RAG reduce hallucinations?", "top_k": 2},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["retrieved"]
    assert "retrieved chunks" in payload["answer"].lower()


def test_query_without_ingest_returns_empty_retrieval() -> None:
    client.delete("/reset")
    res = client.post(
        "/query",
        json={"question": "What is RAG?", "top_k": 2},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["retrieved"] == []


def test_persist_save_and_load(tmp_path) -> None:
    client.delete("/reset")
    load = client.post("/demo-load")
    assert load.status_code == 200

    index_file = tmp_path / "api_index.pkl"
    save_res = client.post("/persist/save", json={"path": str(index_file)})
    assert save_res.status_code == 200

    client.delete("/reset")
    load_res = client.post("/persist/load", json={"path": str(index_file)})
    assert load_res.status_code == 200
    assert load_res.json()["loaded"] is True
