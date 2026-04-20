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

