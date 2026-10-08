from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askdocs.api import create_app
from askdocs.engine import Engine

DOCS = Path(__file__).resolve().parent.parent / "examples" / "docs"


@pytest.fixture
def client():
    engine = Engine()
    engine.ingest_path(DOCS)
    return TestClient(create_app(engine))


def test_health_reports_chunk_count(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 3


def test_query_returns_answer_and_sources(client):
    r = client.post("/query", json={"question": "How long do access tokens last?"})
    assert r.status_code == 200
    body = r.json()
    assert "one hour" in body["answer"]
    assert body["sources"][0].endswith("security.md")
    assert body["hits"][0]["score"] > 0


def test_ingest_then_query_new_document():
    client = TestClient(create_app())
    r = client.post(
        "/documents",
        json={"source": "pets.md", "text": "Our office dog is called Biscuit and loves walks."},
    )
    assert r.status_code == 201
    assert r.json() == {"source": "pets.md", "chunks": 1}
    answer = client.post("/query", json={"question": "What is the office dog called?"}).json()
    assert "Biscuit" in answer["answer"]


def test_validation_errors(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "hi", "top_k": 99}).status_code == 422
    assert client.post("/documents", json={"source": "x"}).status_code == 422


def test_ingest_persists_when_index_dir_set(tmp_path):
    client = TestClient(create_app(index_dir=str(tmp_path)))
    client.post("/documents", json={"source": "a.md", "text": "persist this content"})
    assert (tmp_path / "records.json").exists()