import httpx
import pytest
from fastapi.testclient import TestClient

from askdocs.api import create_app
from askdocs.config import Settings
from askdocs.engine import Engine
from askdocs.llm import AnswerError, GeminiAnswerer, build_answerer, build_prompt
from askdocs.store import Hit, Record

HITS = [Hit(Record(0, "security.md", "Tokens expire after one hour."), 0.5)]


def make_answerer(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiAnswerer("test-key", model="m", client=client)


def test_prompt_contains_context_and_question():
    prompt = build_prompt("How long?", HITS)
    assert "Tokens expire after one hour." in prompt
    assert "[security.md]" in prompt
    assert "How long?" in prompt


def test_gemini_success_sends_key_and_parses_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers["x-goog-api-key"]
        seen["url"] = str(request.url)
        body = {"candidates": [{"content": {"parts": [{"text": " One hour. "}]}}]}
        return httpx.Response(200, json=body)

    assert make_answerer(handler).answer("How long?", HITS) == "One hour."
    assert seen["key"] == "test-key"
    assert seen["url"].endswith("/models/m:generateContent")


@pytest.mark.parametrize(
    "status,body",
    [(429, ""), (200, {"candidates": []}), (200, "not json")],
)
def test_gemini_failures_raise_answer_error(status, body):
    def handler(request):
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body)

    with pytest.raises(AnswerError):
        make_answerer(handler).answer("q", HITS)


def test_network_error_is_wrapped():
    def boom(request):
        raise httpx.ConnectError("down")

    with pytest.raises(AnswerError):
        make_answerer(boom).answer("q", HITS)


def test_build_answerer_validates_settings():
    assert build_answerer(Settings()).__class__.__name__ == "ExtractiveAnswerer"
    with pytest.raises(ValueError):
        build_answerer(Settings(llm="gemini"))
    with pytest.raises(ValueError):
        build_answerer(Settings(llm="nope"))
    assert isinstance(build_answerer(Settings(llm="gemini", gemini_api_key="k")), GeminiAnswerer)


def test_api_maps_llm_failure_to_502():
    engine = Engine(answerer=make_answerer(lambda request: httpx.Response(500)))
    engine.ingest_text("a.md", "Tokens expire after one hour.")
    client = TestClient(create_app(engine))
    r = client.post("/query", json={"question": "When do tokens expire?"})
    assert r.status_code == 502
    assert "500" in r.json()["detail"]