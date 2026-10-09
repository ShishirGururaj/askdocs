"""Optional LLM-backed answerer using the Gemini REST API (no extra SDK needed)."""

from __future__ import annotations

import httpx

from askdocs.answer import Answerer, ExtractiveAnswerer
from askdocs.config import Settings
from askdocs.store import Hit

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = """You answer questions using only the context below.
If the context does not contain the answer, say you could not find it.
Be concise.

Context:
{context}

Question: {question}
Answer:"""


class AnswerError(RuntimeError):
    """Raised when the LLM backend fails or returns something unusable."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{h.record.source}]\n{h.record.text}" for h in hits)
    return PROMPT.format(context=context, question=question)


class GeminiAnswerer:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-flash-latest",
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=timeout)

    def answer(self, question: str, hits: list[Hit]) -> str:
        payload = {"contents": [{"parts": [{"text": build_prompt(question, hits)}]}]}
        try:
            response = self._client.post(
                GEMINI_URL.format(model=self.model),
                headers={"x-goog-api-key": self.api_key},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise AnswerError(f"could not reach Gemini: {exc}") from exc
        if response.status_code != 200:
            raise AnswerError(f"Gemini returned HTTP {response.status_code}")
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AnswerError("unexpected response shape from Gemini") from exc


def build_answerer(settings: Settings) -> Answerer:
    if settings.llm == "extractive":
        return ExtractiveAnswerer()
    if settings.llm == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("ASKDOCS_LLM=gemini requires GEMINI_API_KEY to be set")
        return GeminiAnswerer(settings.gemini_api_key, settings.gemini_model)
    raise ValueError(f"unknown ASKDOCS_LLM provider: {settings.llm!r}")