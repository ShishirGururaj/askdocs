"""FastAPI application exposing the engine over HTTP."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from askdocs import __version__
from askdocs.config import Settings
from askdocs.engine import Engine
from askdocs.llm import AnswerError, build_answerer


class DocumentIn(BaseModel):
    source: str = Field(min_length=1, max_length=500, examples=["handbook.md"])
    text: str = Field(min_length=1)


class DocumentOut(BaseModel):
    source: str
    chunks: int


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)


class HitOut(BaseModel):
    source: str
    score: float
    text: str


class QueryOut(BaseModel):
    answer: str
    sources: list[str]
    hits: list[HitOut]


def create_app(engine: Engine | None = None, index_dir: str | None = None) -> FastAPI:
    """Build the app. If ``index_dir`` is set, the index is saved after every ingest."""
    engine = engine or Engine()
    app = FastAPI(title="askdocs", version=__version__)

    @app.exception_handler(AnswerError)
    async def answer_error_handler(request: Request, exc: AnswerError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "chunks": len(engine.store)}

    @app.post("/documents", response_model=DocumentOut, status_code=201)
    def add_document(doc: DocumentIn) -> DocumentOut:
        chunks = engine.ingest_text(doc.source, doc.text)
        if index_dir:
            engine.save(index_dir)
        return DocumentOut(source=doc.source, chunks=chunks)

    @app.post("/query", response_model=QueryOut)
    def query(body: QueryIn) -> QueryOut:
        result = engine.ask(body.question, k=body.top_k)
        return QueryOut(
            answer=result.text,
            sources=result.sources,
            hits=[
                HitOut(source=h.record.source, score=round(h.score, 4), text=h.record.text)
                for h in result.hits
            ],
        )

    return app


def create_app_from_env() -> FastAPI:
    """App factory for uvicorn: loads an existing index from ``ASKDOCS_INDEX`` if present."""
    settings = Settings.from_env()
    exists = (Path(settings.index_dir) / "records.json").exists()
    answerer = build_answerer(settings)
    engine = (
        Engine.load(settings.index_dir, answerer=answerer) if exists else Engine(answerer=answerer)
    )
    return create_app(engine, index_dir=settings.index_dir)