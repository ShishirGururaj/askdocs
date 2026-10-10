"""The Engine ties chunking, embedding and storage together."""

from __future__ import annotations

import re
from pathlib import Path

from askdocs.answer import NO_ANSWER, Answer, Answerer, ExtractiveAnswerer
from askdocs.chunking import chunk_text
from askdocs.embeddings import HashingEmbedder, tokenize
from askdocs.store import Hit, VectorStore

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _normalize(text: str) -> str:
    """Turn markdown headings into their own sentences so they don't fuse with the next line."""
    return _HEADING.sub(r"\1.", text)

class Engine:
    def __init__(
        self,
        embedder: HashingEmbedder | None = None,
        store: VectorStore | None = None,
        chunk_size: int = 120,
        overlap: int = 30,
        answerer: Answerer | None = None,
        min_score: float = 0.05,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.store = store or VectorStore(dim=self.embedder.dim)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.answerer = answerer or ExtractiveAnswerer()
        self.min_score = min_score

    def ingest_text(self, source: str, text: str) -> int:
        """Chunk, embed and store ``text``. Returns the number of chunks added."""
        chunks = chunk_text(_normalize(text), self.chunk_size, self.overlap)
        if not chunks:
            return 0
        return self.store.add(source, chunks, self.embedder.embed(chunks))
    
    def ask(self, question: str, k: int = 4) -> Answer:
        """Retrieve the top-k chunks and answer from them.

        Hits must clear ``min_score`` *and* share at least one term with the question.
        The second check guards against hash collisions in the offline embedder, which
        can otherwise give an unrelated chunk a respectable score.
        """
        q_terms = set(tokenize(question))
        hits = [
            h
            for h in self.search(question, k)
            if h.score >= self.min_score and q_terms & set(tokenize(h.record.text))
        ]
        if not hits:
            return Answer(text=NO_ANSWER)
        sources = list(dict.fromkeys(h.record.source for h in hits))
        return Answer(text=self.answerer.answer(question, hits), sources=sources, hits=hits)


    def ingest_path(self, path: str | Path) -> tuple[int, int]:
        """Ingest a file or a directory tree. Returns ``(files, chunks)``."""
        root = Path(path)
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        n_files = n_chunks = 0
        for f in files:
            if not f.is_file() or f.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            added = self.ingest_text(str(f), f.read_text(encoding="utf-8", errors="ignore"))
            n_files += 1
            n_chunks += added
        return n_files, n_chunks

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.store.search(self.embedder.embed([query])[0], k)

    def save(self, directory: str | Path) -> None:
        self.store.save(directory)

    @classmethod
    def load(cls, directory: str | Path, **kwargs) -> Engine:
        store = VectorStore.load(directory)
        return cls(embedder=HashingEmbedder(dim=store.dim), store=store, **kwargs)