"""The Engine ties chunking, embedding and storage together."""

from __future__ import annotations

from pathlib import Path

from askdocs.chunking import chunk_text
from askdocs.embeddings import HashingEmbedder
from askdocs.store import Hit, VectorStore

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}


class Engine:
    def __init__(
        self,
        embedder: HashingEmbedder | None = None,
        store: VectorStore | None = None,
        chunk_size: int = 120,
        overlap: int = 30,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.store = store or VectorStore(dim=self.embedder.dim)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_text(self, source: str, text: str) -> int:
        """Chunk, embed and store ``text``. Returns the number of chunks added."""
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        if not chunks:
            return 0
        return self.store.add(source, chunks, self.embedder.embed(chunks))

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