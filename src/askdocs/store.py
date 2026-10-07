"""In-memory vector store with exact cosine-similarity search."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Record:
    id: int
    source: str
    text: str


@dataclass(frozen=True)
class Hit:
    record: Record
    score: float


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._records: list[Record] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[Record]:
        return list(self._records)

    def add(self, source: str, texts: list[str], vectors: np.ndarray) -> int:
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have the same length")
        if len(texts) and vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {vectors.shape[1]}")
        start = len(self._records)
        self._records.extend(
            Record(id=start + i, source=source, text=t) for i, t in enumerate(texts)
        )
        if len(texts):
            self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        return len(texts)

    def search(self, query: np.ndarray, k: int = 5) -> list[Hit]:
        if not self._records or k <= 0:
            return []
        scores = self._vectors @ query.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(self._records[i], float(scores[i])) for i in top]
    
        def save(self, directory: str | Path) -> None:
        """Write ``vectors.npy`` and ``records.json`` into ``directory``."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        payload = {"dim": self.dim, "records": [asdict(r) for r in self._records]}
        (path / "records.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        path = Path(directory)
        payload = json.loads((path / "records.json").read_text(encoding="utf-8"))
        store = cls(dim=payload["dim"])
        store._vectors = np.load(path / "vectors.npy")
        store._records = [Record(**r) for r in payload["records"]]
        if len(store._records) != len(store._vectors):
            raise ValueError("corrupt index: record and vector counts differ")
        return store