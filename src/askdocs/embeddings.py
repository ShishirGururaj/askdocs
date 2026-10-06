"""A deterministic, offline text embedder based on the hashing trick.

No model download, no API key: tokens (and bigrams) are hashed into a fixed-size
vector with signed counts, then L2-normalised. Good enough for keyword-ish semantic
search, and it makes the whole pipeline testable without a network.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the "
    "this to was were will with what which who how why when where do does did can".split()
)


def _stem(token: str) -> str:
    """Very small plural stripper: 'servers' -> 'server' (but not 'class')."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class HashingEmbedder:
    def __init__(self, dim: int = 512, use_bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.use_bigrams = use_bigrams

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        feats: Counter[str] = Counter(tokens)
        if self.use_bigrams:
            feats.update(f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, count in self._features(text).items():
                h = zlib.crc32(feat.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self.dim] += sign * (1.0 + math.log(count))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out