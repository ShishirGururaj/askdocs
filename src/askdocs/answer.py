"""Turn retrieved chunks into an answer.

``ExtractiveAnswerer`` needs no LLM: it picks the sentences that best overlap with the
question. Any object with the same ``answer`` method can replace it (see ``llm.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from askdocs.embeddings import tokenize
from askdocs.store import Hit

NO_ANSWER = "I couldn't find anything relevant in the indexed documents."
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class Answerer(Protocol):
    def answer(self, question: str, hits: list[Hit]) -> str: ...


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


class ExtractiveAnswerer:
    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def answer(self, question: str, hits: list[Hit]) -> str:
        q_tokens = set(tokenize(question))
        scored: list[tuple[int, float, int, str]] = []
        order = 0
        for hit in hits:
            for sentence in _SENTENCE_SPLIT.split(hit.record.text):
                overlap = len(q_tokens & set(tokenize(sentence)))
                if overlap:
                    scored.append((overlap, overlap + hit.score, order, sentence.strip()))
                order += 1
        if not scored:
            return NO_ANSWER
        best = sorted(scored, key=lambda s: -s[0])[: self.max_sentences]
        return " ".join(s for _, _, s in sorted(best, key=lambda s: s[1]))
        # Drop sentences that match far fewer question terms than the best one.
        top_overlap = max(s[0] for s in scored)
        strong = [s for s in scored if s[0] > top_overlap / 2]
        best = sorted(strong, key=lambda s: -s[1])[: self.max_sentences]
        return " ".join(s[3] for s in sorted(best, key=lambda s: s[2]))