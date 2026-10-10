"""Measure retrieval quality against a small golden set of questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from askdocs.engine import Engine


@dataclass(frozen=True)
class EvalReport:
    total: int
    hit_at_k: float
    mrr: float
    misses: list[str]


def load_cases(path: str | Path) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    for case in cases:
        if "question" not in case or "expected_source" not in case:
            raise ValueError("each case needs 'question' and 'expected_source'")
    return cases


def evaluate(engine: Engine, cases: list[dict], k: int = 3) -> EvalReport:
    """Hit@k: share of questions whose expected source is in the top k.

    MRR: mean of 1/rank of the first correct source (0 when it is not retrieved).
    """
    if not cases:
        raise ValueError("no evaluation cases given")
    hits = 0
    reciprocal_ranks = 0.0
    misses: list[str] = []
    for case in cases:
        retrieved = [h.record.source for h in engine.search(case["question"], k)]
        rank = next((i for i, s in enumerate(retrieved, 1) if case["expected_source"] in s), None)
        if rank:
            hits += 1
            reciprocal_ranks += 1 / rank
        else:
            misses.append(case["question"])
    return EvalReport(
        total=len(cases),
        hit_at_k=hits / len(cases),
        mrr=reciprocal_ranks / len(cases),
        misses=misses,
    )