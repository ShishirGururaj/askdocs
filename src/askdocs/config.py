"""Runtime settings read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    index_dir: str = ".askdocs"
    top_k: int = 4
    llm: str = "extractive"  # "extractive" (offline) or "gemini"
    gemini_model: str = "gemini-flash-latest"
    gemini_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            index_dir=env.get("ASKDOCS_INDEX", ".askdocs"),
            top_k=int(env.get("ASKDOCS_TOP_K", "4")),
            llm=env.get("ASKDOCS_LLM", "extractive").lower(),
            gemini_model=env.get("ASKDOCS_GEMINI_MODEL", "gemini-flash-latest"),
            gemini_api_key=env.get("GEMINI_API_KEY"),
        )