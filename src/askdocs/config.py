"""Runtime settings read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    index_dir: str = ".askdocs"
    top_k: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            index_dir=os.environ.get("ASKDOCS_INDEX", cls.index_dir),
            top_k=int(os.environ.get("ASKDOCS_TOP_K", cls.top_k)),
        )