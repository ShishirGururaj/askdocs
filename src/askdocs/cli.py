"""Command-line interface: ``askdocs ingest|ask``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from askdocs import __version__
from askdocs.config import Settings
from askdocs.engine import Engine


def _load_or_new(index_dir: str) -> Engine:
    return Engine.load(index_dir) if (Path(index_dir) / "records.json").exists() else Engine()


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="askdocs", description="Ask questions of your docs.")
    parser.add_argument("--version", action="version", version=f"askdocs {__version__}")
    parser.add_argument("--index", default=settings.index_dir, help="index directory")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index a file or directory of .md/.txt/.rst")
    ingest.add_argument("path")

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=settings.top_k, help="chunks to retrieve")
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    args = build_parser(settings).parse_args(argv)

    if args.command == "ingest":
        path = Path(args.path)
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
        engine = _load_or_new(args.index)
        files, chunks = engine.ingest_path(path)
        engine.save(args.index)
        print(f"Indexed {chunks} chunks from {files} files into {args.index}")
        return 0

    if not (Path(args.index) / "records.json").exists():
        msg = f"error: no index at {args.index}; run `askdocs ingest <path>` first"
        print(msg, file=sys.stderr)
        return 2
    result = Engine.load(args.index).ask(args.question, k=args.k)
    print(result.text)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"  - {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())