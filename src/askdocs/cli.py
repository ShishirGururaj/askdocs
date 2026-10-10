"""Command-line interface: ``askdocs ingest|ask|eval|serve``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from askdocs import __version__
from askdocs.config import Settings
from askdocs.engine import Engine
from askdocs.evaluation import evaluate, load_cases
from askdocs.llm import AnswerError, build_answerer


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
    ev = sub.add_parser("eval", help="measure retrieval quality on a golden question set")
    ev.add_argument("cases", help="JSON file: [{question, expected_source}, ...]")
    ev.add_argument("-k", type=int, default=3)
    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    args = build_parser(settings).parse_args(argv)

    if args.command == "eval":
        if not (Path(args.index) / "records.json").exists():
            print(f"error: no index at {args.index}", file=sys.stderr)
            return 2
        report = evaluate(Engine.load(args.index), load_cases(args.cases), k=args.k)
        print(f"questions: {report.total}")
        print(f"hit@{args.k}:    {report.hit_at_k:.2f}")
        print(f"MRR:      {report.mrr:.2f}")
        for question in report.misses:
            print(f"  miss: {question}")
        return 0 if not report.misses else 1

    if args.command == "serve":
        import uvicorn

        from askdocs.api import create_app_from_env

        uvicorn.run(create_app_from_env(), host=args.host, port=args.port)
        return 0

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
    try:    
        engine = Engine.load(args.index, answerer=build_answerer(settings))
        result = engine.ask(args.question, k=args.k)
    except (AnswerError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.text)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"  - {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())