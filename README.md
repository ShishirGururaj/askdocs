# askdocs
 
-A small, dependency-light RAG (retrieval-augmented generation) service in Python.
-Ingest your docs, ask questions, get answers with sources.
[![CI](https://github.com/ShishirGururaj/askdocs/actions/workflows/ci.yml/badge.svg)](https://github.com/ShishirGururaj/askdocs/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
 
-> Work in progress. See the commit history for the build-up, one step at a time.
A small, readable **RAG (retrieval-augmented generation) service** in Python. Point it at a folder
of docs, then ask questions over a CLI or an HTTP API and get answers with sources.

It runs **fully offline by default** (no API key, no model download) and can switch to Gemini
for generated answers with one environment variable.

```
            ┌──────────┐   ┌────────────┐   ┌─────────────┐
 docs ────▶ │ chunking │──▶│ embeddings │──▶│ vector store│──▶ saved to disk
            └──────────┘   └────────────┘   └──────┬──────┘
                                                   │ top-k cosine search
 question ─────────────────────────────────────────┤
                                                   ▼
                                   ┌────────────────────────────┐
                                   │ answerer                   │
                                   │  • extractive (offline)    │──▶ answer + sources
                                   │  • Gemini (optional)       │
                                   └────────────────────────────┘
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

askdocs ingest examples/docs
askdocs ask "How long do access tokens last?"
```

```
Dependencies are scanned weekly ... access tokens expire after one hour.

Sources:
  - examples/docs/security.md
```

## HTTP API

```bash
askdocs serve --port 8000        # interactive docs at http://localhost:8000/docs

curl -X POST localhost:8000/documents -H 'content-type: application/json' \
  -d '{"source": "faq.md", "text": "Refunds are processed within five business days."}'

curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "How fast are refunds processed?", "top_k": 4}'
```

| Method | Path         | Purpose                                      |
| ------ | ------------ | -------------------------------------------- |
| GET    | `/health`    | Liveness and number of indexed chunks        |
| POST   | `/documents` | Chunk, embed and index a text document       |
| POST   | `/query`     | Retrieve, answer and return sources + scores |

## Configuration

| Variable              | Default                | Meaning                                  |
| --------------------- | ---------------------- | ---------------------------------------- |
| `ASKDOCS_INDEX`       | `.askdocs`             | Where the index is stored                |
| `ASKDOCS_TOP_K`       | `4`                    | Chunks retrieved per question            |
| `ASKDOCS_LLM`         | `extractive`           | `extractive` or `gemini`                 |
| `GEMINI_API_KEY`      | unset                  | Required when `ASKDOCS_LLM=gemini`       |
| `ASKDOCS_GEMINI_MODEL`| `gemini-flash-latest`  | Any model that supports `generateContent`|

```bash
export ASKDOCS_LLM=gemini GEMINI_API_KEY=...
askdocs ask "Summarise our deployment process"
```

## Measuring retrieval quality

RAG systems regress silently, so there is a tiny evaluation harness:

```bash
askdocs ingest examples/docs
askdocs eval examples/eval.json     # prints hit@3 and MRR, exits 1 on any miss
```

Add your own `{question, expected_source}` pairs to track quality as you change chunk sizes,
embedders or ranking.

## Docker

```bash
docker build -t askdocs .
docker run -p 8000:8000 -v askdocs-data:/data askdocs
```

## Design notes

- **Offline-first embedder.** A hashing-trick embedder (signed token + bigram counts, L2-normalised)
  keeps the pipeline deterministic, testable and dependency-free. The trade-off is that it is lexical,
  not truly semantic. `Engine` only needs an object with `.dim` and `.embed(list[str])`, so swapping in
  a neural embedder is a small change.
- **Collision guard.** Hash collisions can give unrelated chunks a decent score, so `Engine.ask`
  also requires a shared term with the question (see the regression test in `tests/test_answer.py`).
- **Exact search.** Brute-force cosine similarity over a NumPy matrix is exact and fast up to
  roughly 10^5 chunks. Beyond that, put an ANN index behind `VectorStore`'s interface.
- **No SDK lock-in.** Gemini is called over REST with `httpx`, and tests use `httpx.MockTransport`,
  so the suite never touches the network.

## Development

```bash
ruff check .
pytest -q
```

## Roadmap

- [ ] Neural embedder behind the same interface
- [ ] Hybrid ranking (BM25 + vectors) and MMR de-duplication
- [ ] PDF and HTML ingestion
- [ ] Streaming answers
- [ ] Auth and per-collection indexes for the API