from pathlib import Path

from askdocs.cli import main

DOCS = str(Path(__file__).resolve().parent.parent / "examples" / "docs")


def test_ingest_then_ask(tmp_path, capsys):
    idx = str(tmp_path / "idx")
    assert main(["--index", idx, "ingest", DOCS]) == 0
    assert "Indexed" in capsys.readouterr().out

    assert main(["--index", idx, "ask", "How often are dependencies scanned?"]) == 0
    out = capsys.readouterr().out
    assert "weekly" in out
    assert "security.md" in out


def test_ask_without_index_fails_cleanly(tmp_path, capsys):
    assert main(["--index", str(tmp_path / "nope"), "ask", "hi"]) == 2
    assert "ingest" in capsys.readouterr().err


def test_ingest_missing_path(tmp_path, capsys):
    assert main(["--index", str(tmp_path), "ingest", str(tmp_path / "missing")]) == 2

+def test_serve_starts_uvicorn(monkeypatch, tmp_path):
    import uvicorn

    calls = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.update(host=host, port=port))
    monkeypatch.setenv("ASKDOCS_INDEX", str(tmp_path / "idx"))
    assert main(["serve", "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert calls == {"host": "0.0.0.0", "port": 9000}