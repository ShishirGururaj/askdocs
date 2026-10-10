from pathlib import Path

from askdocs.engine import Engine

DOCS = Path(__file__).resolve().parent.parent / "examples" / "docs"


def test_ingest_text_counts_chunks():
    engine = Engine(chunk_size=5, overlap=1)
    text = " ".join(f"w{i}" for i in range(12))
    assert engine.ingest_text("a.md", text) == 3
    assert len(engine.store) == 3


def test_ingest_blank_text_adds_nothing():
    assert Engine().ingest_text("blank.md", "  ") == 0


def test_ingest_directory_filters_by_suffix(tmp_path):
    (tmp_path / "a.md").write_text("hello docs here")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("nested file content")
    files, chunks = Engine().ingest_path(tmp_path)
    assert (files, chunks) == (2, 2)


def test_search_finds_the_right_document():
    engine = Engine()
    engine.ingest_path(DOCS)
    hit = engine.search("how long do access tokens last", k=1)[0]
    assert hit.record.source.endswith("security.md")


def test_save_and_load(tmp_path):
    engine = Engine()
    engine.ingest_path(DOCS)
    engine.save(tmp_path)
    loaded = Engine.load(tmp_path)
    q = "rolling deployments replicas"
    assert loaded.search(q, 1)[0].record.id == engine.search(q, 1)[0].record.id


def test_markdown_headings_become_their_own_sentences():
    engine = Engine()
    engine.ingest_text("a.md", "# Title here\nBody text follows.")
    assert engine.store.records[0].text == "Title here. Body text follows."