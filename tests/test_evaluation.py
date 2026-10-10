from pathlib import Path

import pytest

from askdocs.cli import main
from askdocs.engine import Engine
from askdocs.evaluation import evaluate, load_cases

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "examples" / "eval.json"


def test_golden_set_meets_quality_bar():
    engine = Engine()
    engine.ingest_path(ROOT / "examples" / "docs")
    report = evaluate(engine, load_cases(CASES), k=3)
    assert report.hit_at_k == 1.0
    assert report.mrr >= 0.8


def test_misses_are_reported():
    engine = Engine()
    engine.ingest_text("a.md", "completely unrelated content about gardening")
    report = evaluate(engine, [{"question": "token expiry", "expected_source": "b.md"}])
    assert report.hit_at_k == 0.0
    assert report.misses == ["token expiry"]


def test_empty_cases_rejected():
    with pytest.raises(ValueError):
        evaluate(Engine(), [])


def test_cases_require_expected_fields(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('[{"question": "x"}]')
    with pytest.raises(ValueError):
        load_cases(bad)


def test_cli_eval(tmp_path, capsys):
    idx = str(tmp_path / "idx")
    main(["--index", idx, "ingest", str(ROOT / "examples" / "docs")])
    capsys.readouterr()
    assert main(["--index", idx, "eval", str(CASES)]) == 0
    assert "hit@3" in capsys.readouterr().out