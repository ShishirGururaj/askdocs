from pathlib import Path

from askdocs.answer import NO_ANSWER, ExtractiveAnswerer
from askdocs.engine import Engine
from askdocs.store import Hit, Record

DOCS = Path(__file__).resolve().parent.parent / "examples" / "docs"


def make_engine(**kwargs):
    engine = Engine(**kwargs)
    engine.ingest_path(DOCS)
    return engine


def test_answer_cites_the_right_source():
    result = make_engine().ask("How long until access tokens expire?")
    assert "one hour" in result.text
    assert result.sources[0].endswith("security.md")


def test_unrelated_question_gets_graceful_fallback():
    # This query collides with a chunk in the hash space; the lexical guard must reject it.
    result = make_engine().ask("zebra xylophone quantum")
    assert result.text == NO_ANSWER
    assert result.sources == []


def test_extractive_answerer_returns_best_sentences_in_document_order():
    hit = Hit(Record(0, "x.md", "Gamma delta. Zeta eta. Alpha delta."), 0.9)
    text = ExtractiveAnswerer(max_sentences=2).answer("alpha gamma delta", [hit])
    assert text == "Gamma delta. Alpha delta."


def test_custom_answerer_is_used():
    class Shout:
        def answer(self, question, hits):
            return "HELLO"

    result = make_engine(answerer=Shout()).ask("deploy docker replicas")
    assert result.text == "HELLO"


def test_weakly_matching_sentences_are_dropped():
    hit = Hit(Record(0, "x.md", "Tokens expire after one hour. Lunch is at noon every hour."), 0.5)
    text = ExtractiveAnswerer().answer("when do tokens expire after an hour", [hit])
    assert text == "Tokens expire after one hour."