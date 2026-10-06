import pytest

from askdocs.chunking import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n ") == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("one two three", size=10, overlap=2) == ["one two three"]


def test_windows_overlap():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_text(text, size=4, overlap=1)
    assert chunks[0] == "0 1 2 3"
    assert chunks[1] == "3 4 5 6"
    assert chunks[-1].endswith("9")


def test_every_word_is_covered():
    words = [f"w{i}" for i in range(57)]
    chunks = chunk_text(" ".join(words), size=10, overlap=3)
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (5, 5), (5, -1)])
def test_invalid_arguments(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", size=size, overlap=overlap)