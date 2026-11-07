from ragbench.chunking import FixedSizeChunker, SentenceChunker, make_chunker
from ragbench.text import content_tokens, is_abstention, split_sentences, tokenize
from ragbench.types import Document


def test_tokenize_handles_thousands_separators():
    assert "1150" in tokenize("Spare batteries cost 1,150 US dollars")


def test_content_tokens_stems_plurals_and_drops_stopwords():
    assert content_tokens("The days of the policies") == ["day", "policy"]


def test_split_sentences_handles_bullets():
    sents = split_sentences("# Title\n\nFirst one. Second one.\n- Bullet: 3 items.")
    assert sents == ["Title", "First one.", "Second one.", "Bullet: 3 items."]


def test_abstention_detection():
    assert is_abstention("I don't know.")
    assert is_abstention("The context does not mention the top speed.")
    assert not is_abstention("The top speed is 60 km/h.")


def test_sentence_chunker_overlap():
    doc = Document("d", "A one. B two. C three. D four. E five.")
    chunks = SentenceChunker(size=2, overlap=1).chunk([doc])
    assert [c.text for c in chunks] == ["A one. B two.", "B two. C three.", "C three. D four.", "D four. E five."]
    assert chunks[0].id == "d#0" and chunks[0].doc_id == "d"


def test_fixed_chunker_covers_all_words():
    words = [f"w{i}" for i in range(25)]
    chunks = FixedSizeChunker(size=10, overlap=3).split(Document("d", " ".join(words)))
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)
    assert all(len(c.split()) <= 10 for c in chunks)


def test_make_chunker_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        make_chunker({"type": "nope"})
