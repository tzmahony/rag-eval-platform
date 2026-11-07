import pytest

from ragbench.chunking import SentenceChunker
from ragbench.data import load_corpus
from ragbench.retrieval import make_retriever
from ragbench.types import Chunk, Document


@pytest.fixture
def chunks():
    docs = [
        Document("pets", "Dogs are loyal pets. Cats are independent pets."),
        Document("cars", "Electric cars use batteries. Diesel cars use fuel."),
        Document("food", "Pasta is made from wheat. Rice is a grain."),
    ]
    return SentenceChunker(size=1, overlap=0).chunk(docs)


@pytest.mark.parametrize(
    "cfg",
    [
        {"type": "bm25"},
        {"type": "tfidf"},
        {"type": "lsa", "dim": 4},
        {"type": "hybrid", "retrievers": [{"type": "bm25"}, {"type": "tfidf"}]},
    ],
)
def test_retrievers_rank_obvious_match_first(chunks, cfg):
    r = make_retriever(cfg)
    r.index(chunks)
    hits = r.retrieve("which cars use batteries", k=3)
    assert hits[0].chunk.id == "cars#0"
    assert [h.rank for h in hits] == [1, 2, 3]


def test_retriever_requires_index():
    with pytest.raises(RuntimeError):
        make_retriever({"type": "bm25"}).retrieve("q", 1)


def test_bm25_on_example_corpus(examples_dir):
    docs = load_corpus(examples_dir / "corpus")
    chunks = SentenceChunker(3, 1).chunk(docs)
    r = make_retriever({"type": "bm25"})
    r.index(chunks)
    top = r.retrieve("What is the tank capacity of the K9 Applicator?", k=3)
    assert top[0].chunk.doc_id == "product-k9-applicator"


def test_hybrid_validates_weights():
    from ragbench.retrieval import BM25Retriever, HybridRetriever

    with pytest.raises(ValueError):
        HybridRetriever([BM25Retriever()], weights=[1.0, 2.0])


def test_index_rejects_empty():
    with pytest.raises(ValueError):
        make_retriever({"type": "bm25"}).index([])


def test_chunk_type_roundtrip():
    c = Chunk("a#0", "a", "text", 0)
    assert c.doc_id == "a"
