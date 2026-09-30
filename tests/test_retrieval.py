from chatv1.rag import Document, chunk_document
from chatv1.retrieval import LexicalRetriever


def test_lexical_retriever_ranks_matching_chunks():
    docs = [
        Document("a", "Python API authentication tokens", "a", {}),
        Document("b", "React components and hooks", "b", {}),
    ]
    chunks = tuple(c for d in docs for c in chunk_document(d, chunk_size=100))
    results = LexicalRetriever(chunks).retrieve("Python authentication", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.document_id == "a"


def test_lexical_retriever_rejects_invalid_inputs():
    retriever = LexicalRetriever(())
    try:
        retriever.retrieve("", top_k=1)
        assert False
    except ValueError:
        pass
