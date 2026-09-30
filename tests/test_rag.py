import pytest

from chatv1.rag import Chunk, Document, RetrievedChunk, build_context, chunk_document


def test_chunk_document_and_context():
    doc = Document("doc1", "alpha beta gamma delta", "notes.txt", {})
    chunks = chunk_document(doc, chunk_size=10, overlap=2)
    assert chunks
    context = build_context([RetrievedChunk(chunks[0], 1.0)])
    assert "[notes.txt]" in context
    assert "alpha" in context


def test_rag_validation():
    with pytest.raises(ValueError):
        chunk_document(Document("d", "", "x", {}))
    with pytest.raises(ValueError):
        chunk_document(Document("d", "abc", "x", {}), chunk_size=2, overlap=2)
    with pytest.raises(ValueError):
        build_context([], max_chunks=0)
