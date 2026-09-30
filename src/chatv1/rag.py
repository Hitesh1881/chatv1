from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    source: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]: ...


def chunk_document(document: Document, *, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    if not document.text.strip():
        raise ValueError("document text must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = " ".join(document.text.split())
    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap
    while start < len(text):
        piece = text[start:start + chunk_size]
        if piece:
            chunks.append(Chunk(f"{document.document_id}:{index}", document.document_id, piece, document.source, dict(document.metadata)))
        index += 1
        start += step
    return chunks


def build_context(results: Iterable[RetrievedChunk], *, max_chunks: int = 5) -> str:
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    return "\n\n".join(f"[{item.chunk.source}]\n{item.chunk.text}" for item in list(results)[:max_chunks])
