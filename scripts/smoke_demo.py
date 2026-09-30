from pathlib import Path

from chatv1.health import check_health
from chatv1.retrieval import LexicalRetriever
from chatv1.rag import Document, chunk_document


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = (Document("demo", "ChatV1 supports RAG, memory, streaming, and local image generation.", "demo", {}),)
    chunks = tuple(c for d in docs for c in chunk_document(d))
    results = LexicalRetriever(chunks).retrieve("RAG streaming", top_k=2)
    report = check_health(str(root / "artifacts" / "tiny_chatv1.pt"))
    print(f"RAG smoke: {'OK' if results else 'FAILED'}")
    print(f"Checkpoint: {report.status}")
    if not results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
