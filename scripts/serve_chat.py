from pathlib import Path
from http.server import HTTPServer

import torch

from chatv1.chat_http import make_chat_handler
from chatv1.chat_service import ChatService
from chatv1.conversation_sqlite import SQLiteConversationStore
from chatv1.device import get_device
from chatv1.engine import ChatEngine
from chatv1.hf_model import HFChatModel
from chatv1.memory import InMemoryStore
from chatv1.rag import Document, chunk_document
from chatv1.retrieval import LexicalRetriever


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    device = get_device()
    model, tokenizer = HFChatModel.load(device=device)

    memory = InMemoryStore()
    docs = [
        Document(
            "demo",
            "ChatV1 is a local AI research system with retrieval, memory, a modular inference layer, and local image generation.",
            "demo-doc",
            {},
        )
    ]
    chunks = tuple(chunk for doc in docs for chunk in chunk_document(doc))
    retriever = LexicalRetriever(chunks)

    engine = ChatEngine(
        model=model,
        tokenizer=tokenizer,
        memory=memory,
        retriever=retriever,
    )
    service = ChatService(
        engine,
        SQLiteConversationStore(root / "artifacts" / "chatv1.sqlite3"),
        retrieve_top_k=3,
    )
    server = HTTPServer(("127.0.0.1", 8001), make_chat_handler(lambda: service))
    print(f"Chat API: http://127.0.0.1:8001/v1/chat/completions")
    print(f"Device: {device} | Model: Qwen/Qwen2.5-0.5B-Instruct | RAG: on | Memory: on")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
