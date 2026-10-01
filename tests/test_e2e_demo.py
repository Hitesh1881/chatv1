import json
import threading
from http.server import HTTPServer
from urllib.request import Request, urlopen

import torch
from torch import nn

from chatv1.chat_http import make_chat_handler
from chatv1.chat_service import ChatService
from chatv1.conversation_sqlite import SQLiteConversationStore
from chatv1.engine import ChatEngine
from chatv1.memory import InMemoryStore, MemoryItem
from chatv1.rag import Document, chunk_document
from chatv1.retrieval import LexicalRetriever
from chatv1.tokenizer import CharTokenizer


class EchoModel(nn.Module):
    def __init__(self, token_id: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.token_id = token_id

    def generate(self, input_ids, **kwargs):
        extra = torch.tensor([[self.token_id]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)

    def generate_stream(self, input_ids, **kwargs):
        yield self.token_id


def test_chat_demo_end_to_end(tmp_path):
    corpus = "".join(chr(i) for i in range(32, 127)) + (
        "Context User Assistant hello python RAG memory streaming demo "
        "ChatV1 is an original research AI system. "
        "Python authentication and RAG are supported."
    )
    tokenizer = CharTokenizer(corpus)
    dot_id = tokenizer.encode(".")[0]
    model = EchoModel(dot_id)
    memory = InMemoryStore()
    memory.put(MemoryItem("m1", "demo", "python"))
    docs = (
        Document("d1", "Python authentication and RAG are supported.", "docs", {}),
    )
    chunks = tuple(c for doc in docs for c in chunk_document(doc, chunk_size=200, overlap=20))
    retriever = LexicalRetriever(chunks)
    engine = ChatEngine(model=model, tokenizer=tokenizer, memory=memory, retriever=retriever)
    service = ChatService(
        engine,
        SQLiteConversationStore(tmp_path / "chat.sqlite3"),
        retrieve_top_k=1,
    )

    server = HTTPServer(("127.0.0.1", 0), make_chat_handler(lambda: service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps(
            {"conversation_id": "demo", "prompt": "python", "stream": False}
        ).encode()
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        assert response.status == 200
        assert payload["conversation_id"] == "demo"
        assert payload["text"] == "."

        persisted = service.conversations.get("demo")
        assert [(m.role, m.content) for m in persisted.messages] == [
            ("user", "python"),
            ("assistant", "."),
        ]

        body = json.dumps(
            {"conversation_id": "demo", "prompt": "RAG", "stream": True}
        ).encode()
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions/stream",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            stream = response.read().decode()
        assert "data: " in stream
        assert '"token": "."' in stream
        assert "data: [DONE]" in stream
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
