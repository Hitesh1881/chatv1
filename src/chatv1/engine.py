from dataclasses import dataclass
from typing import Callable

from .inference import GenerationConfig, generate_text
from .memory import MemoryStore
from .rag import RetrievedChunk, build_context


@dataclass(frozen=True)
class ChatEngineRequest:
    conversation_id: str
    prompt: str
    generation: GenerationConfig = GenerationConfig()
    retrieve_top_k: int = 0


@dataclass(frozen=True)
class ChatEngineResponse:
    text: str
    context: str
    memory_count: int


class ChatEngine:
    """Orchestrates memory/retrieval/context and a replaceable text model."""

    def __init__(self, *, model, tokenizer, memory: MemoryStore | None = None, retriever=None):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.retriever = retriever

    def _context(self, request: ChatEngineRequest) -> tuple[str, int]:
        parts: list[str] = []
        memory_count = 0
        if self.memory is not None:
            memories = self.memory.search(request.conversation_id, request.prompt)
            memory_count = len(memories)
            parts.extend(f"[memory] {item.text}" for item in memories)
        if self.retriever is not None and request.retrieve_top_k:
            results: list[RetrievedChunk] = self.retriever.retrieve(request.prompt, top_k=request.retrieve_top_k)
            parts.append(build_context(results, max_chunks=request.retrieve_top_k))
        return "\n\n".join(parts), memory_count

    def respond(self, request: ChatEngineRequest) -> ChatEngineResponse:
        if not request.conversation_id.strip() or not request.prompt.strip():
            raise ValueError("conversation_id and prompt are required")
        context, memory_count = self._context(request)
        prompt = f"Context:\n{context}\n\nUser: {request.prompt}\nAssistant:" if context else f"User: {request.prompt}\nAssistant:"
        ids = self.tokenizer.encode(prompt)
        import torch
        input_ids = torch.tensor([ids], dtype=torch.long, device=next(self.model.parameters()).device)
        output = generate_text(self.model, input_ids, request.generation)
        text = self.tokenizer.decode(output[0].tolist()[len(ids):])
        return ChatEngineResponse(text=text, context=context, memory_count=memory_count)
