from dataclasses import dataclass, field

from .inference import GenerationConfig, generate_text
from .memory import MemoryStore
from .rag import RetrievedChunk, build_context


@dataclass(frozen=True)
class ChatEngineRequest:
    conversation_id: str
    prompt: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
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
        # Some causal models truncate long prompts to their context window before
        # generating. Slice from the actual returned sequence length rather than
        # the original prompt length, otherwise RAG can make valid generations
        # look empty when len(ids) > the model's block size.
        generated_start = min(len(ids), output.size(1))
        text = self.tokenizer.decode(output[0, generated_start:].tolist())
        return ChatEngineResponse(text=text, context=context, memory_count=memory_count)
