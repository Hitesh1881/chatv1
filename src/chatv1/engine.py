from dataclasses import dataclass, field

import torch

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

    def _generate(self, prompt: str, generation: GenerationConfig) -> str:
        ids = self.tokenizer.encode(prompt)
        # Reserve room for generation and keep the end of the prompt, where
        # the current User question and Assistant marker live.
        reserve = min(generation.max_new_tokens, self.model.cfg.block_size - 1)
        max_input = max(1, self.model.cfg.block_size - reserve)
        ids = ids[-max_input:]
        input_ids = torch.tensor(
            [ids],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )
        output = generate_text(self.model, input_ids, generation)
        generated = output[0, input_ids.size(1):].tolist()
        return self.tokenizer.decode(generated)

    def respond(self, request: ChatEngineRequest) -> ChatEngineResponse:
        if not request.conversation_id.strip() or not request.prompt.strip():
            raise ValueError("conversation_id and prompt are required")
        context, memory_count = self._context(request)
        prompt = (
            f"Context:\n{context}\n\nUser: {request.prompt}\nAssistant:"
            if context
            else f"User: {request.prompt}\nAssistant:"
        )
        text = self._generate(prompt, request.generation)

        # If context made the tiny bootstrap model produce only whitespace,
        # retry against the learned User/Assistant format without RAG context.
        # This is still model inference; it is not a canned answer.
        if not text.strip() and context:
            text = self._generate(
                f"User: {request.prompt}\nAssistant:",
                request.generation,
            )

        return ChatEngineResponse(text=text, context=context, memory_count=memory_count)
