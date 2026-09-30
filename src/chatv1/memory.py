from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    conversation_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class MemoryStore(Protocol):
    def put(self, item: MemoryItem) -> None: ...
    def search(self, conversation_id: str, query: str, *, top_k: int = 5) -> list[MemoryItem]: ...


class InMemoryStore:
    """Deterministic baseline memory store; replaceable by a vector/DB backend."""
    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def put(self, item: MemoryItem) -> None:
        if not item.memory_id or not item.conversation_id or not item.text.strip():
            raise ValueError("memory_id, conversation_id and text are required")
        self._items[item.memory_id] = item

    def search(self, conversation_id: str, query: str, *, top_k: int = 5) -> list[MemoryItem]:
        if not conversation_id:
            raise ValueError("conversation_id is required")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = set(query.lower().split())
        scored = []
        for item in self._items.values():
            if item.conversation_id != conversation_id:
                continue
            score = sum(term in item.text.lower().split() for term in terms)
            if score:
                scored.append((score, item.memory_id, item))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [item for _, _, item in scored[:top_k]]
