from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass
class Conversation:
    conversation_id: str
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role")
        if not content.strip():
            raise ValueError("message content must not be empty")
        self.messages.append(Message(role, content))


class ConversationStore(Protocol):
    def get(self, conversation_id: str) -> Conversation: ...
    def save(self, conversation: Conversation) -> None: ...


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._items: dict[str, Conversation] = {}

    def get(self, conversation_id: str) -> Conversation:
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        item = self._items.get(conversation_id)
        if item is None:
            item = Conversation(conversation_id)
            self._items[conversation_id] = item
        return item

    def save(self, conversation: Conversation) -> None:
        if not conversation.conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        self._items[conversation.conversation_id] = conversation
