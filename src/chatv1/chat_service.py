from dataclasses import dataclass

from .conversation import ConversationStore
from .engine import ChatEngine, ChatEngineRequest
from .inference import GenerationConfig


@dataclass(frozen=True)
class ChatServiceResponse:
    conversation_id: str
    text: str


class ChatService:
    def __init__(self, engine: ChatEngine, conversations: ConversationStore, *, retrieve_top_k: int = 0):
        if retrieve_top_k < 0:
            raise ValueError("retrieve_top_k must not be negative")
        self.engine = engine
        self.conversations = conversations
        self.retrieve_top_k = retrieve_top_k

    def reply(self, conversation_id: str, prompt: str, *, generation: GenerationConfig | None = None) -> ChatServiceResponse:
        conversation = self.conversations.get(conversation_id)
        conversation.add("user", prompt)
        request = ChatEngineRequest(conversation_id, prompt, generation or GenerationConfig(), self.retrieve_top_k)
        result = self.engine.respond(request)
        conversation.add("assistant", result.text)
        self.conversations.save(conversation)
        return ChatServiceResponse(conversation_id, result.text)
