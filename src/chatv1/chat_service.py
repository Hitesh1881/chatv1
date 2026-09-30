from dataclasses import dataclass

from .conversation import ConversationStore
from .engine import ChatEngine, ChatEngineRequest


@dataclass(frozen=True)
class ChatServiceResponse:
    conversation_id: str
    text: str


class ChatService:
    def __init__(self, engine: ChatEngine, conversations: ConversationStore):
        self.engine = engine
        self.conversations = conversations

    def reply(self, conversation_id: str, prompt: str) -> ChatServiceResponse:
        conversation = self.conversations.get(conversation_id)
        conversation.add("user", prompt)
        result = self.engine.respond(ChatEngineRequest(conversation_id, prompt))
        conversation.add("assistant", result.text)
        self.conversations.save(conversation)
        return ChatServiceResponse(conversation_id, result.text)
