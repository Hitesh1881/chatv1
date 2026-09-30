import pytest

from chatv1.conversation import InMemoryConversationStore


def test_conversation_persists_messages():
    store = InMemoryConversationStore()
    conversation = store.get("c1")
    conversation.add("user", "hello")
    conversation.add("assistant", "hi")
    store.save(conversation)
    assert [m.role for m in store.get("c1").messages] == ["user", "assistant"]


def test_invalid_message():
    with pytest.raises(ValueError):
        store = InMemoryConversationStore()
        store.get("c1").add("unknown", "x")
