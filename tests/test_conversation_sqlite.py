from chatv1.conversation import Conversation
from chatv1.conversation_sqlite import SQLiteConversationStore


def test_sqlite_store_round_trip(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chat.db")
    conversation = Conversation("demo")
    conversation.add("user", "hello")
    conversation.add("assistant", "hi")
    store.save(conversation)

    restored = store.get("demo")
    assert [(m.role, m.content) for m in restored.messages] == [("user", "hello"), ("assistant", "hi")]
