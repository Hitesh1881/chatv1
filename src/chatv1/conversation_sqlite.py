import sqlite3
from pathlib import Path

from .conversation import Conversation, ConversationStore


class SQLiteConversationStore(ConversationStore):
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS messages (conversation_id TEXT NOT NULL, seq INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT NOT NULL, content TEXT NOT NULL)")

    def get(self, conversation_id: str) -> Conversation:
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        with self._connect() as db:
            rows = db.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY seq", (conversation_id,)).fetchall()
        conversation = Conversation(conversation_id)
        for role, content in rows:
            conversation.add(role, content)
        return conversation

    def save(self, conversation: Conversation) -> None:
        if not conversation.conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        with self._connect() as db:
            db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation.conversation_id,))
            db.executemany("INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)", [(conversation.conversation_id, m.role, m.content) for m in conversation.messages])
