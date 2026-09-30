import pytest

from chatv1.memory import InMemoryStore, MemoryItem


def test_memory_store_search():
    store = InMemoryStore()
    store.put(MemoryItem("1", "chat-1", "user likes Python", {}))
    store.put(MemoryItem("2", "chat-2", "user likes React", {}))
    assert [x.memory_id for x in store.search("chat-1", "Python")] == ["1"]


def test_memory_validation():
    store = InMemoryStore()
    with pytest.raises(ValueError):
        store.search("chat-1", "")
