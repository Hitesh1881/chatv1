import pytest
import torch
from torch import nn

from chatv1.engine import ChatEngine, ChatEngineRequest
from chatv1.inference import GenerationConfig
from chatv1.memory import InMemoryStore, MemoryItem
from chatv1.tokenizer import CharTokenizer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, **kwargs):
        return input_ids


def test_engine_without_context():
    tokenizer = CharTokenizer("User: Assistant: hello\\n")
    engine = ChatEngine(model=DummyModel(), tokenizer=tokenizer)
    response = engine.respond(ChatEngineRequest("c1", "hello", GenerationConfig(max_new_tokens=1)))
    assert response.memory_count == 0


def test_engine_uses_memory():
    tokenizer = CharTokenizer("User: Assistant: hello python\\n[memory]")
    memory = InMemoryStore()
    memory.put(MemoryItem("1", "c1", "python"))
    engine = ChatEngine(model=DummyModel(), tokenizer=tokenizer, memory=memory)
    response = engine.respond(ChatEngineRequest("c1", "python", GenerationConfig(max_new_tokens=1)))
    assert response.memory_count == 1
    assert "python" in response.context


def test_engine_requires_prompt():
    tokenizer = CharTokenizer("User: Assistant:")
    engine = ChatEngine(model=DummyModel(), tokenizer=tokenizer)
    with pytest.raises(ValueError):
        engine.respond(ChatEngineRequest("c1", ""))
