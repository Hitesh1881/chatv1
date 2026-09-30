import pytest

from chatv1.tokenizer import CharTokenizer


def test_round_trip():
    tokenizer = CharTokenizer("abc abc")
    text = "cab"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unknown_character_fails():
    tokenizer = CharTokenizer("abc")
    with pytest.raises(ValueError, match="unknown characters"):
        tokenizer.encode("abd")


def test_state_round_trip():
    tokenizer = CharTokenizer("hello world")
    restored = CharTokenizer.from_state_dict(tokenizer.state_dict())
    assert restored.encode("hello") == tokenizer.encode("hello")
    assert restored.decode(tokenizer.encode("world")) == "world"


def test_empty_training_text_fails():
    with pytest.raises(ValueError, match="must not be empty"):
        CharTokenizer("")
