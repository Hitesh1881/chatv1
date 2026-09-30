import pytest
import torch

from chatv1.data import encode_text, sample_batch, split_data
from chatv1.tokenizer import CharTokenizer


def test_encode_and_split():
    tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
    data = encode_text(tokenizer, "abcdefghijklmnopqrst")
    split = split_data(data, 0.2)
    assert len(split.train) + len(split.validation) == len(data)


def test_sample_batch_shapes():
    data = torch.arange(40)
    x, y = sample_batch(data, block_size=8, batch_size=3, device=torch.device("cpu"))
    assert x.shape == y.shape == (3, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_invalid_split():
    with pytest.raises(ValueError):
        split_data(torch.arange(10), 0)


def test_small_data_fails():
    with pytest.raises(ValueError):
        split_data(torch.arange(10))
