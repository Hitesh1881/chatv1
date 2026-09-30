import pytest
import torch

from chatv1.inference import GenerationConfig, generate_text, stream_token_ids


class DummyModel:
    def generate(self, input_ids, **kwargs):
        assert kwargs["max_new_tokens"] == 2
        return torch.tensor([[1, 2, 3, 4]])


def test_generation_contract():
    cfg = GenerationConfig(max_new_tokens=2)
    ids = torch.tensor([[1, 2]])
    assert generate_text(DummyModel(), ids, cfg).tolist() == [[1, 2, 3, 4]]
    assert list(stream_token_ids(DummyModel(), ids, cfg)) == [3, 4]


@pytest.mark.parametrize(
    "cfg",
    [GenerationConfig(max_new_tokens=0), GenerationConfig(temperature=0), GenerationConfig(top_k=-1)],
)
def test_generation_validation(cfg):
    with pytest.raises(ValueError):
        cfg.validate()
