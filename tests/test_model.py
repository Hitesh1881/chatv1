import torch
from chatv1.model import ChatV1, ModelConfig


def test_forward_and_loss():
    cfg = ModelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=4, n_embd=32)
    model = ChatV1(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_generation_shape():
    cfg = ModelConfig(vocab_size=32, block_size=16, n_layer=2, n_head=4, n_embd=32)
    model = ChatV1(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(x, max_new_tokens=5)
    assert out.shape == (1, 9)
