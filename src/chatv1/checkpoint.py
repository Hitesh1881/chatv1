from pathlib import Path

import torch

from .model import ChatV1, ModelConfig
from .tokenizer import CharTokenizer


def save_checkpoint(path: Path, model, optimizer, tokenizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.cfg.__dict__,
            "tokenizer": tokenizer.state_dict(),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
        },
        path,
    )


def load_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ChatV1(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = CharTokenizer.from_state_dict(checkpoint["tokenizer"])
    return model, tokenizer, checkpoint
