from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class DatasetSplit:
    train: torch.Tensor
    validation: torch.Tensor


def encode_text(tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise ValueError("encoded text must contain at least two tokens")
    return torch.tensor(ids, dtype=torch.long)


def split_data(data: torch.Tensor, validation_fraction: float = 0.1) -> DatasetSplit:
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional")
    if len(data) < 20:
        raise ValueError("data is too small to split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    cut = int(len(data) * (1 - validation_fraction))
    if cut < 2 or len(data) - cut < 2:
        raise ValueError("split would create an unusable partition")
    return DatasetSplit(data[:cut], data[cut:])


def sample_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional")
    if block_size <= 0 or batch_size <= 0:
        raise ValueError("block_size and batch_size must be positive")
    if len(data) <= block_size:
        raise ValueError("data must be longer than block_size")
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts]).to(device)
    return x, y
