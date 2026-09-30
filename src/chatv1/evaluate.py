from dataclasses import dataclass
import math

import torch

from .data import sample_batch


@dataclass(frozen=True)
class EvalResult:
    loss: float
    perplexity: float


@torch.no_grad()
def evaluate(model, data: torch.Tensor, block_size: int, batches: int, batch_size: int, device) -> EvalResult:
    if batches <= 0:
        raise ValueError("batches must be positive")
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = sample_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    loss = sum(losses) / len(losses)
    return EvalResult(loss=loss, perplexity=math.exp(min(loss, 50.0)))
