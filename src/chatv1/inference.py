from dataclasses import dataclass
from typing import Iterator, Protocol

import torch


class TextGenerator(Protocol):
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor: ...


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must not be negative")


@torch.no_grad()
def generate_text(
    model: TextGenerator,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    config.validate()
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must have shape [batch, sequence] and be non-empty")
    return model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
    )


def stream_token_ids(
    model: TextGenerator,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> Iterator[int]:
    """Small provider-agnostic streaming contract for incremental UI/API work."""
    output = generate_text(model, input_ids, config)
    prompt_len = input_ids.size(1)
    for token_id in output[0, prompt_len:].tolist():
        yield int(token_id)
