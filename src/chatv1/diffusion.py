from __future__ import annotations

import io
from dataclasses import dataclass

import torch

from .image import GeneratedImage, ImageGenerationRequest


@dataclass(frozen=True)
class DiffusionConfig:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    device: str | None = None
    torch_dtype: torch.dtype = torch.float16


class LocalDiffusionGenerator:
    """Lazy-loaded local Diffusers backend for Apple Silicon/CUDA/CPU."""

    def __init__(self, config: DiffusionConfig | None = None) -> None:
        self.config = config or DiffusionConfig()
        self._pipeline = None

    def _device(self) -> str:
        if self.config.device:
            return self.config.device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError("Install image dependencies with: pip install -e '.[image]'") from exc

        device = self._device()
        dtype = self.config.torch_dtype if device != "cpu" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        pipe = pipe.to(device)
        if device == "mps":
            pipe.enable_attention_slicing()
        self._pipeline = pipe
        return pipe

    def generate(self, request: ImageGenerationRequest) -> GeneratedImage:
        request.validate()
        pipe = self._load()
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self._device()).manual_seed(request.seed)
        with torch.inference_mode():
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                generator=generator,
            )
        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return GeneratedImage(buffer.getvalue(), "image/png", request.seed)
