from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    seed: int | None = None

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width > 2048 or self.height > 2048:
            raise ValueError("width and height must be <= 2048")
        if self.steps <= 0 or self.steps > 100:
            raise ValueError("steps must be between 1 and 100")


@dataclass(frozen=True)
class GeneratedImage:
    image_bytes: bytes
    mime_type: str
    seed: int | None


class ImageGenerator(Protocol):
    def generate(self, request: ImageGenerationRequest) -> GeneratedImage: ...


def generate_image(generator: ImageGenerator, request: ImageGenerationRequest) -> GeneratedImage:
    """Provider/model-agnostic image generation boundary for local open models."""
    request.validate()
    result = generator.generate(request)
    if not result.image_bytes:
        raise ValueError("image generator returned empty image data")
    if not result.mime_type.startswith("image/"):
        raise ValueError("image generator returned a non-image MIME type")
    return result
