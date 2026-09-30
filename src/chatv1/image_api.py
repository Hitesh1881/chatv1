from dataclasses import dataclass

from .image import GeneratedImage, ImageGenerationRequest, ImageGenerator, generate_image


@dataclass(frozen=True)
class ImageResponse:
    image: GeneratedImage


def generate_image_response(generator: ImageGenerator, request: ImageGenerationRequest) -> ImageResponse:
    """Application service boundary for HTTP/UI adapters."""
    return ImageResponse(image=generate_image(generator, request))
