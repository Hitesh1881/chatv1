import pytest

from chatv1.image import GeneratedImage, ImageGenerationRequest, generate_image


class DummyImageGenerator:
    def generate(self, request):
        return GeneratedImage(b"png", "image/png", request.seed)


def test_image_generation_contract():
    request = ImageGenerationRequest("a cinematic mountain")
    result = generate_image(DummyImageGenerator(), request)
    assert result.image_bytes == b"png"
    assert result.mime_type == "image/png"


@pytest.mark.parametrize("request", [
    ImageGenerationRequest(""),
    ImageGenerationRequest("x", width=0),
    ImageGenerationRequest("x", height=3000),
    ImageGenerationRequest("x", steps=0),
])
def test_image_request_validation(request):
    with pytest.raises(ValueError):
        request.validate()
