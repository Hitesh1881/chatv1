from chatv1.image import GeneratedImage, ImageGenerationRequest
from chatv1.image_api import generate_image_response


class DummyGenerator:
    def generate(self, request):
        return GeneratedImage(b"data", "image/png", request.seed)


def test_generate_image_response():
    response = generate_image_response(DummyGenerator(), ImageGenerationRequest("hello", seed=7))
    assert response.image.image_bytes == b"data"
    assert response.image.seed == 7
