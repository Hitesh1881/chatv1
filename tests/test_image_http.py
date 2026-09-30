import base64
import json
import threading
from http.server import HTTPServer
from urllib.request import Request, urlopen

from chatv1.http import make_image_handler
from chatv1.image import GeneratedImage


class DummyGenerator:
    def generate(self, request):
        return GeneratedImage(b"png-bytes", "image/png", request.seed)


def test_image_http_end_to_end():
    server = HTTPServer(("127.0.0.1", 0), make_image_handler(lambda: DummyGenerator()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"prompt": "test image", "seed": 42}).encode()
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/images/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        assert response.status == 200
        assert payload["mime_type"] == "image/png"
        assert payload["seed"] == 42
        assert base64.b64decode(payload["image_base64"]) == b"png-bytes"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
