import json
from http.server import BaseHTTPRequestHandler
from typing import Callable

from .image import ImageGenerationRequest
from .image_api import generate_image_response


def make_image_handler(generator_factory: Callable[[], object]):
    class ImageHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/images/generate":
                self._send_json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                request = ImageGenerationRequest(
                    prompt=str(payload.get("prompt", "")),
                    negative_prompt=str(payload.get("negative_prompt", "")),
                    width=int(payload.get("width", 512)),
                    height=int(payload.get("height", 512)),
                    steps=int(payload.get("steps", 20)),
                    seed=payload.get("seed"),
                )
                response = generate_image_response(generator_factory(), request)
                self._send_json(200, {
                    "mime_type": response.image.mime_type,
                    "seed": response.image.seed,
                    "image_base64": __import__("base64").b64encode(response.image.image_bytes).decode("ascii"),
                })
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": "invalid_request", "message": str(exc)})
            except Exception as exc:  # boundary must not crash the server
                self._send_json(500, {"error": "generation_failed", "message": str(exc)})

    return ImageHandler
