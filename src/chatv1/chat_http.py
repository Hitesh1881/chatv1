import json
from http.server import BaseHTTPRequestHandler
from typing import Callable

from .chat_service import ChatService


def make_chat_handler(service_factory: Callable[[], ChatService]):
    class ChatHandler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
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
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                conversation_id = str(payload.get("conversation_id", "default"))
                prompt = str(payload.get("prompt", ""))
                result = service_factory().reply(conversation_id, prompt)
                self._json(200, {"conversation_id": result.conversation_id, "text": result.text})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": "invalid_request", "message": str(exc)})
            except Exception as exc:
                self._json(500, {"error": "chat_failed", "message": str(exc)})

    return ChatHandler
