import json
from http.server import BaseHTTPRequestHandler
from typing import Callable

from .chat_service import ChatService
from .inference import GenerationConfig, stream_token_ids
import torch

MAX_BODY_BYTES = 64 * 1024
MAX_PROMPT_CHARS = 8_000
MAX_CONVERSATION_ID_CHARS = 128


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

        def _read_request(self) -> tuple[str, str, dict]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            conversation_id = str(payload.get("conversation_id", "default"))
            prompt = str(payload.get("prompt", ""))
            if not conversation_id.strip() or len(conversation_id) > MAX_CONVERSATION_ID_CHARS:
                raise ValueError("invalid conversation_id")
            if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
                raise ValueError("prompt must be non-empty and within the size limit")
            return conversation_id, prompt, payload

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/v1/chat/completions", "/v1/chat/completions/stream"}:
                self._json(404, {"error": "not_found"})
                return
            try:
                conversation_id, prompt, payload = self._read_request()
                if self.path.endswith("/stream") or bool(payload.get("stream")):
                    service = service_factory()
                    result = service.engine
                    conversation = service.conversations.get(conversation_id)
                    conversation.add("user", prompt)
                    request = __import__("chatv1.engine", fromlist=["ChatEngineRequest"]).ChatEngineRequest(conversation_id, prompt, GenerationConfig(), service.retrieve_top_k)
                    context, _ = result._context(request)
                    full_prompt = f"Context:\\n{context}\\n\\nUser: {prompt}\\nAssistant:" if context else f"User: {prompt}\\nAssistant:"
                    ids = service.engine.tokenizer.encode(full_prompt)
                    input_ids = torch.tensor([ids], dtype=torch.long, device=next(service.engine.model.parameters()).device)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    generated = []
                    for token_id in stream_token_ids(service.engine.model, input_ids, request.generation):
                        token = service.engine.tokenizer.decode([token_id])
                        generated.append(token)
                        self.wfile.write(("data: " + json.dumps({"token": token}) + "\\n\\n").encode())
                        self.wfile.flush()
                    conversation.add("assistant", "".join(generated))
                    service.conversations.save(conversation)
                    self.wfile.write(b"data: [DONE]\\n\\n")
                    self.wfile.flush()
                    return
                result = service_factory().reply(conversation_id, prompt)
                self._json(200, {"conversation_id": result.conversation_id, "text": result.text})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": "invalid_request", "message": str(exc)})
            except Exception:
                if self.path.endswith("/stream"):
                    try:
                        self.wfile.write(("data: " + json.dumps({"error": "chat_failed"}) + "\\n\\n").encode())
                        self.wfile.flush()
                    except Exception:
                        pass
                else:
                    self._json(500, {"error": "chat_failed"})

    return ChatHandler
