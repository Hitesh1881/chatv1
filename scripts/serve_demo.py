import argparse
from http.server import HTTPServer

from chatv1.chat_http import make_chat_handler
from chatv1.chat_service import ChatService
from chatv1.conversation import InMemoryConversationStore
from chatv1.diffusion import DiffusionConfig, LocalDiffusionGenerator
from chatv1.engine import ChatEngine
from chatv1.http import make_image_handler
from chatv1.tokenizer import CharTokenizer


def build_chat_service():
    # Placeholder wiring until a trained checkpoint is loaded by the demo.
    raise RuntimeError("load a trained ChatV1 checkpoint and tokenizer before enabling chat")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    args = parser.parse_args()
    image_generator = LocalDiffusionGenerator(DiffusionConfig(model_id=args.model))
    server = HTTPServer((args.host, args.port), make_image_handler(lambda: image_generator))
    print(f"Image API: http://{args.host}:{args.port}/v1/images/generate")
    print("Chat API wiring is intentionally disabled until a trained checkpoint is loaded.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
