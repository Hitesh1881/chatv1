from pathlib import Path
from http.server import HTTPServer

import torch

from chatv1.chat_http import make_chat_handler
from chatv1.chat_service import ChatService
from chatv1.conversation import InMemoryConversationStore
from chatv1.engine import ChatEngine
from chatv1.device import get_device
from chatv1.model import ChatV1, ModelConfig
from chatv1.tokenizer import CharTokenizer


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**checkpoint["config"])
    tokenizer = CharTokenizer.from_state_dict(checkpoint["tokenizer"])
    model = ChatV1(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, tokenizer


def main() -> None:
    checkpoint = Path("artifacts/tiny_chatv1.pt")
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}. Run scripts/train_tiny.py first.")
    device = get_device()
    model, tokenizer = load_model(checkpoint, device)
    service = ChatService(ChatEngine(model=model, tokenizer=tokenizer), InMemoryConversationStore())
    server = HTTPServer(("127.0.0.1", 8001), make_chat_handler(lambda: service))
    print("Chat API: http://127.0.0.1:8001/v1/chat/completions")
    print(f"Device: {device}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
