from pathlib import Path
import argparse

import torch

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tiny_chatv1.pt"))
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    device = get_device()
    model, tokenizer = load_model(args.checkpoint, device)
    print("ChatV1 tiny demo. Type 'exit' to quit.")

    while True:
        prompt = input("You: ")
        if prompt.strip().lower() == "exit":
            break
        prefix = prompt + "\nAssistant:"
        try:
            ids = tokenizer.encode(prefix)
        except ValueError as exc:
            print(f"Input contains unsupported characters: {exc}")
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(x, max_new_tokens=args.max_new_tokens)
        decoded = tokenizer.decode(output[0].tolist())
        print("Assistant:", decoded[len(prefix):].strip())


if __name__ == "__main__":
    main()
