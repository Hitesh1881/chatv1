from pathlib import Path
import argparse
import random

import torch

from chatv1.device import get_device
from chatv1.model import ChatV1, ModelConfig
from chatv1.tokenizer import CharTokenizer

TEXT = """
User: hello
Assistant: Hello! I am ChatV1, an experimental AI system.
User: what can you do?
Assistant: I can learn patterns from training text and generate responses.
User: what is Python?
Assistant: Python is a programming language used to build software and automation.
User: explain React.
Assistant: React is a library for building user interfaces from components.
""" * 200

OUT = Path("artifacts")


def batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts]).to(device)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = CharTokenizer(TEXT)
    data = torch.tensor(tok.encode(TEXT), dtype=torch.long)

    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128)
    device = get_device()
    model = ChatV1(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    start_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if checkpoint.get("optimizer"):
            opt.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        print(f"resumed step={start_step} from {args.resume}")

    model.train()
    for step in range(start_step, start_step + args.steps):
        x, y = batch(data, cfg.block_size, 16, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f}")

        if (step + 1) % 100 == 0:
            OUT.mkdir(exist_ok=True)
            torch.save(
                {
                    "config": cfg.__dict__,
                    "tokenizer": tok.state_dict(),
                    "state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step + 1,
                },
                OUT / "tiny_chatv1_latest.pt",
            )

    OUT.mkdir(exist_ok=True)
    final = OUT / "tiny_chatv1.pt"
    torch.save(
        {
            "config": cfg.__dict__,
            "tokenizer": tok.state_dict(),
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": start_step + args.steps,
        },
        final,
    )
    print(f"saved {final}")


if __name__ == "__main__":
    main()
