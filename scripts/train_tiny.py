from pathlib import Path
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
OUT.mkdir(exist_ok=True)


def main():
    torch.manual_seed(42)
    tok = CharTokenizer(TEXT)
    data = torch.tensor(tok.encode(TEXT), dtype=torch.long)
    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128)
    device = get_device()
    model = ChatV1(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    model.train()
    for step in range(400):
        ix = torch.randint(0, len(data) - cfg.block_size - 1, (16,))
        x = torch.stack([data[i:i+cfg.block_size] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix]).to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    torch.save({"config": cfg.__dict__, "tokenizer": tok.stoi, "state_dict": model.state_dict()}, OUT / "tiny_chatv1.pt")
    print(f"saved {OUT / 'tiny_chatv1.pt'}")


if __name__ == "__main__":
    main()
