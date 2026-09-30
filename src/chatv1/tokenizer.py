class CharTokenizer:
    """Deterministic character tokenizer for the tiny bootstrap stage."""

    def __init__(self, text: str):
        if not text:
            raise ValueError("training text must not be empty")
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unknown = sorted({ch for ch in text if ch not in self.stoi})
        if unknown:
            raise ValueError(f"unknown characters: {unknown[:5]}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        result = []
        for raw_id in ids:
            token_id = int(raw_id)
            if token_id not in self.itos:
                raise ValueError(f"unknown token id: {token_id}")
            result.append(self.itos[token_id])
        return "".join(result)

    def state_dict(self) -> dict:
        return {"stoi": self.stoi}

    @classmethod
    def from_state_dict(cls, state: dict):
        obj = cls.__new__(cls)
        obj.stoi = {str(ch): int(i) for ch, i in state["stoi"].items()}
        obj.itos = {i: ch for ch, i in obj.stoi.items()}
        return obj
