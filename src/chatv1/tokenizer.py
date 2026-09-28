class CharTokenizer:
    """Small deterministic character tokenizer for the first training stage."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text: str):
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            raise ValueError(f"unknown characters: {unknown[:5]}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)
