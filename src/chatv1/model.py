from dataclasses import dataclass
from typing import Iterator
import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(y))

    def forward_cached(self, x: torch.Tensor, past_key_value=None):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if t > 1:
            total = k.size(2)
            past = total - t
            mask = torch.tril(
                torch.ones(t, total, device=x.device, dtype=torch.bool),
                diagonal=past,
            )
            att = att.masked_fill(~mask.view(1, 1, t, total), float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

    def forward_cached(self, x: torch.Tensor, past_key_value=None):
        attn_out, present = self.attn.forward_cached(self.ln1(x), past_key_value)
        x = x + attn_out
        return x + self.mlp(self.ln2(x)), present


class ChatV1(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        _, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block size {self.cfg.block_size}")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def _sample(self, logits, temperature, top_k):
        logits = logits / max(temperature, 1e-5)
        if top_k:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        return torch.multinomial(torch.softmax(logits, dim=-1), 1)

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens=64, temperature=0.8, top_k=40) -> Iterator[int]:
        idx = idx[:, -self.cfg.block_size:]
        t = idx.size(1)
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        past_key_values = []
        for block in self.blocks:
            x, present = block.forward_cached(x)
            past_key_values.append(present)
        logits = self.lm_head(self.ln_f(x))
        for _ in range(max_new_tokens):
            next_token = self._sample(logits[:, -1, :], temperature, top_k)
            yield int(next_token[0, 0])
            pos_id = min(t, self.cfg.block_size - 1)
            x = self.drop(self.token_emb(next_token) + self.pos_emb(torch.tensor([pos_id], device=idx.device)))
            new_past = []
            for block, past in zip(self.blocks, past_key_values):
                x, present = block.forward_cached(x, past)
                new_past.append(present)
            past_key_values = new_past
            logits = self.lm_head(self.ln_f(x))
            t += 1

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.8, top_k=40):
        for token_id in self.generate_stream(idx, max_new_tokens, temperature, top_k):
            token = torch.tensor([[token_id]], device=idx.device)
            idx = torch.cat((idx, token), dim=1)
        return idx
