from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class HFModelConfig:
    block_size: int = 4096


class HFChatTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, prompt: str) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return list(ids)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class HFChatModel:
    def __init__(self, model, device):
        self.model = model
        self.cfg = HFModelConfig()
        self.device = device

    @classmethod
    def load(cls, model_id: str = MODEL_ID, device: torch.device | None = None):
        if device is None:
            device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        return cls(model, device), HFChatTokenizer(tokenizer)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens=128, temperature=0.7, top_k=40):
        attention_mask = torch.ones_like(input_ids)
        do_sample = temperature > 0 and top_k != 1
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.model.config.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_k"] = top_k
        return self.model.generate(**kwargs)
