from dataclasses import dataclass


@dataclass(frozen=True)
class ChatRequest:
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40


@dataclass(frozen=True)
class ChatResponse:
    text: str


def validate_request(request: ChatRequest) -> None:
    if not request.prompt.strip():
        raise ValueError("prompt must not be empty")
    if request.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if request.temperature <= 0:
        raise ValueError("temperature must be positive")
    if request.top_k < 0:
        raise ValueError("top_k must not be negative")
