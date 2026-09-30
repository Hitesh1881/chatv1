import pytest

from chatv1.api import ChatRequest, validate_request


def test_valid_request():
    validate_request(ChatRequest(prompt="hello"))


@pytest.mark.parametrize(
    "request",
    [
        ChatRequest(prompt=""),
        ChatRequest(prompt="hello", max_new_tokens=0),
        ChatRequest(prompt="hello", temperature=0),
        ChatRequest(prompt="hello", top_k=-1),
    ],
)
def test_invalid_requests(request):
    with pytest.raises(ValueError):
        validate_request(request)
