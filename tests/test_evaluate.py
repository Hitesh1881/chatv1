import torch

from chatv1.evaluate import evaluate, EvalResult


class DummyModel:
    def eval(self):
        return self

    def __call__(self, x, y):
        return x, torch.tensor(1.0)


def test_evaluate_returns_metrics():
    result = evaluate(DummyModel(), torch.arange(40), 8, 2, 2, torch.device("cpu"))
    assert isinstance(result, EvalResult)
    assert result.loss == 1.0
    assert result.perplexity > 1.0
