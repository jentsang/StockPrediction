import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.lstm.lstm_model import LSTMModel
from src.models.gru.gru_model import GRUModel
from src.models.model_factory import build_model, available_models


BASE_CONFIG = {
    "model": {
        "type": "lstm",
        "input_size": 18,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "output_size": 1,
        "bidirectional": False,
    }
}


def _config(model_type: str, bidirectional: bool = False) -> dict:
    return {"model": {**BASE_CONFIG["model"], "type": model_type, "bidirectional": bidirectional}}


@pytest.mark.parametrize("model_type", ["lstm", "gru"])
def test_forward_pass(model_type):
    model = build_model(_config(model_type))
    x = torch.randn(8, 60, 18)
    out = model(x)
    assert out.shape == (8, 1), f"{model_type}: expected (8,1), got {out.shape}"


@pytest.mark.parametrize("model_type", ["lstm", "gru"])
def test_bidirectional(model_type):
    model = build_model(_config(model_type, bidirectional=True))
    x = torch.randn(4, 60, 18)
    out = model(x)
    assert out.shape == (4, 1)


def test_gru_fewer_params_than_lstm():
    lstm = build_model(_config("lstm"))
    gru = build_model(_config("gru"))
    lstm_params = sum(p.numel() for p in lstm.parameters())
    gru_params = sum(p.numel() for p in gru.parameters())
    assert gru_params < lstm_params, (
        f"GRU ({gru_params}) should have fewer params than LSTM ({lstm_params})"
    )


def test_factory_unknown_type():
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model({"model": {**BASE_CONFIG["model"], "type": "transformer"}})


def test_available_models():
    models = available_models()
    assert "lstm" in models
    assert "gru" in models
