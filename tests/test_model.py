import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.lstm_model import LSTMModel


MOCK_CONFIG = {
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


def test_lstm_forward_pass():
    model = LSTMModel.from_config(MOCK_CONFIG)
    batch_size, seq_len, input_size = 8, 60, 18
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)
    assert out.shape == (batch_size, 1), f"Expected (8, 1), got {out.shape}"


def test_lstm_bidirectional():
    config = {
        "model": {**MOCK_CONFIG["model"], "bidirectional": True}
    }
    model = LSTMModel.from_config(config)
    x = torch.randn(4, 60, 18)
    out = model(x)
    assert out.shape == (4, 1)


def test_lstm_parameter_count():
    model = LSTMModel.from_config(MOCK_CONFIG)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0, "Model should have trainable parameters"
