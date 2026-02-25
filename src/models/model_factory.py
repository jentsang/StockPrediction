"""
Model factory — returns the right model based on config["model"]["type"].

Supported types:
  lstm  → LSTMModel
  gru   → GRUModel

Usage:
    from src.models.model_factory import build_model
    model = build_model(config)
"""

import torch.nn as nn

from src.models.gru.gru_model import GRUModel
from src.models.lstm.lstm_model import LSTMModel

_REGISTRY: dict[str, type] = {
    "lstm": LSTMModel,
    "gru": GRUModel,
}


def build_model(config: dict) -> nn.Module:
    model_type = config["model"].get("type", "lstm").lower()
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[model_type]
    return cls.from_config(config)


def available_models() -> list[str]:
    return list(_REGISTRY.keys())
