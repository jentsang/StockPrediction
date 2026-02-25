"""
Model evaluation — computes regression and directional accuracy metrics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        preds = []
        with torch.no_grad():
            for (X_batch,) in loader:
                out = self.model(X_batch.to(self.device))
                preds.append(out.cpu().numpy())
        return np.concatenate(preds).flatten()

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = self.predict(X)
        metrics = {
            "mse": float(np.mean((y_true - y_pred) ** 2)),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "directional_accuracy": self._directional_accuracy(y_true, y_pred),
        }
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.6f}")
        return metrics

    @staticmethod
    def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fraction of time the model correctly predicts up/down direction."""
        if len(y_true) < 2:
            return 0.0
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        return float(np.mean(actual_dir == pred_dir))
