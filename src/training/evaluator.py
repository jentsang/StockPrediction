"""
Model evaluation — regression metrics and binary classification metrics.

Regression  : MSE, RMSE, MAE, directional accuracy
Classification : accuracy, precision, recall, F1, ROC-AUC
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task: str = "regression",
    ):
        self.model  = model
        self.device = device
        self.task   = task

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw model outputs (logits for classification, values for regression)."""
        self.model.eval()
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader  = DataLoader(dataset, batch_size=256, shuffle=False)
        preds   = []
        with torch.no_grad():
            for (X_batch,) in loader:
                out = self.model(X_batch.to(self.device))
                preds.append(out.cpu().numpy())
        return np.concatenate(preds).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Apply sigmoid to logits → probabilities in [0, 1]. Classification only."""
        logits = self.predict(X)
        return 1.0 / (1.0 + np.exp(-logits))

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        if self.task == "classification":
            return self._evaluate_classification(X, y_true)
        return self._evaluate_regression(X, y_true)

    # ── Private ───────────────────────────────────────────────────────────────

    def _evaluate_regression(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = self.predict(X)
        metrics = {
            "mse":                  float(np.mean((y_true - y_pred) ** 2)),
            "rmse":                 float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mae":                  float(np.mean(np.abs(y_true - y_pred))),
            "directional_accuracy": self._directional_accuracy(y_true, y_pred),
        }
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.6f}")
        return metrics

    def _evaluate_classification(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score,
            )
        except ImportError:
            raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")

        y_prob   = self.predict_proba(X)
        y_labels = (y_prob >= 0.5).astype(int)
        y_int    = y_true.astype(int)

        metrics = {
            "accuracy":   round(float(accuracy_score(y_int, y_labels)),  4),
            "precision":  round(float(precision_score(y_int, y_labels, zero_division=0)), 4),
            "recall":     round(float(recall_score(y_int, y_labels, zero_division=0)),    4),
            "f1":         round(float(f1_score(y_int, y_labels, zero_division=0)),        4),
            "roc_auc":    round(float(roc_auc_score(y_int, y_prob)),                      4),
            "up_rate_actual":    round(float(y_int.mean()),    4),
            "up_rate_predicted": round(float(y_labels.mean()), 4),
        }
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")
        return metrics

    @staticmethod
    def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return 0.0
        actual_dir = np.sign(np.diff(y_true))
        pred_dir   = np.sign(np.diff(y_pred))
        return float(np.mean(actual_dir == pred_dir))
