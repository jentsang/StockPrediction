"""
Training loop with early stopping, LR scheduling, and checkpoint saving.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "models" / "checkpoints"


class Trainer:
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.cfg = config["training"]
        self.device = self._resolve_device(config["training"]["device"])
        self.model.to(self.device)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )
        self.criterion = nn.MSELoss()
        self.scheduler = self._build_scheduler()

    # ── Public ────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        symbol: str = "model",
    ) -> dict:
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.cfg["epochs"] + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                f"Epoch {epoch:>4}/{self.cfg['epochs']} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(symbol)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg["early_stopping_patience"]:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
        return history

    def load_best(self, symbol: str = "model") -> None:
        path = CHECKPOINT_DIR / f"{symbol}_best.pth"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Loaded best checkpoint from {path}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(
            dataset, batch_size=self.cfg["batch_size"], shuffle=shuffle
        )

    def _save_checkpoint(self, symbol: str) -> None:
        path = CHECKPOINT_DIR / f"{symbol}_best.pth"
        torch.save(self.model.state_dict(), path)

    def _build_scheduler(self):
        sched = self.cfg.get("lr_scheduler", "plateau")
        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg["epochs"]
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=True
        )

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
