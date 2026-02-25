"""
Training loop with early stopping, LR scheduling, and checkpoint saving.

CUDA optimisations (enabled automatically when a GPU is present):
  - cudnn.benchmark   : lets cuDNN auto-tune kernels for the fixed input shape
  - pin_memory        : page-locks CPU tensors so transfers overlap with compute
  - non_blocking      : asynchronous host->device copies
  - AMP (float16)     : forward pass in half precision (~2x throughput on modern GPUs)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
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

        self.is_cuda = self.device.type == "cuda"

        # cuDNN auto-tuner: finds fastest convolution algorithm for fixed input shape
        if self.is_cuda:
            torch.backends.cudnn.benchmark = True

        # AMP: float16 forward pass — only meaningful on CUDA
        use_amp_cfg = self.cfg.get("mixed_precision", True)
        self.use_amp = self.is_cuda and use_amp_cfg
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        logger.info(
            f"Device: {self.device} | "
            f"AMP: {'enabled' if self.use_amp else 'disabled'} | "
            f"cudnn.benchmark: {torch.backends.cudnn.benchmark if self.is_cuda else 'N/A'}"
        )

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

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # ReduceLROnPlateau requires the metric; other schedulers just step()
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

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
                # non_blocking overlaps CPU->GPU transfer with prior GPU work
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True).unsqueeze(1)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    preds = self.model(X_batch)
                    loss = self.criterion(preds, y_batch)

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

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
            dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=shuffle,
            pin_memory=self.is_cuda,   # page-lock CPU memory for faster transfers
            num_workers=0,             # keep 0 on Windows (avoids multiprocessing issues)
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
            self.optimizer, patience=5, factor=0.5
        )

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
