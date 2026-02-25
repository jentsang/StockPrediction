"""
Feature engineering and dataset preparation.

Steps:
  1. Add technical indicators (RSI, MACD, Bollinger Bands, SMAs, EMAs)
  2. Normalize features
  3. Build sliding-window sequences for LSTM input
  4. Train / val / test split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, config: dict):
        self.cfg = config
        self.feat_cfg = config["features"]
        self.seq_len = config["data"]["sequence_length"]
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    # ── Public ────────────────────────────────────────────────────────────────

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_technical_indicators(df)
        df = df.dropna()
        logger.info(f"Feature matrix shape after engineering: {df.shape}")
        return df

    def build_sequences(
        self, df: pd.DataFrame, target_col: str = "close"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays suitable for LSTM training."""
        if self.feat_cfg.get("normalize", True):
            scaled = self.scaler.fit_transform(df.values)
            df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
        else:
            df_scaled = df

        target_idx = df_scaled.columns.get_loc(target_col)
        data = df_scaled.values

        X, y = [], []
        for i in range(self.seq_len, len(data)):
            X.append(data[i - self.seq_len : i])
            y.append(data[i, target_idx])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train_val_test_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        n = len(X)
        train_end = int(n * self.cfg["data"]["train_split"])
        val_end = train_end + int(n * self.cfg["data"]["val_split"])

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(
            f"Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    def inverse_transform_close(self, values: np.ndarray, df_columns: list) -> np.ndarray:
        """Undo normalization on the close price column only."""
        dummy = np.zeros((len(values), len(df_columns)))
        close_idx = df_columns.index("close")
        dummy[:, close_idx] = values.flatten()
        return self.scaler.inverse_transform(dummy)[:, close_idx]

    # ── Private ───────────────────────────────────────────────────────────────

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import ta
        except ImportError:
            raise ImportError("ta not installed. Run: pip install ta")

        fc = self.feat_cfg

        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(
            close=df["close"], window=fc["rsi_period"]
        ).rsi()

        # MACD
        macd = ta.trend.MACD(
            close=df["close"],
            window_fast=fc["macd_fast"],
            window_slow=fc["macd_slow"],
            window_sign=fc["macd_signal"],
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=df["close"],
            window=fc["bb_period"],
            window_dev=fc["bb_std"],
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()

        # SMAs
        for period in fc["sma_periods"]:
            df[f"sma_{period}"] = ta.trend.SMAIndicator(
                close=df["close"], window=period
            ).sma_indicator()

        # EMAs
        for period in fc["ema_periods"]:
            df[f"ema_{period}"] = ta.trend.EMAIndicator(
                close=df["close"], window=period
            ).ema_indicator()

        # Volume change
        df["volume_change"] = df["volume"].pct_change()

        return df
