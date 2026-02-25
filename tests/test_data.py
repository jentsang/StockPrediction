import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.processor import DataProcessor


MOCK_CONFIG = {
    "data": {
        "source": "yfinance",
        "timeframe": "1Day",
        "start_date": "2022-01-01",
        "end_date": "2023-01-01",
        "sequence_length": 10,
        "train_split": 0.70,
        "val_split": 0.15,
        "test_split": 0.15,
    },
    "features": {
        "normalize": True,
        "technical_indicators": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2,
            "sma_periods": [10, 20, 50],
            "ema_periods": [9, 21],
        },
    },
}


def _make_mock_ohlcv(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        }
    )


def test_process_adds_features():
    df = _make_mock_ohlcv()
    proc = DataProcessor(MOCK_CONFIG)
    feat_df = proc.process(df)
    assert feat_df.shape[1] > 5, "Expected more columns after feature engineering"
    assert feat_df.isnull().sum().sum() == 0, "No NaNs expected after dropna()"


def test_build_sequences_shapes():
    df = _make_mock_ohlcv()
    proc = DataProcessor(MOCK_CONFIG)
    feat_df = proc.process(df)
    X, y = proc.build_sequences(feat_df)
    seq_len = MOCK_CONFIG["data"]["sequence_length"]
    assert X.shape[1] == seq_len
    assert X.shape[2] == feat_df.shape[1]
    assert len(X) == len(y)


def test_train_val_test_split_sizes():
    df = _make_mock_ohlcv()
    proc = DataProcessor(MOCK_CONFIG)
    feat_df = proc.process(df)
    X, y = proc.build_sequences(feat_df)
    X_train, y_train, X_val, y_val, X_test, y_test = proc.train_val_test_split(X, y)
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(X), "All samples must be accounted for"
