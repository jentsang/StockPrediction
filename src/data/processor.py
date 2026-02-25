"""
Feature engineering and dataset preparation.

Steps:
  1. Add technical indicators (RSI, MACD, Bollinger Bands, SMAs, EMAs)
  2. Add advanced indicators (ATR, OBV, Stochastic, Williams %R, ADX)
  3. Optionally merge external market features (VIX, yields, DXY, sector ETFs)
  4. Normalize features
  5. Build sliding-window sequences for LSTM input
  6. Train / val / test split
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

    def process(
        self,
        df: pd.DataFrame,
        external_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build the full feature matrix.

        Parameters
        ----------
        df          : OHLCV DataFrame for a single symbol.
        external_df : Optional output of ExternalDataFetcher.fetch().
                      When provided, macro/cross-asset columns are merged in
                      and derived features (vix_change, yield_spread, etc.)
                      are computed.
        """
        df = df.copy()
        df = self._add_technical_indicators(df)
        df = self._add_advanced_indicators(df)
        if external_df is not None:
            df = self._merge_external_features(df, external_df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        logger.info(f"Feature matrix shape after engineering: {df.shape}")
        return df

    def build_sequences(
        self, df: pd.DataFrame, target_col: str = "close"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays suitable for LSTM/GRU training.

        Scaler fit strategy (no lookahead bias)
        ----------------------------------------
        The MinMaxScaler is fit exclusively on the training rows, then used to
        transform val and test rows.  Training rows are defined as all rows
        whose data participates in a training sequence — i.e. the first
        ``train_split × n_sequences`` sequences, which cover rows
        ``0 : train_seq_end + seq_len``.

        Concretely (default config, 1208 feature rows):
          n_sequences    = 1208 − 60  = 1148
          train_seq_end  = int(1148 × 0.70) = 803
          scaler fit on  rows 0 → 862  (803 + 60 − 1)
          scaler transform → all 1208 rows
        """
        data = df.values

        if self.feat_cfg.get("normalize", True):
            n_seq = len(data) - self.seq_len
            train_seq_end = int(n_seq * self.cfg["data"]["train_split"])
            # +seq_len covers the lookback window AND the last training target row
            train_row_end = train_seq_end + self.seq_len

            self.scaler.fit(data[:train_row_end])
            scaled = self.scaler.transform(data)
        else:
            scaled = data

        target_idx = list(df.columns).index(target_col)

        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i - self.seq_len : i])
            y.append(scaled[i, target_idx])

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

        fc = self.feat_cfg["technical_indicators"]

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

    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import ta
        except ImportError:
            raise ImportError("ta not installed. Run: pip install ta")

        fc = self.feat_cfg.get("advanced_indicators", {})
        atr_p      = fc.get("atr_period",      14)
        stoch_p    = fc.get("stoch_period",     14)
        stoch_s    = fc.get("stoch_smooth",      3)
        williams_p = fc.get("williams_period",  14)
        adx_p      = fc.get("adx_period",       14)

        # ATR — Average True Range (volatility regime)
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=atr_p
        ).average_true_range()

        # OBV — On-Balance Volume (cumulative buy/sell pressure)
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=df["close"], volume=df["volume"]
        ).on_balance_volume()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"],
            window=stoch_p, smooth_window=stoch_s,
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Williams %R — momentum / overbought-oversold
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            high=df["high"], low=df["low"], close=df["close"], lbp=williams_p
        ).williams_r()

        # ADX — Average Directional Index (trend strength, 0-100)
        df["adx"] = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=adx_p
        ).adx()

        return df

    def _merge_external_features(
        self, df: pd.DataFrame, external_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align external market data to the stock DataFrame's trading calendar
        and compute derived cross-asset features.

        Added columns
        -------------
        vix            : VIX close
        vix_change     : VIX 1-day pct change (fear momentum)
        yield_10y      : 10-year Treasury yield
        yield_spread   : 10yr yield minus 3-month T-bill (term premium)
        dxy            : US Dollar Index close
        dxy_change     : DXY 1-day pct change
        spy_rs         : Stock close / SPY close (relative strength)
        xlk_rs         : Stock close / XLK close (tech-sector relative strength)
        """
        ext = external_df.copy()

        # Normalise to tz-naive midnight dates so all tickers land on the same row.
        idx = pd.to_datetime(ext.index)
        ext.index = idx.tz_convert(None) if idx.tz is not None else idx
        ext.index = ext.index.normalize()

        # Some tickers use different UTC offsets (e.g. SPY at 05:00, VIX at 06:00),
        # producing split rows after normalization. groupby().last() merges them by
        # taking the last non-NaN value per column per date.
        if ext.index.duplicated().any():
            ext = ext.groupby(level=0).last()

        if df.index.tz is not None:
            df_align_idx = df.index.normalize().tz_localize(None)
        else:
            df_align_idx = df.index.normalize()

        ext = ext.reindex(df_align_idx, method="ffill")
        ext.index = df.index  # restore original (possibly tz-aware) index

        if "vix_close" in ext.columns:
            df["vix"]        = ext["vix_close"]
            df["vix_change"] = ext["vix_close"].pct_change()

        if "yield_10y_close" in ext.columns:
            df["yield_10y"] = ext["yield_10y_close"]
            if "yield_3m_close" in ext.columns:
                df["yield_spread"] = ext["yield_10y_close"] - ext["yield_3m_close"]

        if "dxy_close" in ext.columns:
            df["dxy"]        = ext["dxy_close"]
            df["dxy_change"] = ext["dxy_close"].pct_change()

        if "spy_close" in ext.columns:
            spy = ext["spy_close"].replace(0, np.nan)
            df["spy_rs"] = df["close"] / spy

        if "xlk_close" in ext.columns:
            xlk = ext["xlk_close"].replace(0, np.nan)
            df["xlk_rs"] = df["close"] / xlk

        return df
