"""
Data fetcher — pulls historical OHLCV bars from Alpaca or yFinance,
plus an ExternalDataFetcher for market-wide macro/cross-asset features.

Usage:
    fetcher = DataFetcher(config)
    df = fetcher.fetch("AAPL", start="2022-01-01", end="2024-01-01")

    ext = ExternalDataFetcher(config)
    ext_df = ext.fetch(start="2022-01-01", end="2024-01-01")
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


class DataFetcher:
    def __init__(self, config: dict):
        self.config = config
        self.source = config["data"]["source"]
        self.timeframe = config["data"]["timeframe"]
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch(
        self,
        symbol: str,
        start: str = None,
        end: str = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        start = start or self.config["data"]["start_date"]
        end = end or self.config["data"]["end_date"]

        cache_path = RAW_DATA_DIR / f"{symbol}_{self.timeframe}_{start}_{end}.parquet"
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data for {symbol} from {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching {symbol} [{self.timeframe}] {start} -> {end} via {self.source}")

        if self.source == "alpaca":
            df = self._fetch_alpaca(symbol, start, end)
        elif self.source == "yfinance":
            df = self._fetch_yfinance(symbol, start, end)
        else:
            raise ValueError(f"Unknown data source: {self.source}")

        df.to_parquet(cache_path)
        logger.info(f"Saved {len(df)} rows to {cache_path}")
        return df

    def fetch_multiple(self, symbols: list, **kwargs) -> dict[str, pd.DataFrame]:
        return {sym: self.fetch(sym, **kwargs) for sym in symbols}

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_alpaca(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in your .env file"
            )

        client = StockHistoricalDataClient(api_key, secret_key)

        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = tf_map.get(self.timeframe, TimeFrame(1, TimeFrameUnit.Day))

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
        )
        bars = client.get_stock_bars(request)
        df = bars.df.reset_index()

        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        df = df.rename(columns={"timestamp": "datetime"})
        df = df.set_index("datetime")
        return df

    def _fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        interval_map = {
            "1Min": "1m",
            "5Min": "5m",
            "15Min": "15m",
            "1Hour": "1h",
            "1Day": "1d",
        }
        interval = interval_map.get(self.timeframe, "1d")

        # yFinance only supports intraday history for the last 60 days
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        df.index.name = "datetime"
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]


class ExternalDataFetcher:
    """
    Downloads and caches market-wide macro/cross-asset features.

    Fetched series
    --------------
    vix_close    : CBOE Volatility Index daily close
    yield_10y    : US 10-year Treasury yield (^TNX)
    yield_3m     : US 13-week T-bill yield (^IRX)  — short-rate proxy
    dxy_close    : US Dollar Index close (DX-Y.NYB)
    spy_close    : SPY ETF close — broad market benchmark
    xlk_close    : XLK (Tech sector ETF) close — sector benchmark

    Derived features (computed in DataProcessor)
    --------------------------------------------
    vix_change, yield_spread, dxy_change, spy_rs, xlk_rs
    """

    # ticker_name -> yfinance symbol
    _TICKERS: dict[str, str] = {
        "vix":      "^VIX",
        "yield_10y": "^TNX",
        "yield_3m":  "^IRX",
        "dxy":      "DX-Y.NYB",
        "spy":      "SPY",
        "xlk":      "XLK",
    }

    def __init__(self, config: dict):
        self.config = config
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        start: str = None,
        end: str = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return a date-aligned DataFrame of external market features."""
        start = start or self.config["data"]["start_date"]
        end   = end   or self.config["data"]["end_date"]

        cache_path = RAW_DATA_DIR / f"external_{start}_{end}.parquet"
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached external data from {cache_path}")
            return pd.read_parquet(cache_path)

        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        logger.info(f"Fetching external market data {start} -> {end}")

        series: list[pd.Series] = []
        for name, ticker in self._TICKERS.items():
            try:
                raw = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
                if raw.empty:
                    logger.warning(f"  {ticker}: returned empty DataFrame — skipping")
                    continue
                idx = pd.to_datetime(raw.index)
                # tz_convert handles tz-aware; tz-naive indexes need no conversion
                raw.index = idx.tz_convert(None) if idx.tz is not None else idx
                raw.index = raw.index.normalize()  # collapse to midnight date before concat
                raw.index.name = "datetime"
                series.append(raw["Close"].rename(f"{name}_close"))
                logger.info(f"  {ticker:15s} : {len(raw)} rows")
            except Exception as exc:
                logger.warning(f"  {ticker}: fetch failed — {exc}")

        if not series:
            raise RuntimeError(
                "ExternalDataFetcher: all ticker downloads failed. "
                "Check your internet connection or yfinance version."
            )

        result = pd.concat(series, axis=1)
        # yfinance can return duplicate dates across tickers after tz conversion;
        # deduplicate here so the cached parquet is always clean.
        result = result[~result.index.duplicated(keep="last")]
        result.to_parquet(cache_path)
        logger.info(f"Saved external data ({len(result)} rows) to {cache_path}")
        return result
