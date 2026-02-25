"""
CLI script to download and cache historical data for all configured symbols,
plus optional external market/macro data (VIX, yields, DXY, sector ETFs).

Usage:
    python scripts/download_data.py                          # all symbols (yfinance)
    python scripts/download_data.py --symbol AAPL --source yfinance
    python scripts/download_data.py --external               # external data only
    python scripts/download_data.py --all                    # symbols + external
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetcher import DataFetcher, ExternalDataFetcher
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download historical stock + external data")
    parser.add_argument("--symbol",   type=str,  default=None,  help="Single symbol override")
    parser.add_argument("--source",   type=str,  default=None,  help="Data source override (alpaca|yfinance)")
    parser.add_argument("--start",    type=str,  default=None,  help="Start date override (YYYY-MM-DD)")
    parser.add_argument("--end",      type=str,  default=None,  help="End date override (YYYY-MM-DD)")
    parser.add_argument("--external", action="store_true",       help="Download external market data (VIX, yields, DXY, SPY, XLK)")
    parser.add_argument("--all",      action="store_true",       help="Download symbol data AND external data")
    args = parser.parse_args()

    config = load_config()

    if args.source:
        config["data"]["source"] = args.source

    download_symbols  = not args.external or args.all
    download_external = args.external or args.all

    # ── Symbol OHLCV data ─────────────────────────────────────────────────────
    if download_symbols:
        symbols = [args.symbol] if args.symbol else config["symbols"]
        fetcher = DataFetcher(config)

        for sym in symbols:
            try:
                df = fetcher.fetch(sym, start=args.start, end=args.end, use_cache=False)
                logger.info(f"{sym}: {len(df)} rows | columns: {list(df.columns)}")
            except Exception as e:
                logger.error(f"Failed to fetch {sym}: {e}")

    # ── External market/macro data ────────────────────────────────────────────
    if download_external:
        ext_fetcher = ExternalDataFetcher(config)
        try:
            ext_df = ext_fetcher.fetch(start=args.start, end=args.end, use_cache=False)
            logger.info(f"External data: {len(ext_df)} rows | columns: {list(ext_df.columns)}")
        except Exception as e:
            logger.error(f"Failed to fetch external data: {e}")


if __name__ == "__main__":
    main()
