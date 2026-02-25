"""
CLI script to download and cache historical data for all configured symbols.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --symbol AAPL --source yfinance
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetcher import DataFetcher
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download historical stock data")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol override")
    parser.add_argument("--source", type=str, default=None, help="Data source override (alpaca|yfinance)")
    parser.add_argument("--start", type=str, default=None, help="Start date override (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date override (YYYY-MM-DD)")
    args = parser.parse_args()

    config = load_config()

    if args.source:
        config["data"]["source"] = args.source

    symbols = [args.symbol] if args.symbol else config["symbols"]
    fetcher = DataFetcher(config)

    for sym in symbols:
        try:
            df = fetcher.fetch(
                sym,
                start=args.start,
                end=args.end,
                use_cache=False,
            )
            logger.info(f"{sym}: {len(df)} rows | columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Failed to fetch {sym}: {e}")


if __name__ == "__main__":
    main()
