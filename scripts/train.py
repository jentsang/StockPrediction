"""
End-to-end training script — GRU (or LSTM) with full 32-feature input.

Usage:
    python scripts/train.py                      # all symbols in config
    python scripts/train.py --symbol AAPL
    python scripts/train.py --symbol MSFT --epochs 50
    python scripts/train.py --no-external        # skip external market features
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetcher import DataFetcher, ExternalDataFetcher
from src.data.processor import DataProcessor
from src.models.model_factory import build_model
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train GRU/LSTM stock prediction model")
    parser.add_argument("--symbol",      type=str,  default=None)
    parser.add_argument("--epochs",      type=int,  default=None)
    parser.add_argument("--no-external", action="store_true",
                        help="Disable external market features (VIX, yields, DXY, sector ETFs)")
    args = parser.parse_args()

    config = load_config()
    config["data"]["source"] = "yfinance"

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.no_external:
        config["external_data"]["enabled"] = False

    symbols    = [args.symbol] if args.symbol else config["symbols"]
    use_ext    = config.get("external_data", {}).get("enabled", True)

    fetcher    = DataFetcher(config)
    processor  = DataProcessor(config)

    # Fetch external data once — same date window applies to all symbols
    ext_df = None
    if use_ext:
        ext_fetcher = ExternalDataFetcher(config)
        ext_df = ext_fetcher.fetch()
        logger.info(f"External features loaded: {list(ext_df.columns)}")

    for symbol in symbols:
        logger.info(f"\n{'='*52}\nTraining {config['model']['type'].upper()} for: {symbol}\n{'='*52}")

        # 1. Fetch + engineer features
        raw_df  = fetcher.fetch(symbol)
        feat_df = processor.process(raw_df, external_df=ext_df)
        config["model"]["input_size"] = feat_df.shape[1]
        logger.info(f"Feature matrix: {feat_df.shape}  |  input_size set to {feat_df.shape[1]}")

        # 2. Build sequences and split
        X, y = processor.build_sequences(feat_df, target_col="close")
        X_train, y_train, X_val, y_val, X_test, y_test = processor.train_val_test_split(X, y)
        logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

        # 3. Build model
        model = build_model(config)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {config['model']['type'].upper()} | params: {total_params:,}")

        # 4. Train
        trainer = Trainer(model, config)
        history = trainer.train(X_train, y_train, X_val, y_val, symbol=symbol)

        # 5. Load best checkpoint and evaluate on test set
        trainer.load_best(symbol=symbol)
        evaluator = Evaluator(model, trainer.device)
        logger.info(f"\nTest set metrics for {symbol}:")
        metrics = evaluator.evaluate(X_test, y_test)

        logger.info(
            f"\nSummary [{symbol}] | "
            f"RMSE={metrics['rmse']:.6f} | "
            f"MAE={metrics['mae']:.6f} | "
            f"Dir.Acc={metrics['directional_accuracy']:.2%}"
        )

    logger.info("\nAll symbols complete.")


if __name__ == "__main__":
    main()
