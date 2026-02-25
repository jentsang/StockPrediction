"""
End-to-end training script.

Usage:
    python scripts/train.py
    python scripts/train.py --symbol MSFT --epochs 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.models.lstm_model import LSTMModel
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM stock prediction model")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config()

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    symbols = [args.symbol] if args.symbol else config["symbols"]

    fetcher = DataFetcher(config)
    processor = DataProcessor(config)

    for symbol in symbols:
        logger.info(f"\n{'='*50}\nTraining model for: {symbol}\n{'='*50}")

        # 1. Fetch data
        raw_df = fetcher.fetch(symbol)

        # 2. Feature engineering
        feat_df = processor.process(raw_df)

        # Dynamically update input_size to match actual feature count
        config["model"]["input_size"] = feat_df.shape[1]

        # 3. Build sequences
        X, y = processor.build_sequences(feat_df, target_col="close")
        X_train, y_train, X_val, y_val, X_test, y_test = processor.train_val_test_split(X, y)

        # 4. Build model
        model = LSTMModel.from_config(config)
        logger.info(f"Model: {model}")

        # 5. Train
        trainer = Trainer(model, config)
        history = trainer.train(X_train, y_train, X_val, y_val, symbol=symbol)

        # 6. Load best checkpoint and evaluate
        trainer.load_best(symbol=symbol)
        evaluator = Evaluator(model, trainer.device)
        logger.info(f"\nTest set evaluation for {symbol}:")
        metrics = evaluator.evaluate(X_test, y_test)

    logger.info("All symbols trained successfully.")


if __name__ == "__main__":
    main()
