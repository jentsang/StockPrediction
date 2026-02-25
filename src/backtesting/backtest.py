"""
Simple signal-based backtester.

Strategy:
  - BUY  when predicted_close > current_close * (1 + threshold)
  - SELL when predicted_close < current_close * (1 - threshold)
  - Stop loss and take profit enforced per position

Outputs a summary dict and an equity curve DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Backtester:
    def __init__(self, config: dict):
        bt = config["backtesting"]
        self.initial_capital = bt["initial_capital"]
        self.commission = bt["commission"]
        self.slippage = bt["slippage"]
        self.position_size = bt["position_size"]
        self.stop_loss = bt["stop_loss"]
        self.take_profit = bt["take_profit"]

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        threshold: float = 0.002,
    ) -> tuple[dict, pd.DataFrame]:
        """
        Args:
            prices:      Actual close prices (already inverse-transformed).
            predictions: Model-predicted close prices (inverse-transformed).
            threshold:   Minimum predicted move to trigger a trade.

        Returns:
            (summary_dict, equity_curve_df)
        """
        capital = self.initial_capital
        position = 0.0          # shares held
        entry_price = 0.0
        trades = []
        equity_curve = []

        for i in range(1, len(prices)):
            price = float(prices[i])
            pred = float(predictions[i - 1])
            prev_price = float(prices[i - 1])

            # ── Check open position ───────────────────────────────────────
            if position > 0:
                pnl_pct = (price - entry_price) / entry_price
                if pnl_pct <= -self.stop_loss or pnl_pct >= self.take_profit:
                    exit_price = price * (1 - self.slippage)
                    proceeds = position * exit_price * (1 - self.commission)
                    capital += proceeds
                    trades.append(
                        {
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl_pct": pnl_pct,
                            "type": "sl" if pnl_pct <= -self.stop_loss else "tp",
                        }
                    )
                    position = 0.0

            # ── Signal generation ─────────────────────────────────────────
            if position == 0:
                if pred > prev_price * (1 + threshold):
                    # BUY signal
                    buy_price = price * (1 + self.slippage)
                    trade_capital = capital * self.position_size
                    position = (trade_capital * (1 - self.commission)) / buy_price
                    capital -= trade_capital
                    entry_price = buy_price

            equity_curve.append(
                {
                    "step": i,
                    "price": price,
                    "equity": capital + position * price,
                }
            )

        # Close any open position at end
        if position > 0:
            last_price = float(prices[-1]) * (1 - self.slippage)
            capital += position * last_price * (1 - self.commission)

        equity_df = pd.DataFrame(equity_curve)
        summary = self._compute_summary(equity_df, trades)
        self._log_summary(summary)
        return summary, equity_df

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_summary(self, equity_df: pd.DataFrame, trades: list) -> dict:
        final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        equity = equity_df["equity"].values
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        max_drawdown = float(drawdowns.min())

        daily_returns = pd.Series(equity).pct_change().dropna()
        sharpe = (
            float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
            if daily_returns.std() > 0
            else 0.0
        )

        win_trades = [t for t in trades if t["pnl_pct"] > 0]

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "total_trades": len(trades),
            "win_rate_pct": round(len(win_trades) / len(trades) * 100, 2) if trades else 0.0,
        }

    def _log_summary(self, summary: dict) -> None:
        logger.info("── Backtest Summary ──────────────────────")
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")
        logger.info("──────────────────────────────────────────")
