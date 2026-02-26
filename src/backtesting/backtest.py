"""
Signal-based backtester with risk controls.

Strategy:
  - BUY  when predicted_close > current_close * (1 + threshold)
  - Hard stop loss, take profit, and trailing stop enforced per position
  - Portfolio-level circuit breakers:
      * max_drawdown_halt  — no new entries once portfolio drops N% from its peak
      * max_consecutive_losses — pause entries after N losses in a row

Risk (config: backtesting.risk, 1–100)
  risk = % of capital deployed per trade.
    risk=10  → 10% of capital per trade
    risk=50  → 50% of capital per trade
    risk=100 → 100% of capital per trade (all-in)

  Exit / circuit-breaker params are fixed and can be overridden individually:
    stop_loss=1.5%  take_profit=3.0%  trailing_stop=1.5%
    max_drawdown_halt=12%  max_consecutive_losses=3

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
        self.commission      = bt["commission"]
        self.slippage        = bt["slippage"]

        # classification | regression — controls how predictions are interpreted
        self.task = config.get("model", {}).get("task", "regression")

        # ── Risk: % of capital per trade ──────────────────────────────────────
        risk = float(bt.get("risk", 10))
        if not 1 <= risk <= 100:
            raise ValueError(f"backtesting.risk must be between 1 and 100, got {risk}")

        self.risk          = risk
        self.position_size = bt.get("position_size", risk / 100.0)

        self.stop_loss       = bt.get("stop_loss",       0.015)
        self.take_profit     = bt.get("take_profit",     0.030)
        self.trailing_stop   = bt.get("trailing_stop",   0.015)
        self.max_drawdown_halt      = bt.get("max_drawdown_halt",      0.12)
        self.max_consecutive_losses = bt.get("max_consecutive_losses", 3)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[dict, pd.DataFrame]:
        """
        Args:
            prices:      Actual close prices in USD.
            predictions: Regression — predicted close prices.
                         Classification — raw model logits (sigmoid applied internally).
            threshold:   Regression: min predicted price move % to trigger a buy.
                         Classification: min up-probability to trigger a buy (default 0.5).

        Returns:
            (summary_dict, equity_curve_df)
        """
        predictions = np.array(predictions, dtype=np.float64)
        if self.task == "classification":
            # convert logits → up-probabilities
            predictions = 1.0 / (1.0 + np.exp(-predictions))
        capital = self.initial_capital
        position = 0.0          # shares held
        entry_price = 0.0
        price_high = 0.0        # highest price seen since entry (trailing stop)
        trades = []
        equity_curve = []

        portfolio_peak = self.initial_capital
        consecutive_losses = 0

        for i in range(1, len(prices)):
            price = float(prices[i])
            pred = float(predictions[i - 1])
            prev_price = float(prices[i - 1])

            # ── Maintain portfolio peak for drawdown circuit breaker ──────
            current_equity = capital + position * price
            if current_equity > portfolio_peak:
                portfolio_peak = current_equity

            # ── Check open position ───────────────────────────────────────
            exit_reason = None
            if position > 0:
                price_high = max(price_high, price)
                pnl_pct = (price - entry_price) / entry_price
                trailing_pnl = (price - price_high) / price_high if self.trailing_stop else None

                if pnl_pct <= -self.stop_loss:
                    exit_reason = "sl"
                elif pnl_pct >= self.take_profit:
                    exit_reason = "tp"
                elif self.trailing_stop and trailing_pnl is not None and trailing_pnl <= -self.trailing_stop:
                    exit_reason = "trailing_sl"

                if exit_reason:
                    exit_price = price * (1 - self.slippage)
                    proceeds = position * exit_price * (1 - self.commission)
                    capital += proceeds
                    trades.append(
                        {
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl_pct": pnl_pct,
                            "type": exit_reason,
                        }
                    )
                    if pnl_pct <= 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    position = 0.0
                    price_high = 0.0

            # ── Circuit breakers ──────────────────────────────────────────
            drawdown_from_peak = (current_equity - portfolio_peak) / portfolio_peak
            halted_drawdown = (
                self.max_drawdown_halt is not None
                and drawdown_from_peak <= -self.max_drawdown_halt
            )
            halted_losses = (
                self.max_consecutive_losses is not None
                and consecutive_losses >= self.max_consecutive_losses
            )

            # ── Signal generation ─────────────────────────────────────────
            if position == 0 and not halted_drawdown and not halted_losses:
                if self.task == "classification":
                    buy_signal = pred >= threshold          # pred is now a probability
                else:
                    buy_signal = pred > prev_price * (1 + threshold)

                if buy_signal:
                    buy_price = price * (1 + self.slippage)
                    trade_capital = capital * self.position_size
                    position = (trade_capital * (1 - self.commission)) / buy_price
                    capital -= trade_capital
                    entry_price = buy_price
                    price_high = buy_price

            equity_curve.append(
                {
                    "step": i,
                    "price": price,
                    "equity": capital + position * price,
                    "halted": halted_drawdown or halted_losses,
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

        win_trades  = [t for t in trades if t["pnl_pct"] > 0]
        sl_trades   = [t for t in trades if t["type"] == "sl"]
        tp_trades   = [t for t in trades if t["type"] == "tp"]
        tsl_trades  = [t for t in trades if t["type"] == "trailing_sl"]
        halted_bars = int(equity_df["halted"].sum()) if "halted" in equity_df.columns else 0

        return {
            "risk_level":         self.risk,
            "initial_capital":    self.initial_capital,
            "final_equity":       round(final_equity, 2),
            "total_return_pct":   round(total_return * 100, 2),
            "max_drawdown_pct":   round(max_drawdown * 100, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "total_trades":       len(trades),
            "win_rate_pct":       round(len(win_trades) / len(trades) * 100, 2) if trades else 0.0,
            "stop_loss_exits":    len(sl_trades),
            "trailing_sl_exits":  len(tsl_trades),
            "take_profit_exits":  len(tp_trades),
            "halted_bars":        halted_bars,
        }

    def _log_summary(self, summary: dict) -> None:
        logger.info("── Backtest Summary ──────────────────────")
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")
        logger.info("──────────────────────────────────────────")
