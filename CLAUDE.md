---
description: 
alwaysApply: true
---

---
description: 
alwaysApply: true
---

# CLAUDE.md — Project Context for AI Assistants

This file gives AI assistants instant context about this project. Read it before making any changes.

---

## What This Project Is

A **locally-running stock price prediction and day-trading system** built in Python. It downloads historical OHLCV data, engineers 32 technical/macro features, trains a GRU or LSTM deep learning model, and backtests trading signals — all on the user's local machine. No cloud infrastructure. Trading platform integration (Alpaca) is planned but not yet implemented.

**Current phase:** Model development and backtesting on historical daily data.  
**Next phase:** Switch to intraday (1-min) data, then live paper trading via Alpaca.

---

## Repository Layout

```
StockPrediction/
├── CLAUDE.md                        ← you are here
├── INDICATORS.md                    ← glossary for every feature column
├── README.md                        ← setup and usage guide
├── requirements.txt
├── config/
│   └── config.yaml                  ← single source of truth for all settings
├── data/
│   ├── raw/                         ← .parquet files cached here by DataFetcher
│   └── processed/                   ← results, equity curves
├── models/
│   └── checkpoints/                 ← best .pth weights saved per symbol during training
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb      ← full train → evaluate → backtest loop
├── scripts/
│   ├── download_data.py             ← CLI: fetch and cache historical data
│   ├── train.py                     ← CLI: end-to-end train pipeline
│   └── inspect_data.py              ← quick dataset summary printer
├── src/
│   ├── data/
│   │   ├── fetcher.py               ← DataFetcher (Alpaca + yFinance)
│   │   └── processor.py             ← DataProcessor (feature engineering + sequences)
│   ├── models/
│   │   ├── model_factory.py         ← build_model(config) — always use this, not direct imports
│   │   ├── lstm/
│   │   │   ├── lstm_model.py        ← LSTMModel (nn.LSTM)
│   │   │   └── lstm_architecture.png
│   │   └── gru/
│   │       ├── gru_model.py         ← GRUModel (nn.GRU)
│   │       └── gru_architecture.png
│   ├── training/
│   │   ├── trainer.py               ← Trainer (train loop, early stopping, checkpointing)
│   │   └── evaluator.py             ← Evaluator (predict, MSE/RMSE/MAE/directional accuracy)
│   ├── backtesting/
│   │   └── backtest.py              ← Backtester (signal-based, stop loss + take profit)
│   └── utils/
│       ├── config.py                ← load_config() — loads config/config.yaml
│       └── logger.py                ← get_logger(name) — structured stdout logger
└── tests/
    ├── test_data.py                 ← DataProcessor unit tests
    └── test_model.py                ← LSTMModel, GRUModel, model_factory tests
```

---

## Data Pipeline

```
DataFetcher.fetch(symbol)
    └── yFinance or Alpaca API
    └── Cached as data/raw/{SYMBOL}_{TIMEFRAME}_{start}_{end}.parquet

DataProcessor.process(df, external_df=None)
    ├── _add_technical_indicators()   → RSI, MACD, BB, SMA(10/20/50), EMA(9/21), volume_change
    ├── _add_advanced_indicators()    → ATR, OBV, Stochastic(%K/%D), Williams%R, ADX
    └── _merge_external_features()    → VIX, vix_change, yield_10y, yield_spread, DXY, dxy_change,
                                        spy_rs, xlk_rs  (optional, requires external_df)

DataProcessor.build_sequences(df)
    └── Sliding window of seq_len=60 bars → (X, y) numpy arrays

DataProcessor.train_val_test_split(X, y)
    └── 70% train / 15% val / 15% test  (chronological, no shuffle)
```

**Feature count:** 18 base + 6 advanced + 8 external = **32 total** (when external data enabled).  
**config `model.input_size`** must match actual feature count — `scripts/train.py` auto-updates this at runtime.

---

## Models

Both models share identical hyperparameters and the same `from_config(config)` classmethod interface.

### GRU (`src/models/gru/gru_model.py`) — current default
- 2 gates: reset (r), update (z)
- Single hidden state `h0` — no cell state
- ~25% fewer parameters than LSTM
- Faster training, better on smaller datasets
- Recommended starting point

### LSTM (`src/models/lstm/lstm_model.py`)
- 3 gates: input, forget, output
- Two states: hidden `h0` + cell `c0`
- Better for very long sequences (60+ day lookbacks)
- Marginal accuracy edge at scale

### Switching models
Change one line in `config.yaml`:
```yaml
model:
  type: gru   # or lstm
```

### Adding a new model
1. Create `src/models/{name}/{name}_model.py` with a class that implements `forward()` and `from_config(config)`
2. Add an architecture diagram PNG in the same directory
3. Register in `src/models/model_factory.py`:
```python
from src.models.{name}.{name}_model import {Name}Model
_REGISTRY["{name}"] = {Name}Model
```

---

## Training Pipeline

```python
# scripts/train.py and notebooks/03_model_training.ipynb
fetcher  = DataFetcher(config)
processor = DataProcessor(config)
df_raw   = fetcher.fetch(symbol)
df_feat  = processor.process(df_raw, external_df=None)   # pass external_df for macro features
config["model"]["input_size"] = df_feat.shape[1]          # IMPORTANT: always update before build_model()
X, y     = processor.build_sequences(df_feat)
X_train, y_train, X_val, y_val, X_test, y_test = processor.train_val_test_split(X, y)
model    = build_model(config)                            # from src.models.model_factory
trainer  = Trainer(model, config)
history  = trainer.train(X_train, y_train, X_val, y_val, symbol=symbol)
trainer.load_best(symbol)                                 # restores best checkpoint from models/checkpoints/
evaluator = Evaluator(model, trainer.device)
metrics  = evaluator.evaluate(X_test, y_test)             # MSE, RMSE, MAE, directional_accuracy
```

Key trainer behaviours:
- **Early stopping** — patience=15 epochs (config: `training.early_stopping_patience`)
- **Gradient clipping** — max_norm=1.0 (prevents exploding gradients)
- **LR scheduler** — cosine annealing by default; also supports ReduceLROnPlateau
- **Checkpointing** — saves best val-loss weights to `models/checkpoints/{symbol}_best.pth`

---

## Backtesting

```python
backtester = Backtester(config)
summary, equity_df = backtester.run(actual_prices, pred_prices, threshold=0.002)
```

**Strategy logic (long-only):**
- BUY when `predicted_close > current_close × (1 + threshold)`
- Position automatically closed at stop loss (−2%) or take profit (+4%)
- 10% of capital risked per trade, 0.1% commission + 0.05% slippage applied

**Key metrics returned:** `final_equity`, `total_return_pct`, `max_drawdown_pct`, `sharpe_ratio`, `total_trades`, `win_rate_pct`

---

## Configuration Reference (`config/config.yaml`)

| Key | Current Value | Notes |
|---|---|---|
| `data.source` | `alpaca` | Change to `yfinance` for quick runs (no API key) |
| `data.timeframe` | `15Min` | `1Min`/`5Min`/`15Min`/`1Hour`/`1Day` — yFinance intraday is last ~60 calendar days |
| `data.sequence_length` | `60` | Lookback window in bars fed to model |
| `model.type` | `gru` | Change to `lstm` to swap architecture |
| `model.input_size` | `32` | Auto-updated at runtime by train.py |
| `external_data.enabled` | `true` | Set `false` to skip macro features |
| `training.device` | `auto` | Uses CUDA if available, else CPU |

---

## Environment Setup

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env   # then add Alpaca API keys
```

**Required .env keys** (only needed for Alpaca data source or live trading):
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**yFinance** requires no API key and works out of the box for daily data.

---

## Running Commands

```powershell
# Download data (yfinance, no key needed)
python scripts/download_data.py --source yfinance --start 2018-01-01 --end 2024-12-31

# Download single symbol
python scripts/download_data.py --symbol AAPL --source yfinance

# Train all symbols
python scripts/train.py

# Train single symbol, override epochs
python scripts/train.py --symbol AAPL --epochs 50

# Run tests
python -m pytest tests/ -v
```

---

## Coding Conventions

- **Config is the single source of truth** — never hardcode hyperparameters in model or training files
- **Always use `build_model(config)`** from `model_factory.py` — never import `LSTMModel`/`GRUModel` directly in scripts or notebooks
- **Always update `config["model"]["input_size"]`** to `df_feat.shape[1]` before calling `build_model()` — the actual feature count varies depending on whether external data is enabled
- **Data is cached as `.parquet`** in `data/raw/` — `use_cache=True` by default in `DataFetcher.fetch()`
- **Logger**: use `get_logger(__name__)` from `src.utils.logger` — do not use `print()` in src/ or scripts/
- **Tests**: mock configs in tests must mirror the nested structure in `config.yaml` — `features.technical_indicators` is a sub-dict, not flat

---

## Known Issues / Limitations

- **Windows terminal encoding**: the `→` unicode arrow breaks PowerShell's cp1252 encoding — use `->` in all logger strings
- **yFinance intraday limit**: yFinance only returns the last ~60 days for intervals < 1 day — use Alpaca for multi-year intraday history
- **git version**: git 2.26 (currently installed) does not support `--trailer` flag — upgrade to 2.31+ before running git commit via Cursor

---

## Roadmap

- [ ] Alpaca live paper trading integration (WebSocket stream)
- [ ] Walk-forward validation (rolling train window)
- [ ] Multi-symbol portfolio backtesting
- [ ] Transformer / hybrid LSTM-Transformer model option
- [ ] Sentiment features (news headlines via free API)
- [ ] Plotly Dash monitoring dashboard
