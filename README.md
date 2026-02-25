# StockPrediction

A locally-running LSTM-based stock price prediction system designed for day trading. Fetches historical OHLCV data, engineers technical indicators, trains a deep learning model, and backtests trading signals — all on your own machine.

---

## Project Structure

```
StockPrediction/
├── config/
│   └── config.yaml          # All hyperparameters, data settings, backtest config
├── data/
│   ├── raw/                 # Downloaded OHLCV data (auto-cached as .parquet)
│   └── processed/           # Results, equity curves
├── models/
│   └── checkpoints/         # Saved .pth model weights
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── scripts/
│   ├── download_data.py     # CLI: fetch and cache data
│   └── train.py             # CLI: full train → evaluate pipeline
├── src/
│   ├── data/
│   │   ├── fetcher.py       # Alpaca / yFinance data download
│   │   └── processor.py     # Feature engineering + sequence builder
│   ├── models/
│   │   └── lstm_model.py    # LSTM architecture (PyTorch)
│   ├── training/
│   │   ├── trainer.py       # Training loop, early stopping, checkpointing
│   │   └── evaluator.py     # MSE, RMSE, MAE, directional accuracy
│   ├── backtesting/
│   │   └── backtest.py      # Signal-based backtester with stop loss / take profit
│   └── utils/
│       ├── config.py        # YAML config loader
│       └── logger.py        # Structured logging
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/jentsang/StockPrediction.git
cd StockPrediction

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Mac/Linux
```

Edit `.env` and add your [Alpaca API keys](https://alpaca.markets) (free account, no deposit required for paper trading).

### 3. Download historical data

```bash
# Download all symbols defined in config.yaml (uses yfinance, no key needed)
python scripts/download_data.py --source yfinance

# Download a single symbol via Alpaca (requires API key in .env)
python scripts/download_data.py --symbol AAPL --source alpaca
```

### 4. Train the model

```bash
# Train on all configured symbols
python scripts/train.py

# Train on a single symbol
python scripts/train.py --symbol AAPL --epochs 100
```

### 5. Explore with notebooks

```bash
jupyter notebook notebooks/
```

Open them in order:
1. `01_data_exploration.ipynb` — visualize raw price data
2. `02_feature_engineering.ipynb` — inspect technical indicators and correlations
3. `03_model_training.ipynb` — train, evaluate, and backtest interactively

---

## Configuration

All settings live in `config/config.yaml`. Key sections:

| Section | What it controls |
|---|---|
| `symbols` | List of tickers to train on |
| `data` | Source, timeframe, date range, train/val/test split |
| `features` | Which indicators to compute and their parameters |
| `model` | LSTM hidden size, layers, dropout |
| `training` | Epochs, batch size, learning rate, early stopping |
| `backtesting` | Capital, commission, stop loss, take profit |

---

## Model Architecture

```
Input: (batch, seq_len=60, features=18)
  → LSTM (128 hidden, 2 layers, dropout=0.2)
  → Dropout
  → Linear(128 → 1)
Output: predicted next-close price (normalized)
```

**Features engineered (18 total):**
OHLCV raw · RSI · MACD · MACD Signal · MACD Histogram · Bollinger Upper/Lower/Width · SMA(10,20,50) · EMA(9,21) · Volume Change

---

## Backtesting Strategy

| Parameter | Default |
|---|---|
| Initial capital | $10,000 |
| Position size | 10% of capital per trade |
| Stop loss | 2% |
| Take profit | 4% (2:1 reward/risk) |
| Commission | 0.1% |
| Signal | BUY when predicted move > 0.2% threshold |

---

## Roadmap

- [ ] Live paper trading integration (Alpaca WebSocket)
- [ ] Transformer / hybrid LSTM-Transformer model
- [ ] Multi-symbol portfolio backtesting
- [ ] Sentiment feature integration (news headlines)
- [ ] Walk-forward validation
- [ ] Dashboard UI (Plotly Dash)

---

## Data Sources

| Source | Free Tier | Best For |
|---|---|---|
| **Alpaca** | 10yr intraday (1-min bars) | Primary — data + trading in one |
| **yFinance** | 20yr daily | Quick prototyping, no key needed |
| **Polygon.io** | Paid ($79/mo) | Production real-time WebSocket |

---

## Disclaimer

This project is for **educational purposes only**. Nothing here constitutes financial advice. Past backtesting performance does not guarantee future results. Always paper trade before going live.
