# Technical Indicator Glossary

Quick reference for every acronym and column name used in the feature matrix of this project.

---

## OHLCV — Raw Price & Volume

| Column | Full Name | Description |
|---|---|---|
| `open` | Open | Price at the start of the trading period |
| `high` | High | Highest price reached during the period |
| `low` | Low | Lowest price reached during the period |
| `close` | Close | Price at the end of the trading period |
| `volume` | Volume | Total number of shares traded during the period |

---

## RSI — Relative Strength Index

| Column | Description |
|---|---|
| `rsi` | Oscillator (0–100) measuring the speed and magnitude of recent price changes. Above 70 → overbought; below 30 → oversold. Default period: 14. |

**Formula:**
```
RSI = 100 - (100 / (1 + RS))
RS  = Average Gain / Average Loss  (over N periods)
```

---

## MACD — Moving Average Convergence Divergence

| Column | Full Name | Description |
|---|---|---|
| `macd` | MACD Line | Difference between the 12-period EMA and 26-period EMA |
| `macd_signal` | Signal Line | 9-period EMA of the MACD Line; used as a trigger |
| `macd_diff` | MACD Histogram | `macd - macd_signal`; positive = bullish momentum, negative = bearish |

**Formula:**
```
MACD Line   = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram   = MACD Line - Signal Line
```

**Signals:**
- MACD Line crosses **above** Signal Line → bullish (potential buy)
- MACD Line crosses **below** Signal Line → bearish (potential sell)

---

## BB — Bollinger Bands

| Column | Full Name | Description |
|---|---|---|
| `bb_upper` | Upper Band | SMA(20) + 2 × standard deviation |
| `bb_lower` | Lower Band | SMA(20) − 2 × standard deviation |
| `bb_width` | Band Width | `bb_upper - bb_lower`; measures market volatility |

**Formula:**
```
Middle Band = SMA(20)
Upper Band  = SMA(20) + (2 × σ)
Lower Band  = SMA(20) - (2 × σ)
Band Width  = Upper Band - Lower Band
```

**Interpretation:**
- Price near `bb_upper` → potentially overbought
- Price near `bb_lower` → potentially oversold
- Narrow `bb_width` → low volatility (often precedes a breakout)
- Wide `bb_width` → high volatility

---

## SMA — Simple Moving Average

| Column | Full Name | Description |
|---|---|---|
| `sma_10` | 10-period SMA | Average closing price over the last 10 bars (short-term trend) |
| `sma_20` | 20-period SMA | Average closing price over the last 20 bars (medium-term trend) |
| `sma_50` | 50-period SMA | Average closing price over the last 50 bars (long-term trend) |

**Formula:**
```
SMA(N) = (P₁ + P₂ + ... + Pₙ) / N
```

**Common signals:**
- Price crossing **above** SMA → bullish
- Price crossing **below** SMA → bearish
- SMA(10) crossing above SMA(50) → "Golden Cross" (bullish)
- SMA(10) crossing below SMA(50) → "Death Cross" (bearish)

---

## EMA — Exponential Moving Average

| Column | Full Name | Description |
|---|---|---|
| `ema_9` | 9-period EMA | Fast EMA; more reactive to recent price changes |
| `ema_21` | 21-period EMA | Slower EMA; smoother, less noise |

**Formula:**
```
EMA(today) = Price(today) × k  +  EMA(yesterday) × (1 - k)
k = 2 / (N + 1)
```

**Difference from SMA:** EMA gives more weight to recent prices, making it faster to react to price changes than SMA.

---

## Volume Derivative

| Column | Full Name | Description |
|---|---|---|
| `volume_change` | Volume Change | Percentage change in volume from the previous bar: `(V_t - V_{t-1}) / V_{t-1}` |

**Interpretation:**
- Positive spike → increasing trading activity (confirms price moves)
- Negative → decreasing interest (may signal weakening trend)

---

## Feature Matrix Summary

| # | Column | Category |
|---|---|---|
| 1 | `open` | OHLCV |
| 2 | `high` | OHLCV |
| 3 | `low` | OHLCV |
| 4 | `close` | OHLCV |
| 5 | `volume` | OHLCV |
| 6 | `rsi` | RSI |
| 7 | `macd` | MACD |
| 8 | `macd_signal` | MACD |
| 9 | `macd_diff` | MACD |
| 10 | `bb_upper` | Bollinger Bands |
| 11 | `bb_lower` | Bollinger Bands |
| 12 | `bb_width` | Bollinger Bands |
| 13 | `sma_10` | SMA |
| 14 | `sma_20` | SMA |
| 15 | `sma_50` | SMA |
| 16 | `ema_9` | EMA |
| 17 | `ema_21` | EMA |
| 18 | `volume_change` | Volume |
