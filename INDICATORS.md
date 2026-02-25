# Technical Indicator Glossary

Quick reference for every acronym and column name used in the feature matrix of this project.

---

## OHLCV — Raw Price & Volume


| Column   | Full Name | Description                                     |
| -------- | --------- | ----------------------------------------------- |
| `open`   | Open      | Price at the start of the trading period        |
| `high`   | High      | Highest price reached during the period         |
| `low`    | Low       | Lowest price reached during the period          |
| `close`  | Close     | Price at the end of the trading period          |
| `volume` | Volume    | Total number of shares traded during the period |


---

## RSI — Relative Strength Index


| Column | Description                                                                                                                                   |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `rsi`  | Oscillator (0–100) measuring the speed and magnitude of recent price changes. Above 70 → overbought; below 30 → oversold. Default period: 14. |


**Formula:**

```
RSI = 100 - (100 / (1 + RS))
RS  = Average Gain / Average Loss  (over N periods)
```

---

## MACD — Moving Average Convergence Divergence


| Column        | Full Name      | Description                                                           |
| ------------- | -------------- | --------------------------------------------------------------------- |
| `macd`        | MACD Line      | Difference between the 12-period EMA and 26-period EMA                |
| `macd_signal` | Signal Line    | 9-period EMA of the MACD Line; used as a trigger                      |
| `macd_diff`   | MACD Histogram | `macd - macd_signal`; positive = bullish momentum, negative = bearish |


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


| Column     | Full Name  | Description                                       |
| ---------- | ---------- | ------------------------------------------------- |
| `bb_upper` | Upper Band | SMA(20) + 2 × standard deviation                  |
| `bb_lower` | Lower Band | SMA(20) − 2 × standard deviation                  |
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


| Column   | Full Name     | Description                                                     |
| -------- | ------------- | --------------------------------------------------------------- |
| `sma_10` | 10-period SMA | Average closing price over the last 10 bars (short-term trend)  |
| `sma_20` | 20-period SMA | Average closing price over the last 20 bars (medium-term trend) |
| `sma_50` | 50-period SMA | Average closing price over the last 50 bars (long-term trend)   |


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


| Column   | Full Name     | Description                                     |
| -------- | ------------- | ----------------------------------------------- |
| `ema_9`  | 9-period EMA  | Fast EMA; more reactive to recent price changes |
| `ema_21` | 21-period EMA | Slower EMA; smoother, less noise                |


**Formula:**

```
EMA(today) = Price(today) × k  +  EMA(yesterday) × (1 - k)
k = 2 / (N + 1)
```

**Difference from SMA:** EMA gives more weight to recent prices, making it faster to react to price changes than SMA.

---

## Volume Derivative


| Column          | Full Name     | Description                                                                    |
| --------------- | ------------- | ------------------------------------------------------------------------------ |
| `volume_change` | Volume Change | Percentage change in volume from the previous bar: `(V_t - V_{t-1}) / V_{t-1}` |


**Interpretation:**

- Positive spike → increasing trading activity (confirms price moves)
- Negative → decreasing interest (may signal weakening trend)

---

---

## ATR — Average True Range


| Column | Description                                                                                                                                                        |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `atr`  | Rolling volatility measure based on the true range of price movement. High ATR = high volatility regime; low ATR = quiet/consolidating market. Default period: 14. |


**Formula:**

```
True Range = max(high - low, |high - prev_close|, |low - prev_close|)
ATR        = EMA(True Range, N)
```

**Interpretation:**

- Rising ATR → volatility expanding (trend may be accelerating or reversing)
- Falling ATR → volatility contracting (often precedes a breakout)

---

## OBV — On-Balance Volume


| Column | Description                                                                                                                             |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `obv`  | Cumulative volume indicator. Adds volume on up days, subtracts on down days. Divergence between OBV and price often precedes reversals. |


**Formula:**

```
OBV(t) = OBV(t-1) + volume  if close > prev_close
OBV(t) = OBV(t-1) - volume  if close < prev_close
OBV(t) = OBV(t-1)           if close == prev_close
```

---

## Stochastic Oscillator


| Column    | Full Name | Description                                                              |
| --------- | --------- | ------------------------------------------------------------------------ |
| `stoch_k` | %K Line   | Position of current close relative to the N-period high-low range, 0–100 |
| `stoch_d` | %D Line   | 3-period SMA of %K; acts as a signal line                                |


**Formula:**

```
%K = 100 × (close - lowest_low(N)) / (highest_high(N) - lowest_low(N))
%D = SMA(3) of %K
```

**Signals:**

- %K above 80 → overbought; below 20 → oversold
- %K crossing above %D → bullish momentum signal

---

## Williams %R


| Column       | Description                                                                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `williams_r` | Momentum oscillator ranging from −100 to 0. Above −20 = overbought; below −80 = oversold. Less lag than RSI on reversals. Default period: 14. |


**Formula:**

```
%R = -100 × (highest_high(N) - close) / (highest_high(N) - lowest_low(N))
```

---

## ADX — Average Directional Index


| Column | Description                                                                                                                    |
| ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `adx`  | Measures trend *strength* (not direction), 0–100. Above 25 = strong trend; below 20 = weak/ranging market. Default period: 14. |


**Formula:**

```
+DM = high - prev_high  (if positive, else 0)
-DM = prev_low - low    (if positive, else 0)
ADX = EMA of |+DI - -DI| / (+DI + -DI)
```

**Interpretation:**

- ADX > 25 → trend is strong (trade with trend)
- ADX < 20 → market is ranging (oscillators more reliable)

---

## External Market Features

These features are fetched separately via `ExternalDataFetcher` and merged by trading date.


| Column         | Source     | Description                                                                                 |
| -------------- | ---------- | ------------------------------------------------------------------------------------------- |
| `vix`          | `^VIX`     | CBOE Volatility Index close. >30 = fear; <15 = complacency                                  |
| `vix_change`   | derived    | 1-day pct change in VIX (fear momentum)                                                     |
| `yield_10y`    | `^TNX`     | US 10-year Treasury yield. Rising yields pressure growth stocks                             |
| `yield_spread` | derived    | 10yr yield minus 3-month T-bill yield. Negative = yield curve inversion (recession warning) |
| `dxy`          | `DX-Y.NYB` | US Dollar Index close. Strong dollar hurts multinational earnings                           |
| `dxy_change`   | derived    | 1-day pct change in DXY                                                                     |
| `spy_rs`       | derived    | `close / SPY_close` — relative strength vs broad market                                     |
| `xlk_rs`       | derived    | `close / XLK_close` — relative strength vs tech sector ETF                                  |


---

## Feature Matrix Summary


| #   | Column          | Category                    |
| --- | --------------- | --------------------------- |
| 1   | `open`          | OHLCV                       |
| 2   | `high`          | OHLCV                       |
| 3   | `low`           | OHLCV                       |
| 4   | `close`         | OHLCV                       |
| 5   | `volume`        | OHLCV                       |
| 6   | `rsi`           | RSI                         |
| 7   | `macd`          | MACD                        |
| 8   | `macd_signal`   | MACD                        |
| 9   | `macd_diff`     | MACD                        |
| 10  | `bb_upper`      | Bollinger Bands             |
| 11  | `bb_lower`      | Bollinger Bands             |
| 12  | `bb_width`      | Bollinger Bands             |
| 13  | `sma_10`        | SMA                         |
| 14  | `sma_20`        | SMA                         |
| 15  | `sma_50`        | SMA                         |
| 16  | `ema_9`         | EMA                         |
| 17  | `ema_21`        | EMA                         |
| 18  | `volume_change` | Volume                      |
| 19  | `atr`           | Advanced — Volatility       |
| 20  | `obv`           | Advanced — Volume           |
| 21  | `stoch_k`       | Advanced — Momentum         |
| 22  | `stoch_d`       | Advanced — Momentum         |
| 23  | `williams_r`    | Advanced — Momentum         |
| 24  | `adx`           | Advanced — Trend Strength   |
| 25  | `vix`           | External — Market Sentiment |
| 26  | `vix_change`    | External — Market Sentiment |
| 27  | `yield_10y`     | External — Macro            |
| 28  | `yield_spread`  | External — Macro            |
| 29  | `dxy`           | External — Macro            |
| 30  | `dxy_change`    | External — Macro            |
| 31  | `spy_rs`        | External — Cross-Asset      |
| 32  | `xlk_rs`        | External — Cross-Asset      |


