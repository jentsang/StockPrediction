import pandas as pd
import os

symbols = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]
data_dir = "data/raw"

for sym in symbols:
    path = os.path.join(data_dir, f"{sym}_1Day_2018-01-01_2024-12-31.parquet")
    df = pd.read_parquet(path)
    print(f"--- {sym} ---")
    print(f"  Rows      : {len(df)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Columns   : {list(df.columns)}")
    close_min = df["close"].min()
    close_max = df["close"].max()
    close_last = df["close"].iloc[-1]
    print(f"  Close     : min=${close_min:.2f}  max=${close_max:.2f}  last=${close_last:.2f}")
    null_count = df.isnull().sum().sum()
    print(f"  Nulls     : {null_count}")
    print()

print("=== Sample: AAPL first 5 rows ===")
aapl = pd.read_parquet(os.path.join(data_dir, "AAPL_1Day_2018-01-01_2024-12-31.parquet"))
print(aapl.head().to_string())
print()
print("=== Sample: AAPL last 5 rows ===")
print(aapl.tail().to_string())
