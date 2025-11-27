import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

FBI_FILE = os.path.join(DATA_DIR, "fear_greed_index.csv")
TRADES_FILE = os.path.join(DATA_DIR, "historical_data.csv")
PARQUET_FILE = os.path.join(DATA_DIR, "historical_data.parquet")

def load_fgi():
    # Columns: timestamp (epoch), value (int), classification (str), date (YYYY-MM-DD)
    fgi = pd.read_csv(FBI_FILE)
    # Parse date to datetime.date
    fgi["date"] = pd.to_datetime(fgi["date"], errors="coerce").dt.date
    # Regime bins from 'value'
    fgi["regime"] = pd.cut(
        fgi["value"],
        bins=[-1,24,44,54,74,200],  # 200 upper guard
        labels=["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"]
    )
    return fgi[["date","value","classification","regime"]].dropna(subset=["date"])

def load_trades_chunked(to_parquet=True):
    use_cols = [
    "Account","Coin","Execution Price","Size Tokens","Size USD","Side",
    "Timestamp","Timestamp IST","Start Position","Direction","Closed PnL",
    "Transaction Hash","Order ID","Crossed","Fee","Trade ID"]

    chunks = pd.read_csv(TRADES_FILE, usecols=use_cols, chunksize=200_000, low_memory=False)
    df = pd.concat(chunks, ignore_index=True)

    # Timestamps: pick the cleaner of the two
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df["Timestamp IST"] = pd.to_datetime(df["Timestamp IST"], errors="coerce")  # naive local
    ts_primary = "Timestamp" if df["Timestamp"].notna().sum() >= df["Timestamp IST"].notna().sum() else "Timestamp IST"
    df["ts"] = df[ts_primary]
    df["date"] = pd.to_datetime(df["ts"], errors="coerce").dt.date

    # Numeric fields
    for col in ["Execution Price","Size Tokens","Closed PnL","Leverage","Fee","Size USD"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional: write parquet for faster re-runs
    if to_parquet:
        try:
            df.to_parquet(PARQUET_FILE, index=False)
        except Exception:
            pass

    return df

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fgi = load_fgi()
    trades = None
    # Prefer parquet if exists
    if os.path.exists(PARQUET_FILE):
        trades = pd.read_parquet(PARQUET_FILE)
    else:
        trades = load_trades_chunked(to_parquet=True)

    # Merge sentiment onto trades
    merged = trades.merge(fgi, on="date", how="left")
    merged.to_parquet(os.path.join(DATA_DIR, "merged.parquet"), index=False)
    print("Merged dataset saved:", os.path.join(DATA_DIR, "merged.parquet"))
    print("Rows:", len(merged), "Sentiment coverage:", merged["regime"].notna().mean())

if __name__ == "__main__":
    main()
