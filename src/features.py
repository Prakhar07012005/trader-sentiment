import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Encode side: long=1 if buy/long, short=0
    df["side_long"] = df["Side"].astype(str).str.lower().str.contains("buy|long").astype(int)

    # Defensive notional for ROI
    notional = (df["Execution Price"].abs() * df["Size Tokens"].abs())
    df["roi"] = np.where(notional > 0, df["Closed PnL"] / notional, np.nan)

    # Win flag
    df["win_flag"] = (df["Closed PnL"] > 0).astype(int)

    # Leverage buckets
    if "Leverage" in df.columns:
        df["lev_bucket"] = pd.cut(
            df["Leverage"],
            bins=[-np.inf,2,5,10,25,1000],
            labels=["L1(<=2)","L2(2-5)","L3(5-10)","L4(10-25)","L5(>25)"]
        )
    else:
        df["lev_bucket"] = pd.NA

    # Hour of day (optional, if timestamp parsed)
    if "ts" in df.columns:
        dt = pd.to_datetime(df["ts"], errors="coerce")
        df["hour"] = dt.dt.hour

    return df

def main():
    merged_path = os.path.join(DATA_DIR, "merged.parquet")
    df = pd.read_parquet(merged_path)

    df = compute_features(df)
    out_path = os.path.join(DATA_DIR, "features.parquet")
    df.to_parquet(out_path, index=False)
    print("Features saved:", out_path)

if __name__ == "__main__":
    main()
