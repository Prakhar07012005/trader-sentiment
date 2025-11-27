import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

def perf_by_regime(df):
    agg = df.groupby("regime", dropna=False).agg(
        win_rate=("win_flag","mean"),
        mean_roi=("roi","mean"),
        median_roi=("roi","median"),
        pnl_mean=("Closed PnL","mean"),
        pnl_median=("Closed PnL","median"),
        pnl_std=("Closed PnL","std"),
        trades=("win_flag","count")
    ).reset_index()
    agg["sharpe_proxy"] = agg["pnl_mean"] / agg["pnl_std"]
    return agg

def side_by_regime(df):
    return df.groupby(["regime","side_long"], dropna=False).agg(
        win_rate=("win_flag","mean"),
        mean_roi=("roi","mean"),
        trades=("win_flag","count")
    ).reset_index()

def lev_by_regime(df):
    if "lev_bucket" not in df.columns:
        df["lev_bucket"] = pd.NA
    return df.groupby(["regime","lev_bucket"], dropna=False).agg(
        win_rate=("win_flag","mean"),
        mean_roi=("roi","mean"),
        trades=("win_flag","count")
    ).reset_index()

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))

    perf = perf_by_regime(df)
    side = side_by_regime(df)
    lev = lev_by_regime(df)

    perf.to_csv(os.path.join(REPORTS_DIR, "perf_by_regime.csv"), index=False)
    side.to_csv(os.path.join(REPORTS_DIR, "by_side.csv"), index=False)
    lev.to_csv(os.path.join(REPORTS_DIR, "by_lev.csv"), index=False)

    print("Saved summaries to reports/")

if __name__ == "__main__":
    main()
