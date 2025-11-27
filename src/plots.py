import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

ORDER = ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"]

def plot_win_rate(perf):
    plt.figure(figsize=(8,4))
    sns.barplot(data=perf, x="regime", y="win_rate", order=ORDER)
    plt.title("Win rate by sentiment regime")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "win_rate_by_regime.png"), dpi=150)

def plot_roi_distribution(df):
    plt.figure(figsize=(10,4))
    sns.violinplot(data=df, x="regime", y="roi", order=ORDER, cut=0, inner="quartile")
    plt.title("ROI distribution by sentiment regime")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roi_distribution.png"), dpi=150)

def plot_leverage_distribution(df):
    plt.figure(figsize=(10,4))
    sns.countplot(data=df, x="lev_bucket", hue="regime",
                  order=["L1(<=2)","L2(2-5)","L3(5-10)","L4(10-25)","L5(>25)"])
    plt.title("Leverage usage across regimes")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "leverage_by_regime.png"), dpi=150)

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
    perf = pd.read_csv(os.path.join(REPORTS_DIR, "perf_by_regime.csv"))

    plot_win_rate(perf)
    plot_roi_distribution(df)
    plot_leverage_distribution(df)
    print("Saved plots to reports/")

if __name__ == "__main__":
    main()
