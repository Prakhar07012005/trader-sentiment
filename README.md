# 📊 Trader Behavior vs. Market Sentiment

## 🎯 Objective
Analyze how trader performance varies across Bitcoin market sentiment regimes (Extreme Fear → Extreme Greed) by joining **Hyperliquid trade history** with the **Fear-Greed Index**.  
The goal: uncover behavioral patterns and recommend sentiment-aware trading strategies.

---

## 📂 Project Structure
trader-sentiment/ ├─ data/ # Input datasets (Fear-Greed, Hyperliquid trades) ├─ notebooks/ # Jupyter notebooks for exploration ├─ src/ # Modular Python scripts (data_io, features, metrics, plots) ├─ reports/ # Generated plots and CSV summaries ├─ requirements.txt # Python dependencies └─ README.md # Project overview

---

## ⚙️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Prakhar07012005/trader-sentiment.git
   cd trader-sentiment
pip install -r requirements.txt
python -m src.data_io
python -m src.features
python -m src.metrics
python -m src.plots
Outputs will be saved in the reports/ folder:

perf_by_regime.csv → performance metrics by sentiment

win_rate_by_regime.png → bar chart of win rates

roi_distribution.png → ROI distribution violin plot

leverage_by_regime.png → leverage usage plot (blank if leverage column missing)

📊 Key Features
Schema-agnostic pipeline: Handles missing columns gracefully (e.g., leverage absent).

Feature engineering: ROI, win/loss flag, leverage buckets, side encoding.

Analysis: Win rate, ROI stats, Sharpe proxy, side × regime, leverage × regime.

Visuals: Clear plots for recruiter/demo-ready presentation.

Robust design: Works on large CSVs with chunked loading and Parquet conversion.

🔍 Insights (Sample)
Greed regimes show higher win rates and ROI.

Fear regimes compress returns and increase variance.

Short trades outperform during Fear/Extreme Fear.

Trade frequency spikes in Fear but win rate drops → emotional overtrading.

Leverage analysis skipped (column missing) → pipeline still runs without error.

📑 Notes
The plot leverage_by_regime.png may appear blank if leverage data is not present.

This is intentional and demonstrates the pipeline’s robustness.

📬 Submission
Send GitHub repo + summary to:

//confidential

Subject line: Junior Data Scientist – Trader Behavior Insights

🛠️ Tech Stack
Python (pandas, numpy, seaborn, matplotlib, scikit-learn)

Jupyter Notebooks

GitHub for version control

🚀 Future Work
Add predictive modeling (logistic regression / RandomForest).

Include SHAP plots for feature importance.

Extend to multi-symbol sentiment analysis.
