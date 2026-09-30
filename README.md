# Economic Event-Driven Forex Trading Assistant

Final year capstone project (BSc Applied Computing, Singapore Institute of Technology).

An end-to-end ML pipeline that predicts daily EUR/USD direction from macroeconomic event surprises, with a live Streamlit dashboard fed by an automated scraping job.

**Live dashboard:** https://forex-event-trading-assistant-fyp-nvpztjudusmnwdkjirmgee.streamlit.app/

## What it does

- Scrapes 19 years (2007 to 2025) of macroeconomic event data from ForexFactory: actual, forecast and previous values for US CPI, NFP, FOMC, ECB rate decisions, EU CPI and EU Core CPI
- Merges event data with daily EUR/USD prices into 4,727 modelling rows and 48 features
- Engineers surprise features (actual minus consensus, normalised against past releases only) alongside technical and regime features
- Trains and evaluates Logistic Regression, Random Forest and XGBoost using purged walk-forward validation
- Serves live BUY / SELL / HOLD signals on a 5-page Streamlit dashboard, refreshed hourly by GitHub Actions

## Architecture

```
ForexFactory scraper ──┐
                       ├──> merge + feature engineering ──> walk-forward training ──> model pickle
yfinance EUR/USD ──────┘                                                                   │
                                                                                           v
GitHub Actions (hourly) ──> ff_calendar.json ──> live feature builder ──> Streamlit dashboard
```

The live feature builder and the offline training path share one feature implementation, so there is no train/serve skew.

## Data engineering decisions

- **Leakage-safe evaluation:** 36 non-overlapping walk-forward folds with a purge gap, hyperparameter tuning inside training windows only, and signal thresholds chosen on validation data only
- **Past-only normalisation:** surprise percentiles use only prior releases, with a 12-release minimum before a rank is produced
- **Missing data handled explicitly:** event dates are kept even when the forecast is missing, with `*_available` flags so the model can tell "no surprise" from "no data"
- **Bugs found and fixed:** look-ahead bias in quantile clipping, a test-set model-selection leak, and inconsistent lagged-return definitions from earlier iterations
- **Blocked upstream source:** ForexFactory blocks Streamlit Cloud IPs, so scraping runs on GitHub Actions and commits a JSON snapshot the app reads

## Results (summary)

Daily FX direction is close to a random walk, so the goal was to measure how much signal macro surprises actually carry, honestly.

- Mean walk-forward fold AUC [add final figure] across 36 folds, above random in [x/36] folds
- Hybrid (event + technical) features beat either set alone, confirmed with paired t-tests across folds
- As a trading filter layered on buy-and-hold, the model reduced maximum drawdown, but the edge is sensitive to transaction costs

Conclusion: the signal is statistically detectable but modest, so the system is framed as a decision-support and risk filter, not an autonomous trading bot.

## Dashboard pages

1. **Event Calendar** - upcoming and recent macro releases
2. **Live Signal** - current model probability and BUY / SELL / HOLD
3. **Interpretability** - permutation importance and SHAP
4. **Model Analysis** - walk-forward results and feature-set comparison
5. **Results Summary** - headline metrics and backtest

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, GitHub Actions, cloudscraper, yfinance

## Run locally

```bash
git clone https://github.com/Moosesid/forex-event-trading-assistant-fyp.git
cd forex-event-trading-assistant-fyp
pip install -r requirements.txt
streamlit run app_live.py
```

## Repo structure

```
[update to match your actual files]
app_live.py                 Streamlit dashboard
fyp_model.pkl               trained model
results_summary.json        metrics shown on the dashboard
ff_calendar.json            latest scraped events (updated hourly)
.github/workflows/          hourly scrape job
notebooks/                  scraping, feature engineering and evaluation
```

## Limitations and future work

- Actual-minus-consensus is a rough proxy for policy surprise; market-implied measures (e.g. fed funds futures) would be stronger
- Daily horizon dilutes event impact; intraday data would test this directly
- Walk-forward fold loops could be refactored into one parameterised function
