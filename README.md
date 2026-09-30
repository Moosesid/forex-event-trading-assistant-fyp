# Economic Event-Driven Forex Trading Assistant

My final year project for the BSc Applied Computing at Singapore Institute of Technology.

## Overview

This project collects economic announcement data (like US inflation, jobs reports and central bank interest rate decisions), uses it to predict whether the EUR/USD exchange rate will go up or down the next day, and shows the prediction on a live dashboard.

The main focus is the data pipeline: collecting 19 years of data, cleaning it, making sure the model only uses information it would have had at the time, and keeping everything updated automatically.

Streamlit Link: https://forex-event-trading-assistant-fyp-nvpztjudusmnwdkjirmgee.streamlit.app/

## How it works

**1. Data collection**
A Python scraper pulls economic announcements from ForexFactory from 2007 to 2025. Six event types are covered: US CPI (inflation), US Non-Farm Payrolls (jobs), FOMC (US interest rates), ECB (Euro interest rates), EU CPI and EU Core CPI. For each announcement it saves the expected value (the forecast), the actual value, and the previous value.

**2. Combining the data**
The announcements are matched to daily EUR/USD prices by date. After cleaning, this gives 4,727 days of data with 48 input features.

**3. Feature engineering**
The most important feature is the **surprise**: the difference between the actual number and what was expected. Markets usually react to surprises, not to the number itself. To keep surprises comparable across different events, each one is ranked against that event's own past releases only, never future ones. Price trend and market condition features are added alongside.

**4. Model training and testing (walk-forward validation)**
Three models are compared: Logistic Regression, Random Forest and XGBoost. Instead of one train/test split, the data is split into 36 time periods. The model trains on the past, predicts the next period, then moves forward and repeats. A gap is left between training and testing data so nothing overlaps. This mirrors how the model would actually be used day to day.

**5. Dashboard**
A Streamlit web app with five pages: upcoming events, the current BUY / SELL / HOLD signal, which features drove the prediction, model results, and a summary.

**6. Automated data pipeline**
A GitHub Actions job runs on a schedule, scrapes the latest event calendar, and commits it to the repo as `ff_calendar.json`. The dashboard reads this file, so it stays up to date without anything running manually.

## Problems I solved

**The data source blocked the dashboard.** ForexFactory blocks requests from Streamlit's cloud servers. I moved the scraping into GitHub Actions and had the dashboard read the saved file instead of scraping directly.

**Data leakage.** Early versions accidentally let the model use future information, which made results look better than they really were. Examples: scaling data using statistics from the whole dataset (including the future), and picking the best model based on test results. I found and fixed these, and the honest results came out lower but reliable.

**Missing data.** Some announcements had no forecast value. Dropping them would lose real events, so I kept them and added "available" flags so the model can tell the difference between "no surprise" and "no data".

**Training and live data matching.** The dashboard builds its inputs using the same code as the training notebook, so the live predictions are based on exactly the same features the model learned from.

## Results

- The model beat random guessing in most test periods and the result was statistically significant, but the edge is small. This is expected, since daily currency moves are very hard to predict.
- Combining economic surprises with price trends worked better than using either on its own (confirmed with statistical tests across test periods).
- Used as a filter on top of simply holding the currency, it reduced the largest losses, but trading costs quickly eat into the benefit.

Conclusion: the system works best as a decision-support tool to help a trader manage risk around announcements, not as an automatic trading bot. Full results are in the final report in this repo.

## Tools used

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, GitHub Actions, yfinance

## How to run it

```bash
git clone https://github.com/Moosesid/forex-event-trading-assistant-fyp.git
cd forex-event-trading-assistant-fyp
pip install -r requirements.txt
streamlit run app_live.py
```

## Main files

```
app_live.py                                the dashboard
scrape_calendar.py                         scraper that GitHub Actions runs on a schedule
.github/workflows/                         the scheduled job
ff_calendar.json                           latest scraped events, read by the dashboard
fyp_model.pkl                              trained model used by the dashboard
FYP_Main_Final_Run7 Grid.ipynb             main notebook (data prep, training, testing)
ForexFactory_Scrape_final2007_run3.ipynb   scraper for the 2007 to 2025 history
economic_events_master_2007_2025.csv       all collected event data
*_surprise.csv                             surprise data for each event type
run*_*.csv and *.png                       results and charts from each experiment
```
