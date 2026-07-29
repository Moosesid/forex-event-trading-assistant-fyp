"""
FYP Dashboard — Economic Event-Driven Forex Trading Assistant
Muhammad Mursyid Bin Hassan | 2300435
Run: streamlit run app_live_corrected.py
Requires: fyp_model.pkl + all CSV files in same directory
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import date, datetime, timedelta
from PIL import Image

import json
import yfinance as yf
import cloudscraper
from bs4 import BeautifulSoup

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EUR/USD ML Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
.stApp { background-color: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
.signal-buy  { background:#0d2818; border:2px solid #2ea043; border-radius:8px; padding:24px; text-align:center; font-family:'IBM Plex Mono',monospace; font-size:36px; font-weight:600; color:#2ea043; }
.signal-sell { background:#2d0f0f; border:2px solid #f85149; border-radius:8px; padding:24px; text-align:center; font-family:'IBM Plex Mono',monospace; font-size:36px; font-weight:600; color:#f85149; }
.signal-hold { background:#161b22; border:2px solid #8b949e; border-radius:8px; padding:24px; text-align:center; font-family:'IBM Plex Mono',monospace; font-size:36px; font-weight:600; color:#8b949e; }
.price-box   { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px 20px; }
.event-row   { background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px 14px; margin:6px 0; font-size:13px; }
.event-today { border-left:3px solid #f0b429; }
.event-upcoming { border-left:3px solid #30363d; }
.finding-box { background:#161b22; border-left:3px solid #1f6feb; padding:12px 16px; border-radius:0 6px 6px 0; margin:8px 0; font-size:14px; color:#c9d1d9; }
.section-divider { border:none; border-top:1px solid #30363d; margin:32px 0; }
div[data-testid="metric-container"] { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('fyp_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_surprise_csv(path, col):
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['date'])
        return df[col].dropna()
    return pd.Series(dtype=float)

def load_img(path):
    return Image.open(path) if os.path.exists(path) else None

try:
    saved = load_model()
    model = saved['model']
    feature_cols = saved['feature_cols']

    # Older model files stored a separate scaler. The corrected Run 2
    # model stores scaling inside the sklearn Pipeline, so this is optional.
    scaler = saved.get('scaler')
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)

# ── Load results JSON ────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    if not os.path.exists('results_summary.json'):
        return None
    try:
        with open('results_summary.json', 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

# ── Signal logger ─────────────────────────────────────────────────────────────
LOG_FILE = "signal_log.csv"

def log_signal(signal, prob, price, events_today, days_since, days_until):
    """Append a signal entry to the log CSV — max once per hour."""
    try:
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d %H")
        # check if already logged this hour
        if os.path.exists(LOG_FILE):
            import csv
            with open(LOG_FILE, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("timestamp", "")[:13] == hour_key:
                        return  # already logged this hour
        row = {
            "timestamp":    now.strftime("%Y-%m-%d %H:%M:%S"),
            "date":         now.strftime("%Y-%m-%d"),
            "time":         now.strftime("%H:%M"),
            "eur_usd":      round(price, 5),
            "signal":       signal,
            "prob_up":      round(prob, 4),
            "events_today": events_today,
            "days_since":   days_since,
            "days_until":   days_until,
        }
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            writer = __import__("csv").DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass

res = load_results()

# Run 6 evaluation values, used when results_summary.json does not yet
# contain a field. JSON values take priority.
#
# Sources, all from the Run 6 report:
#   single_auc            Table 5, tuned fixed-holdout test AUC
#   rf walk-forward       Table 8, final 36-fold evaluation
#   lr / xgb walk-forward Section 6.8 grid, one-day horizon, same 36 folds
#   strategy trading      Table 10, 1bp transaction cost
RUN6_FALLBACK = {
    'buy_and_hold': {
        'gross_return': 1.81,
        'net_return': 1.80,
        'sharpe': 0.0649,
        'max_drawdown': -23.29,
        'position_changes': 1,
        # legacy key names kept so older widgets still resolve
        'single_split_return': 1.80,
        'single_split_sharpe': 0.0649,
        'single_split_max_drawdown': -23.29,
        'walk_forward_return': 1.80,
        'walk_forward_sharpe': 0.0649,
        'walk_forward_max_drawdown': -23.29,
    },
    'overlay': {
        'gross_return': 4.94,
        'net_return': 1.51,
        'sharpe': 0.0591,
        'max_drawdown': -18.81,
        'position_changes': 333,
    },
    'long_short': {
        'gross_return': -3.07,
        'net_return': -8.10,
        'sharpe': -0.1744,
        'max_drawdown': -19.65,
        'position_changes': 357,
    },
    'long_only': {
        'net_return': -6.63,
        'max_drawdown': -15.20,
    },
    'rf': {
        'single_auc': 0.5309,
        'train_auc': 0.5858,
        'wf_auc': 0.5389,
        'pooled_wf_auc': 0.5145,
        'wf_std': 0.0621,
        'folds_above_50': 26,
        'folds_above_55': 14,
        'n_folds': 36,
    },
    'lr': {
        'single_auc': 0.5071,
        'train_auc': 0.5581,
        'wf_auc': 0.5399,
        'pooled_wf_auc': 0.5281,
        'wf_std': 0.0600,
        'folds_above_50': 25,
        'n_folds': 36,
    },
    'xgb': {
        'single_auc': 0.5265,
        'train_auc': 0.8827,
        'wf_auc': 0.5075,
        'pooled_wf_auc': 0.5036,
        'wf_std': 0.0754,
        'folds_above_50': 20,
        'n_folds': 36,
    },
}

def result_section(name):
    """Return one result section, with current-run fallbacks for missing keys."""
    merged = dict(RUN6_FALLBACK.get(name, {}))
    if isinstance(res, dict) and isinstance(res.get(name), dict):
        merged.update(res[name])
    return merged

# ── Live price data ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)   # refresh every 5 minutes
def fetch_live_price():
    """Download last 300 trading days of EUR/USD for feature computation."""
    df = yf.download('EURUSD=X', period='300d', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).normalize()
    df = df.dropna()
    return df

# ── ForexFactory calendar — live scrape with file fallback ───────────────────
@st.cache_data(ttl=300)
def fetch_ff_calendar():
    """
    Scrape ForexFactory live. Falls back to ff_calendar.json if scraping fails.
    """
    today = date.today()

    def parse_events(soup):
        target_keywords = [
            'CPI m/m', 'Non-Farm Employment Change', 'Federal Funds Rate',
            'Main Refinancing Rate', 'CPI Flash Estimate', 'Core CPI Flash Estimate'
        ]
        target_currencies = {'USD', 'EUR'}
        events = []
        current_date = None
        rows = soup.find_all('tr', class_='calendar__row')
        for row in rows:
            date_cell = row.find('td', class_='calendar__date')
            if date_cell and date_cell.text.strip():
                try:
                    parsed = datetime.strptime(
                        date_cell.text.strip().replace('\n',' ').strip()[:10], '%a %b %d'
                    ).replace(year=today.year)
                    current_date = parsed.date()
                except:
                    pass
            currency_cell = row.find('td', class_='calendar__currency')
            if not currency_cell or currency_cell.text.strip() not in target_currencies:
                continue
            event_cell = row.find('td', class_='calendar__event')
            if not event_cell:
                continue
            event_name = event_cell.text.strip()
            if not any(k in event_name for k in target_keywords):
                continue
            actual   = row.find('td', class_='calendar__actual')
            forecast = row.find('td', class_='calendar__forecast')
            previous = row.find('td', class_='calendar__previous')
            time_cell = row.find('td', class_='calendar__time')
            events.append({
                'date':     current_date,
                'time':     time_cell.text.strip() if time_cell else '—',
                'currency': currency_cell.text.strip(),
                'event':    event_name,
                'forecast': forecast.text.strip() if forecast else '—',
                'previous': previous.text.strip() if previous else '—',
                'actual':   actual.text.strip() if actual else '—',
                'is_today': current_date == today if current_date else False,
                'released': bool(actual and actual.text.strip() not in ('', '—')),
            })
        return events

    # ── Try live scrape ───────────────────────────────────────────────────────
    url = "https://www.forexfactory.com/calendar?week=this"
    soup = None

    # Attempt 1: cloudscraper with Chrome fingerprint
    try:
        _scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        resp = _scraper.get(url, timeout=12)
        if resp.status_code == 200 and 'calendar' in resp.text.lower():
            soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception:
        pass

    # Attempt 2: plain requests with headers
    if soup is None:
        try:
            import requests
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
            }
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
        except Exception:
            pass

    if soup is not None:
        events = parse_events(soup)
        if events:
            # Save to file as cache for fallback
            try:
                with open('ff_calendar.json', 'w') as f:
                    json.dump({
                        'scraped_at': str(datetime.now()),
                        'source': 'live',
                        'events': [{**e, 'date': str(e['date'])} for e in events]
                    }, f, indent=2)
            except Exception:
                pass
            return events

    # ── Fallback: read from ff_calendar.json ──────────────────────────────────
    if os.path.exists('ff_calendar.json'):
        try:
            with open('ff_calendar.json', 'r') as f:
                data = json.load(f)
            events = []
            for e in data.get('events', []):
                try:
                    ev_date = date.fromisoformat(e['date']) if e['date'] else None
                except Exception:
                    ev_date = None
                events.append({
                    'date':     ev_date,
                    'time':     e.get('time', '—'),
                    'currency': e.get('currency', ''),
                    'event':    e.get('event', ''),
                    'forecast': e.get('forecast', '—'),
                    'previous': e.get('previous', '—'),
                    'actual':   e.get('actual', '—'),
                    'is_today': ev_date == today if ev_date else False,
                    'released': e.get('released', False),
                })
            return events
        except Exception:
            pass

    return []


# ── Feature engineering from live price ──────────────────────────────────────
def compute_features_from_price(
    price_df,
    event_surprises,
    event_flags,
    days_since,
    days_until
):
    """Compute the 48 corrected Run 2 features for the latest price row."""
    df = price_df.copy()
    close = df['close']

    df['log_return'] = np.log(close / close.shift(1))
    for lag in [1, 5, 7, 10, 14]:
        df[f'r{lag}d'] = close.pct_change(lag)
    for lag in [5, 7, 10]:
        df[f'mom{lag}'] = df[f'r{lag}d'] - df['r1d']

    df['vol_5d'] = df['log_return'].rolling(5).std()
    df['vol_14d'] = df['log_return'].rolling(14).std()
    df['price_pct_14d'] = (
        (close - close.rolling(14).min())
        / (close.rolling(14).max() - close.rolling(14).min() + 1e-8)
    )

    ma200 = close.rolling(200).mean()
    df['above_200ma'] = (close > ma200).astype(int)
    df['dist_200ma'] = (close - ma200) / ma200
    df['ma50_slope'] = close.rolling(50).mean().pct_change(5)

    dow = pd.to_datetime(df.index).dayofweek
    df['cos_dow'] = np.cos(2 * np.pi * dow / 5)
    df['sin_dow'] = np.sin(2 * np.pi * dow / 5)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_pct'] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-8)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - close.shift(1)).abs(),
        (df['low'] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['atr_norm'] = tr.rolling(14).mean() / close

    # Training used the rolling volatility whenever the latest observation
    # was within three days of a tracked event, including the event day.
    if days_since <= 3:
        df['post_event_vol'] = df['log_return'].rolling(3).std()
    else:
        df['post_event_vol'] = 0.0

    latest_rows = df.replace([np.inf, -np.inf], np.nan).dropna()
    if latest_rows.empty:
        raise ValueError('Not enough valid price history to construct features.')
    latest = latest_rows.iloc[-1].to_dict()

    def expanding_pct(csv_col_series, value):
        if len(csv_col_series) == 0:
            return 0.5
        historical = pd.to_numeric(csv_col_series, errors='coerce').dropna()
        if historical.empty:
            return 0.5
        return float(((historical < value).sum() + 0.5 * (historical == value).sum()) / len(historical))

    vec = {}
    csv_map = {
        'cpi_surprise': ('cpi_surprise.csv', 'cpi_surprise'),
        'nfp_surprise': ('nfp_surprise.csv', 'nfp_surprise'),
        'fomc_surprise': ('fomc_surprise.csv', 'fomc_surprise'),
        'ecb_rate_surprise': ('ecb_rate_surprise.csv', 'ecb_rate_surprise'),
        'eu_cpi_surprise': ('eu_cpi_surprise.csv', 'eu_cpi_surprise'),
        'eu_core_cpi_surprise': (
            'eu_core_cpi_surprise.csv',
            'eu_core_cpi_surprise'
        ),
    }

    for surprise_col, (csv_path, csv_col) in csv_map.items():
        day_col = surprise_col.replace('_surprise', '_day')
        available_col = surprise_col.replace('_surprise', '_available')
        pct_col = f'{surprise_col}_pct'

        is_event = int(bool(event_flags.get(surprise_col, False)))
        is_available = int(surprise_col in event_surprises)
        value = float(event_surprises.get(surprise_col, 0.0))

        historical = load_surprise_csv(csv_path, csv_col)
        percentile = expanding_pct(historical, value) if is_available else 0.5

        vec[surprise_col] = value
        vec[available_col] = is_available
        vec[pct_col] = percentile
        vec[day_col] = is_event

    vec['event_day'] = int(any(event_flags.values()))
    vec['days_since_event'] = int(days_since)
    vec['days_until_event'] = int(days_until)

    price_features = [
        'r1d', 'r5d', 'r7d', 'r10d', 'r14d',
        'mom5', 'mom7', 'mom10',
        'vol_5d', 'vol_14d', 'price_pct_14d',
        'above_200ma', 'dist_200ma', 'ma50_slope',
        'cos_dow', 'sin_dow', 'rsi', 'macd_hist',
        'bb_pct', 'atr_norm', 'post_event_vol'
    ]
    for feature in price_features:
        vec[feature] = latest.get(feature, 0.0)

    missing = [feature for feature in feature_cols if feature not in vec]
    if missing:
        raise ValueError(f'Missing live features: {missing}')

    X_input = pd.DataFrame([vec]).reindex(columns=feature_cols)
    X_input = X_input.replace([np.inf, -np.inf], np.nan)
    if X_input.isna().any().any():
        bad_columns = X_input.columns[X_input.isna().any()].tolist()
        raise ValueError(f'NaN values in live features: {bad_columns}')

    return X_input, latest, float(close.iloc[-1])

# Validation-selected thresholds from the Run 6 evaluation.
# Chosen inside each fold's 252-day validation window, never on test data.
# They are not adjustable in the dashboard.
BUY_THRESHOLD = 0.52
SELL_THRESHOLD = 0.48

def get_signal(prob, buy_t=BUY_THRESHOLD, sell_t=SELL_THRESHOLD):
    if prob >= buy_t:   return "BUY"
    if prob <= sell_t:  return "SELL"
    return "HOLD"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 EUR/USD ML Assistant")
    st.markdown("**Muhammad Mursyid Bin Hassan**")
    st.markdown("ICT3902C Capstone | 2300435")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📅 Event Calendar",
        "🎯 Live Signal",
        "🔍 Interpretability",
        "📈 Model Analysis",
        "📋 Results Summary"
    ], label_visibility="collapsed")
    st.markdown("---")
    if MODEL_LOADED:
        _pname = saved.get('model_name', type(model).__name__)
        try:
            _keys = ('max_depth', 'n_estimators', 'min_samples_leaf',
                     'max_features', 'C', 'penalty', 'solver')
            _pr = model.get_params()
            _ps = ', '.join(f"{k}={_pr[k]}" for k in _keys if k in _pr)
        except Exception:
            _ps = 'n/a'
        st.markdown(f"**Model:** {_pname}")
        st.markdown(f"**Params:** {_ps}")
    else:
        st.markdown("**Model:** not loaded")
    st.markdown(f"**Features:** {len(feature_cols) if MODEL_LOADED else 48}")
    st.markdown(f"**Refreshes:** every 5 min")
    if MODEL_LOADED:
        st.success("✓ Model loaded")
    else:
        st.error(f"✗ Model could not be loaded: {MODEL_LOAD_ERROR}")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Live Signal":
    st.markdown("# 🎯 Live EUR/USD Signal")
    st.caption(f"Auto-refreshes every 5 minutes · Last updated: {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # fetch live data
    with st.spinner("Fetching live price data..."):
        price_df = fetch_live_price()

    with st.spinner("Scraping ForexFactory calendar..."):
        ff_events = fetch_ff_calendar()

    today = date.today()

    # All tracked events today determine the event-day flags. Released
    # events with valid actual and forecast values also provide surprises.
    today_all_events = [
        event for event in ff_events
        if event['is_today']
    ]
    today_events = [
        event for event in today_all_events
        if event['released']
    ]

    event_surprises = {}
    event_flags = {
        'cpi_surprise': False,
        'nfp_surprise': False,
        'fomc_surprise': False,
        'ecb_rate_surprise': False,
        'eu_cpi_surprise': False,
        'eu_core_cpi_surprise': False,
    }

    def parse_num(value):
        """Match the scraper's training-time numeric cleaning."""
        try:
            cleaned = (
                str(value)
                .replace('%', '')
                .replace('K', '')
                .replace('M', '')
                .replace('<', '')
                .replace(',', '')
                .strip()
            )
            return float(cleaned)
        except (TypeError, ValueError):
            return None

    def event_feature_key(event):
        name = event.get('event', '')
        currency = event.get('currency', '')
        if 'CPI m/m' in name and currency == 'USD':
            return 'cpi_surprise'
        if 'Non-Farm' in name and currency == 'USD':
            return 'nfp_surprise'
        if 'Federal Funds' in name and currency == 'USD':
            return 'fomc_surprise'
        if 'Main Refinancing' in name and currency == 'EUR':
            return 'ecb_rate_surprise'
        if 'Core CPI Flash' in name and currency == 'EUR':
            return 'eu_core_cpi_surprise'
        if 'CPI Flash' in name and currency == 'EUR':
            return 'eu_cpi_surprise'
        return None

    for event in today_all_events:
        feature_key = event_feature_key(event)
        if feature_key is not None:
            # Event-day presence is independent of release/availability.
            event_flags[feature_key] = True

    for event in today_events:
        feature_key = event_feature_key(event)
        if feature_key is None:
            continue
        actual = parse_num(event.get('actual'))
        forecast = parse_num(event.get('forecast'))
        if actual is not None and forecast is not None:
            event_surprises[feature_key] = actual - forecast

    tracked_events = [
        event for event in ff_events
        if event.get('date') and event_feature_key(event) is not None
    ]
    past_dates = sorted(
        event['date'] for event in tracked_events
        if event['date'] < today and event.get('released')
    )
    future_dates = sorted(
        event['date'] for event in tracked_events
        if event['date'] > today
    )

    if any(event_flags.values()):
        # In the training data both timing features are zero on event days.
        days_since = 0
        days_until = 0
    else:
        days_since = (today - past_dates[-1]).days if past_dates else 3
        days_until = (future_dates[0] - today).days if future_dates else 5

    # compute features + signal
    col_l, col_r = st.columns([1, 1])

    with col_l:
        # price metrics
        if len(price_df) >= 2:
            curr_price  = float(price_df['close'].iloc[-1])
            prev_price  = float(price_df['close'].iloc[-2])
            day_change  = curr_price - prev_price
            day_pct     = day_change / prev_price * 100
            week_change = float(price_df['close'].iloc[-1] - price_df['close'].iloc[-6]) / float(price_df['close'].iloc[-6]) * 100

            arrow = "▲" if day_change >= 0 else "▼"
            color = "#2ea043" if day_change >= 0 else "#f85149"

            st.markdown(f"""
            <div class="price-box">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">EUR/USD</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:40px;font-weight:600;color:#e6edf3;">{curr_price:.5f}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:{color};">{arrow} {day_change:+.5f} ({day_pct:+.2f}%) today</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#8b949e;margin-top:4px;">Week: {week_change:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Today's Events")
        if today_events:
            for ev in today_events:
                surp_key = None
                if 'CPI m/m' in ev['event'] and ev['currency'] == 'USD': surp_key = 'cpi_surprise'
                elif 'Non-Farm' in ev['event']:  surp_key = 'nfp_surprise'
                elif 'Federal Funds' in ev['event']: surp_key = 'fomc_surprise'
                elif 'Main Refinancing' in ev['event']: surp_key = 'ecb_rate_surprise'
                elif 'Core CPI Flash' in ev['event']: surp_key = 'eu_core_cpi_surprise'
                elif 'CPI Flash' in ev['event']: surp_key = 'eu_cpi_surprise'

                surp_val = event_surprises.get(surp_key, None) if surp_key else None
                surp_str = f"Surprise: {surp_val:+.3f}" if surp_val is not None else ""
                surp_color = "#2ea043" if surp_val and surp_val > 0 else "#f85149" if surp_val and surp_val < 0 else "#8b949e"

                st.markdown(f"""
                <div class="event-row event-today">
                    <strong style="color:#f0b429">{ev['currency']}</strong>
                    &nbsp;{ev['event']}&nbsp;
                    <span style="color:#8b949e">A: {ev['actual']} | F: {ev['forecast']} | P: {ev['previous']}</span>
                    <span style="color:{surp_color};margin-left:8px;font-family:'IBM Plex Mono',monospace">{surp_str}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No tracked high-impact events released today.")

        # upcoming events
        upcoming = [e for e in ff_events if not e['is_today'] and e['date'] and e['date'] >= today]
        if upcoming:
            st.markdown("#### Upcoming This Week")
            for ev in upcoming[:6]:
                st.markdown(f"""
                <div class="event-row event-upcoming">
                    <strong style="color:#8b949e">{ev['date'].strftime('%a %d') if ev['date'] else '—'}</strong>
                    &nbsp;<span style="color:#1f6feb">{ev['currency']}</span>
                    &nbsp;{ev['event']}&nbsp;
                    <span style="color:#8b949e">F: {ev['forecast']} | P: {ev['previous']}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_r:
        buy_t = BUY_THRESHOLD
        sell_t = SELL_THRESHOLD

        st.caption(
            f"Fixed signal thresholds: SELL ≤ {sell_t:.2f} · "
            f"BUY ≥ {buy_t:.2f}"
        )

        X_input = None
        feature_error = None

        if MODEL_LOADED and len(price_df) > 200:
            try:
                X_input, latest_feats, curr_close = compute_features_from_price(
                    price_df,
                    event_surprises,
                    event_flags,
                    days_since,
                    days_until
                )

                # The corrected Run 2 model is a Pipeline containing its scaler.
                # Older model files with a separate scaler remain supported.
                if hasattr(model, 'named_steps'):
                    model_input = X_input
                elif scaler is not None:
                    model_input = scaler.transform(X_input)
                else:
                    model_input = X_input

                prob = float(model.predict_proba(model_input)[0, 1])
                signal = get_signal(prob)
                log_signal(
                    signal, prob, curr_close,
                    len(today_events), days_since, days_until
                )
            except Exception as error:
                feature_error = str(error)
                st.error(f"Feature computation error: {feature_error}")
                prob, signal = 0.5, "HOLD"
        else:
            prob, signal = 0.5, "HOLD"

        css_class = {'BUY':'signal-buy','SELL':'signal-sell','HOLD':'signal-hold'}[signal]
        emoji     = {'BUY':'▲','SELL':'▼','HOLD':'—'}[signal]

        st.markdown(f'<div class="{css_class}">{emoji} {signal}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-top:12px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">P(UP) — Model Probability</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:32px;font-weight:600;color:#e6edf3;">{prob:.4f}</div>
        </div>""", unsafe_allow_html=True)

        bar_color = "#2ea043" if prob >= buy_t else "#f85149" if prob <= sell_t else "#8b949e"
        st.markdown(f"""
        <div style="background:#30363d;border-radius:4px;height:14px;margin:10px 0 4px 0;">
            <div style="background:{bar_color};width:{prob*100:.1f}%;height:100%;border-radius:4px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;">
            <span>SELL ≤{sell_t}</span><span>HOLD</span><span>BUY ≥{buy_t}</span>
        </div>""", unsafe_allow_html=True)

        if signal == "BUY":
            st.success("EUR expected to strengthen. Model confidence sufficient for LONG.")
        elif signal == "SELL":
            st.error("USD expected to strengthen. Model confidence sufficient for SHORT.")
        else:
            st.info("Insufficient directional conviction. Stay flat.")

        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;margin-top:12px;">
        Events today: {len(today_events)} released &nbsp;|&nbsp; 
        Days since event: {days_since} &nbsp;|&nbsp; Days until next: {days_until}
        </div>""", unsafe_allow_html=True)

        with st.expander("View feature vector"):
            if X_input is not None:
                st.dataframe(
                    X_input.T.rename(columns={0: 'value'}).round(6),
                    use_container_width=True
                )
            elif feature_error:
                st.warning("Feature vector unavailable because live feature construction failed.")
            elif not MODEL_LOADED:
                st.warning("Feature vector unavailable because the model is not loaded.")
            else:
                st.info("Feature vector will appear after sufficient live price data is loaded.")

        st.caption("⚠️ Demo only. Max 0.1 lot. Not financial advice.")

    # auto-refresh
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Dashboard auto-refreshes every 5 minutes via Streamlit cache TTL.")

    # ── Signal log viewer ─────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### 📋 Signal Log")
    st.caption("Every signal generated since the dashboard started running is recorded here.")
    if os.path.exists(LOG_FILE):
        try:
            log_df = pd.read_csv(LOG_FILE)
            log_df = log_df.sort_values("timestamp", ascending=False).head(50)
            def colour_signal(val):
                if val == "BUY":  return "color: #2ea043; font-weight: bold"
                if val == "SELL": return "color: #f85149; font-weight: bold"
                return "color: #8b949e"
            styled_log = log_df.style.applymap(colour_signal, subset=["signal"])
            st.dataframe(styled_log, hide_index=True, use_container_width=True)
            st.caption(f"Showing last {min(50, len(log_df))} entries from {LOG_FILE}")
        except Exception as e:
            st.warning(f"Could not read log: {e}")
    else:
        st.info("No signals logged yet. Signal log will appear here after the first refresh.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EVENT CALENDAR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Event Calendar":
    st.markdown("# 📅 This Week's Event Calendar")
    # show source info
    if os.path.exists('ff_calendar.json'):
        try:
            with open('ff_calendar.json') as f:
                meta = json.load(f)
            source = meta.get('source', 'file')
            scraped_at = meta.get('scraped_at', 'unknown')
            if source == 'live':
                st.caption(f"High-impact USD and EUR events only · Live scraped from ForexFactory · Last updated: {scraped_at}")
            else:
                st.caption(f"High-impact USD and EUR events only · Loaded from saved file · Last scraped: {scraped_at}")
        except Exception:
            st.caption("High-impact USD and EUR events only · ForexFactory")
    else:
        st.caption("High-impact USD and EUR events only · ForexFactory")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    with st.spinner("Loading ForexFactory calendar..."):
        ff_events = fetch_ff_calendar()

    if not ff_events:
        st.warning("No events found in ff_calendar.json. Run the scraper cell in your notebook to refresh.")
    else:
        today = date.today()
        event_df = pd.DataFrame(ff_events)
        event_df['Date']     = event_df['date'].apply(lambda d: d.strftime('%a %d %b') if d else '—')
        event_df['Currency'] = event_df['currency']
        event_df['Event']    = event_df['event']
        event_df['Time']     = event_df['time']
        event_df['Forecast'] = event_df['forecast']
        event_df['Previous'] = event_df['previous']
        event_df['Actual']   = event_df['actual']
        event_df['Status']   = event_df.apply(
            lambda r: '🟡 TODAY' if r['is_today'] else ('✅ Released' if r['released'] else '⏳ Pending'), axis=1
        )
        st.dataframe(
            event_df[['Date','Time','Currency','Event','Forecast','Previous','Actual','Status']],
            hide_index=True, use_container_width=True
        )

        st.markdown("#### Surprise Summary (Today)")
        today_released = [e for e in ff_events if e['is_today'] and e['released']]
        if today_released:
            for ev in today_released:
                act  = ev['actual']
                fore = ev['forecast']
                try:
                    surp = float(str(act).replace('%','').replace('K','')) - float(str(fore).replace('%','').replace('K',''))
                    color = "#2ea043" if surp > 0 else "#f85149" if surp < 0 else "#8b949e"
                    label = "BEAT" if surp > 0 else "MISS" if surp < 0 else "IN LINE"
                    st.markdown(f"""
                    <div class="event-row event-today">
                        <strong style="color:{color}">{label}</strong> &nbsp;
                        {ev['currency']} {ev['event']} — surprise: 
                        <span style="color:{color};font-family:'IBM Plex Mono',monospace">{surp:+.3f}</span>
                    </div>""", unsafe_allow_html=True)
                except:
                    pass
        else:
            st.info("No events released today.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Analysis":
    st.markdown("# 📈 Model Analysis")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    lr = result_section('lr')
    rf = result_section('rf')
    xgb = result_section('xgb')
    bah = result_section('buy_and_hold')

    st.markdown("### Fixed-Holdout Diagnostic (bias-variance check)")
    st.caption(
        "This split is a diagnostic, not a selection instrument. The gap column "
        "is the informative one."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Random Forest",
        f"{rf['single_auc']:.4f}",
        f"train {rf['train_auc']:.4f} · gap {rf['train_auc'] - rf['single_auc']:.4f}"
    )
    c2.metric(
        "Logistic Regression",
        f"{lr['single_auc']:.4f}",
        f"train {lr['train_auc']:.4f} · gap {lr['train_auc'] - lr['single_auc']:.4f}"
    )
    c3.metric(
        "XGBoost",
        f"{xgb['single_auc']:.4f}",
        f"train {xgb['train_auc']:.4f} · gap {xgb['train_auc'] - xgb['single_auc']:.4f}"
    )

    st.markdown("### Walk-Forward Validation (36 folds, Jul 2017 – Oct 2025)")
    st.caption(
        "Random Forest on the hybrid feature set, ten-year rolling training "
        "window, purge gap at every fold boundary, 2,160 out-of-sample days."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Fold AUC", f"{rf['wf_auc']:.4f}")
    c2.metric("Pooled AUC", f"{rf['pooled_wf_auc']:.4f}")
    c3.metric(
        "Folds > Random",
        f"{rf['folds_above_50']} / {rf['n_folds']}"
    )
    c4.metric(
        "Folds > 0.55",
        f"{rf['folds_above_55']} / {rf['n_folds']}"
    )
    st.caption(
        "Fold-level test against 0.50: t = 3.754, one-sided p = 0.000316, "
        "Bonferroni-corrected p = 0.009484 across the 30 configurations examined."
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Walk-Forward Equity Curves")
        img = load_img("run6_equity_curves.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("Run Section 8 in the notebook to create this figure.")
    with col2:
        st.markdown("#### Walk-Forward Drawdowns")
        img = load_img("run6_drawdowns.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("Run Section 8 in the notebook to create this figure.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Evaluation Diagnostics")
    img = load_img("evaluation_diagnostics.png")
    if img:
        st.image(img, use_container_width=True)
    else:
        st.info("Run Section 6 in the notebook to create this figure.")

    st.markdown("### Model Comparison")
    st.caption(
        "Walk-forward figures are computed on the same 36 folds for all three "
        "models. Trading results are reported by strategy rather than by model, "
        "because a single realised return path does not measure predictive skill."
    )
    comparison = pd.DataFrame({
        'Model': ['Random Forest \u2605', 'Logistic Regression', 'XGBoost'],
        'Holdout AUC': [rf['single_auc'], lr['single_auc'], xgb['single_auc']],
        'Train AUC': [rf['train_auc'], lr['train_auc'], xgb['train_auc']],
        'Overfit Gap': [
            round(rf['train_auc'] - rf['single_auc'], 4),
            round(lr['train_auc'] - lr['single_auc'], 4),
            round(xgb['train_auc'] - xgb['single_auc'], 4),
        ],
        'WF Mean AUC': [rf['wf_auc'], lr['wf_auc'], xgb['wf_auc']],
        'WF Pooled AUC': [
            rf['pooled_wf_auc'], lr['pooled_wf_auc'], xgb['pooled_wf_auc']
        ],
        'WF AUC Std': [rf['wf_std'], lr['wf_std'], xgb['wf_std']],
        'Folds > 0.50': [
            f"{rf['folds_above_50']} / {rf['n_folds']}",
            f"{lr['folds_above_50']} / {lr['n_folds']}",
            f"{xgb['folds_above_50']} / {xgb['n_folds']}",
        ],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption(
        "\u2605 Random Forest on the hybrid feature set was selected on walk-forward "
        "performance during the experimental sequence, not on the final test result. "
        "Random Forest and Logistic Regression differ by 0.0013 in mean fold AUC "
        "against a fold standard deviation of 0.06 and are not meaningfully separable. "
        "XGBoost shows the largest train-test gap, the signature of capacity spent "
        "on memorisation."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Interpretability":
    st.markdown("# 🔍 Feature Interpretability")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Permutation Importance", "SHAP Analysis", "Ablation Study"])

    with tab1:
        st.markdown("#### Permutation Importance — Random Forest")
        st.caption("Drop in AUC when each feature is randomly shuffled. Higher = more critical.")
        img = load_img("permutation_importance_rf.png")
        if img:
            st.image(img, width=700)
        else:
            st.info("Run Section 9 in notebook.")

        st.markdown("#### How to read this chart")
        st.markdown("""<div class="finding-box">
            <strong>X-axis — AUC Drop</strong><br>
            Each bar shows how much the model's accuracy (AUC) drops when that feature's values are randomly shuffled.
            A larger drop means the model relies heavily on that feature to make correct predictions.
            A near-zero or negative bar means the feature adds little or no value.
        </div>""", unsafe_allow_html=True)

        for feat, desc in [
            ("nfp_surprise_pct", "Highest permutation importance in the corrected run; the percentile rank of the latest NFP surprise."),
            ("r7d", "Seven-day return, representing medium-short price direction."),
            ("cpi_surprise", "Raw US CPI actual-minus-forecast surprise."),
            ("days_since_event", "Number of days since the latest tracked event."),
            ("r5d", "Five-day return, capturing recent price movement."),
        ]:
            st.markdown(f"""<div class="finding-box">
                <strong style="font-family:'IBM Plex Mono',monospace">{feat}</strong><br>
                {desc}
            </div>""", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### SHAP Summary Plot")
            st.caption("Each dot = one prediction. Color = feature value. X = SHAP impact.")
            img = load_img("shap_summary_rf.png")
            if img:
                st.image(img, width=700)
            else:
                st.info("Run Section 9 in notebook.")
        with c2:
            st.markdown("#### SHAP Bar Chart")
            st.caption("Mean absolute SHAP value — overall importance ranking.")
            img = load_img("shap_bar_rf.png")
            if img:
                st.image(img, width=700)
            else:
                st.info("Run Section 9 in notebook.")

        st.markdown("#### How to read these charts")
        st.markdown("""<div class="finding-box">
            <strong>Beeswarm plot (left)</strong><br>
            Each dot is one prediction. The X-axis shows how much that feature pushed the model toward UP (positive)
            or DOWN (negative). Color shows the feature value — red = high value, blue = low value.
            Dots clustered near zero mean the feature had little impact on that prediction.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="finding-box">
            <strong>Bar chart (right)</strong><br>
            Average absolute SHAP value per feature across all predictions.
            This gives an overall ranking of which features consistently drive the model's output magnitude,
            regardless of direction.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="finding-box">
            <strong>Key takeaway</strong><br>
            The corrected interpretation is mixed rather than dominated by one family.
            Permutation importance ranks <code>nfp_surprise_pct</code> first, followed by
            short-horizon returns and CPI/event-timing variables. SHAP should be read as
            prediction-level contribution, not as proof that a feature group improves
            out-of-sample AUC.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Ablation Study \u2014 Feature Group Contribution")
        st.caption(
            "Group-level ablation measured across 65 common walk-forward folds. "
            "Each row retrains the model with one feature family removed: "
            "'Technical only' ablates the event features, 'Event only' ablates "
            "market context, and 'Hybrid' retains both. This supersedes the earlier "
            "single-split ablation, whose effect sizes were smaller than the "
            "fold-to-fold standard deviation."
        )

        feature_sets = pd.DataFrame({
            'Configuration': ['Technical only (events ablated)',
                              'Event only (market context ablated)',
                              'Hybrid (nothing ablated)'],
            'Logistic Regression': [0.5115, 0.5105, 0.5215],
            'Random Forest': [0.5123, 0.5064, 0.5298],
            'XGBoost': [0.4956, 0.5055, 0.5130],
        })
        st.dataframe(feature_sets, hide_index=True, use_container_width=True)

        st.caption(
            "The hybrid set is strongest for all three models. Reading down the "
            "columns rather than across the rows is the point: the result is about "
            "feature sets, not about which model wins."
        )

        st.markdown("#### Ablation Significance")
        st.caption(
            "Because every configuration uses the same 65 folds, the ablation can "
            "be tested paired rather than as a comparison of means. Random Forest "
            "shown. A significant positive difference means the ablated group was "
            "carrying real information."
        )

        paired = pd.DataFrame({
            'Ablation': ['Event features removed', 'Market context removed'],
            'AUC lost': ['-0.0176', '-0.0235'],
            't': [2.979, 2.429],
            'p (two-sided)': [0.0041, 0.0180],
            'Hybrid higher in': ['44 / 65 folds', '40 / 65 folds'],
            'Verdict': ['Group carries signal', 'Group carries signal'],
        })
        st.dataframe(paired, hide_index=True, use_container_width=True)

        for text in [
            "Neither event data nor market context alone is sufficient. The "
            "combination is measurably better than either in isolation, and both "
            "differences are statistically significant.",

            "Event features contribute in combination rather than in isolation. "
            "The event-only set reaches 0.5064 for Random Forest, barely "
            "distinguishable from chance, yet adding those features to "
            "market-context variables improves walk-forward AUC by 0.0176.",

            "The interpretation is that event information is conditioning rather "
            "than driving: it is useful in the context of prevailing momentum and "
            "regime, rather than as a standalone predictor.",
        ]:
            st.markdown(
                f'<div class="finding-box">{text}</div>',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Results Summary":
    st.markdown("# 📋 Results Summary")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    lr = result_section('lr')
    rf = result_section('rf')
    xgb = result_section('xgb')
    bah = result_section('buy_and_hold')
    ovl = result_section('overlay')
    lsh = result_section('long_short')
    lon = result_section('long_only')

    st.markdown("""
    **Economic Event-Driven Forex Trading Assistant using Machine Learning**  
    The Run 6 evaluation finds a small but statistically detectable directional
    signal in daily EUR/USD. The clearest benefit is risk-related rather than
    return-related: the overlay reduces maximum drawdown at comparable total
    return. The system is a decision-support and risk-filtering tool, not an
    autonomous trading strategy.
    """)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### Key Findings")
    findings = [
        (
            f"Random Forest on the hybrid feature set was selected on walk-forward "
            f"performance during the experimental sequence, not on the final test set."
        ),
        (
            f"Across {rf['n_folds']} non-overlapping folds covering 2,160 "
            f"out-of-sample days, mean fold AUC was {rf['wf_auc']:.4f} with a "
            f"standard deviation of {rf['wf_std']:.4f}, pooled AUC was "
            f"{rf['pooled_wf_auc']:.4f}, and {rf['folds_above_50']}/"
            f"{rf['n_folds']} folds exceeded 0.50."
        ),
        (
            "The fold-level test against 0.50 gives t = 3.754 and a one-sided "
            "p-value of 0.000316, which survives Bonferroni correction across "
            "the 30 configurations examined."
        ),
        (
            "A hybrid of event and market-context features beats either family "
            "in isolation: +0.0176 over technical-only (p = 0.0041) and +0.0235 "
            "over event-only (p = 0.0180) on paired fold-level tests."
        ),
        (
            f"The Buy-and-Hold plus overlay strategy reduced maximum drawdown "
            f"from {bah['max_drawdown']:.2f}% to {ovl['max_drawdown']:.2f}% at a "
            f"comparable net return of {ovl['net_return']:+.2f}% against "
            f"{bah['net_return']:+.2f}%."
        ),
        (
            "Predictive accuracy declines as the prediction horizon lengthens, "
            "consistently across all three model families, which is what rapid "
            "absorption of announcement information predicts."
        ),
        (
            "Hyperparameter tuning is not driving the result: folds outside the "
            "tuning window scored higher (0.5452) than folds inside it (0.5343)."
        ),
    ]
    for finding in findings:
        st.markdown(
            f'<div class="finding-box">{finding}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### Strategy Comparison")
    st.caption(
        "Walk-forward trading performance at 1 basis point transaction cost, "
        "2,160 out-of-sample days."
    )
    benchmark_table = pd.DataFrame({
        'Strategy': [
            'Buy and Hold',
            'Buy and Hold + ML overlay',
            'ML long-short',
            'ML long-only',
        ],
        'Net Return': [
            f"{bah['net_return']:+.2f}%",
            f"{ovl['net_return']:+.2f}%",
            f"{lsh['net_return']:+.2f}%",
            f"{lon['net_return']:+.2f}%",
        ],
        'Sharpe': [
            f"{bah['sharpe']:.4f}",
            f"{ovl['sharpe']:.4f}",
            f"{lsh['sharpe']:.4f}",
            "n/a",
        ],
        'Max Drawdown': [
            f"{bah['max_drawdown']:.2f}%",
            f"{ovl['max_drawdown']:.2f}%",
            f"{lsh['max_drawdown']:.2f}%",
            f"{lon['max_drawdown']:.2f}%",
        ],
        'Position Changes': [
            bah['position_changes'],
            ovl['position_changes'],
            lsh['position_changes'],
            "n/a",
        ],
    })
    st.dataframe(benchmark_table, hide_index=True, use_container_width=True)

    st.markdown("### Limitations")
    st.markdown("""
    - Daily EUR/USD direction remains only weakly predictable; the edge is
      statistically detectable but small.
    - Announcement effects are largely intraday, so aggregation to a daily
      target discards most of the reaction.
    - The fold-level test assumes independent folds; consecutive ten-year
      training windows share approximately 97.6% of their rows.
    - The 0.52/0.48 thresholds were selected inside each fold's validation window, never on test data.
    - The live dashboard depends on third-party market and calendar data and is a demonstration only.
    """)

    st.markdown("### Future Work")
    st.markdown("""
    - Validate thresholds inside a dedicated rolling validation window.
    - Move to an intraday horizon, which requires a price feed and an event
      calendar that both carry release times.
    - Add regime-specific models and transaction-cost stress testing.
    - Integrate MT5 only with strict demo-account safety controls.
    """)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.caption(
        "ICT3902C Capstone | Muhammad Mursyid Bin Hassan | 2300435 | "
        "Supervisor: Dr Li Xiaorong"
    )
