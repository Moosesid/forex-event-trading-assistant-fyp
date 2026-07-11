"""
FYP Dashboard — Economic Event-Driven Forex Trading Assistant
Muhammad Mursyid Bin Hassan | 2300435
Run: streamlit run app.py
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
    saved       = load_model()
    model       = saved['model']
    scaler      = saved['scaler']
    feature_cols = saved['feature_cols']
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False

# ── Load results JSON ────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    if os.path.exists('results_summary.json'):
        with open('results_summary.json', 'r') as f:
            return json.load(f)
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
def compute_features_from_price(price_df, event_surprises, days_since, days_until, any_event):
    """Compute all 42 features from live price dataframe + event inputs."""
    df = price_df.copy()
    close = df['close']

    df['log_return'] = np.log(close / close.shift(1))
    for lag in [1, 5, 7, 10, 14]:
        df[f'r{lag}d'] = close.pct_change(lag)
    for lag in [5, 7, 10]:
        df[f'mom{lag}'] = df[f'r{lag}d'] - df['r1d']

    df['vol_5d']       = df['log_return'].rolling(5).std()
    df['vol_14d']      = df['log_return'].rolling(14).std()
    df['price_pct_14d'] = (
        (close - close.rolling(14).min()) /
        (close.rolling(14).max() - close.rolling(14).min() + 1e-8)
    )

    df['above_200ma'] = (close > close.rolling(200).mean()).astype(int)
    df['dist_200ma']  = (close - close.rolling(200).mean()) / close.rolling(200).mean()
    df['ma50_slope']  = close.rolling(50).mean().pct_change(5)

    dow = pd.to_datetime(df.index).dayofweek
    df['cos_dow'] = np.cos(2 * np.pi * dow / 5)
    df['sin_dow'] = np.sin(2 * np.pi * dow / 5)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_pct'] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-8)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - close.shift(1)).abs(),
        (df['low']  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_norm'] = tr.rolling(14).mean() / close

    if days_since <= 3 and any_event:
        df['post_event_vol'] = df['log_return'].rolling(3).std()
    else:
        df['post_event_vol'] = 0.0

    # get latest row
    latest = df.dropna().iloc[-1].to_dict()

    # inject event features
    def expanding_pct(csv_col_series, val):
        if len(csv_col_series) == 0:
            return 0.5
        all_vals = pd.Series(list(csv_col_series) + [val])
        return float(all_vals.rank(pct=True).iloc[-1])

    vec = {}
    csv_map = {
        'cpi':      ('cpi_surprise.csv',          'cpi_surprise'),
        'nfp':      ('nfp_surprise.csv',           'nfp_surprise'),
        'fomc':     ('fomc_surprise.csv',          'fomc_surprise'),
        'ecb_rate': ('ecb_rate_surprise.csv',      'ecb_rate_surprise'),
        'eu_cpi':   ('eu_cpi_surprise.csv',        'eu_cpi_surprise'),
        'eu_core':  ('eu_core_cpi_surprise.csv',   'eu_core_cpi_surprise'),
    }

    event_keys = ['cpi', 'nfp', 'fomc', 'ecb_rate', 'eu_cpi', 'eu_core']
    col_names  = ['cpi_surprise', 'nfp_surprise', 'fomc_surprise',
                  'ecb_rate_surprise', 'eu_cpi_surprise', 'eu_core_cpi_surprise']

    for key, col in zip(event_keys, col_names):
        val = event_surprises.get(col, 0)
        is_event = val != 0
        hist = load_surprise_csv(csv_map[key][0], col)
        pct  = expanding_pct(hist, val) if is_event else 0.5

        vec[col]               = val
        vec[col.replace('_surprise', '_day')] = int(is_event)
        vec[col + '_pct']      = pct

    vec['event_day']        = int(any_event)
    vec['days_since_event'] = days_since
    vec['days_until_event'] = days_until

    # price-derived features from latest row
    for f in ['r1d','r5d','r7d','r10d','r14d','mom5','mom7','mom10',
              'vol_5d','vol_14d','price_pct_14d','above_200ma','dist_200ma',
              'ma50_slope','cos_dow','sin_dow','rsi','macd_hist','bb_pct',
              'atr_norm','post_event_vol']:
        vec[f] = latest.get(f, 0)

    return pd.DataFrame([vec])[feature_cols], latest, close.iloc[-1]

def get_signal(prob, buy_t, sell_t):
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
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Params:** C=0.001, L2, saga")
    st.markdown("**Features:** 42")
    st.markdown(f"**Refreshes:** every 5 min")
    if MODEL_LOADED:
        st.success("✓ Model loaded")
    else:
        st.error("✗ fyp_model.pkl not found")
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

    # today's released events → compute surprises
    today_events = [e for e in ff_events if e['is_today'] and e['released']]

    event_surprises = {}
    any_event = len(today_events) > 0

    def parse_num(s):
        try:
            return float(str(s).replace('%','').replace('K','').replace('M','').strip())
        except:
            return None

    for ev in today_events:
        act  = parse_num(ev['actual'])
        fore = parse_num(ev['forecast'])
        if act is not None and fore is not None:
            surprise = act - fore
            name = ev['event']
            if 'CPI m/m' in name and ev['currency'] == 'USD':
                event_surprises['cpi_surprise'] = surprise
            elif 'Non-Farm' in name:
                event_surprises['nfp_surprise'] = surprise
            elif 'Federal Funds' in name:
                event_surprises['fomc_surprise'] = surprise
            elif 'Main Refinancing' in name:
                event_surprises['ecb_rate_surprise'] = surprise
            elif 'Core CPI Flash' in name:
                event_surprises['eu_core_cpi_surprise'] = surprise
            elif 'CPI Flash' in name:
                event_surprises['eu_cpi_surprise'] = surprise

    # days since/until event from FF calendar
    past_events   = [e for e in ff_events if e['date'] and e['date'] < today and e['released']]
    future_events = [e for e in ff_events if e['date'] and e['date'] > today]
    days_since = (today - past_events[-1]['date']).days   if past_events   else 3
    days_until = (future_events[0]['date'] - today).days  if future_events else 5

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
        buy_t  = st.slider("BUY threshold",  0.50, 0.70, 0.57, 0.01)
        sell_t = st.slider("SELL threshold", 0.30, 0.50, 0.43, 0.01)

        if MODEL_LOADED and len(price_df) > 200:
            try:
                X_input, latest_feats, curr_close = compute_features_from_price(
                    price_df, event_surprises, days_since, days_until, any_event
                )
                X_scaled = scaler.transform(X_input)
                prob     = float(model.predict_proba(X_scaled)[0, 1])
                signal   = get_signal(prob, buy_t, sell_t)
                # log every signal generated
                log_signal(signal, prob, curr_close, len(today_events), days_since, days_until)
            except Exception as e:
                st.error(f"Feature computation error: {e}")
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
            if MODEL_LOADED:
                st.dataframe(X_input.T.rename(columns={0:'value'}).round(6), use_container_width=True)

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
            styled_log = log_df.style.map(colour_signal, subset=["signal"])
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

    st.markdown("### Single-Split Performance (80/20 chronological)")
    c1, c2, c3, c4 = st.columns(4)
    if res:
        lr = res['lr']
        c1.metric("AUC-ROC",         f"{lr['single_auc']:.4f}")
        c2.metric("Strategy Return",  f"{lr['return']:+.2f}%")
        c3.metric("Sharpe Ratio",     f"{lr['sharpe']:.2f}")
        c4.metric("Win Rate",         f"{lr['win_rate']*100:.1f}%", f"{lr['n_trades']} trades")
    else:
        c1.metric("AUC-ROC", "0.5358")
        c2.metric("Strategy Return", "+8.13%")
        c3.metric("Sharpe Ratio", "1.51")
        c4.metric("Win Rate", "57.8%", "84 trades")

    st.markdown("### Walk-Forward Validation (13 folds, Nov 2022 – Oct 2025)")
    c1, c2, c3, c4 = st.columns(4)
    if res:
        c1.metric("Mean AUC",       f"{res['lr']['wf_auc']:.4f}")
        c2.metric("Std AUC",        f"±{res['lr']['wf_std']:.4f}")
        c3.metric("Folds > Random", "10 / 13")
        c4.metric("Folds > 0.55",   "6 / 13")
    else:
        c1.metric("Mean AUC", "0.5383")
        c2.metric("Std AUC", "±0.0626")
        c3.metric("Folds > Random", "10 / 13")
        c4.metric("Folds > 0.55", "6 / 13")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Walk-Forward AUC per Fold")
        img = load_img("walkforward_auc_three_models.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("Run Section 8 in notebook.")
    with col2:
        st.markdown("#### Walk-Forward Equity Curves")
        img = load_img("walkforward_equity_three_models.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("Run Section 8 in notebook.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Evaluation Diagnostics")
    img = load_img("evaluation_diagnostics.png")
    if img:
        st.image(img, use_container_width=True)
    else:
        st.info("Run Section 6 in notebook.")

    st.markdown("### Model Comparison")
    if res:
        lr, rf, xgb = res['lr'], res['rf'], res['xgb']
        st.dataframe(pd.DataFrame({
            'Model':            ['Logistic Regression ★', 'Random Forest', 'XGBoost'],
            'Single-Split AUC': [lr['single_auc'], rf['single_auc'], xgb['single_auc']],
            'WF Mean AUC':      [lr['wf_auc'], rf['wf_auc'], xgb['wf_auc']],
            'WF AUC Std':       [lr['wf_std'], rf['wf_std'], xgb['wf_std']],
            'WF Equity Return': [f"{lr['wf_equity']:+.2f}%", f"{rf['wf_equity']:+.2f}%", f"{xgb['wf_equity']:+.2f}%"],
            'Backtest Return':  [f"{lr['return']:+.2f}%", f"{rf['return']:+.2f}%", f"{xgb['return']:+.2f}%"],
            'Sharpe':           [lr['sharpe'], rf['sharpe'], xgb.get('sharpe', '—')],
            'Win Rate':         [f"{lr['win_rate']*100:.1f}%", f"{rf['win_rate']*100:.1f}%", f"{xgb['win_rate']*100:.1f}%"],
            'Trades':           [lr['n_trades'], rf['n_trades'], xgb['n_trades']],
        }), hide_index=True, use_container_width=True)
    else:
        st.dataframe(pd.DataFrame({
            'Model':            ['Logistic Regression ★', 'Random Forest', 'XGBoost'],
            'WF Mean AUC':      [0.5383, 0.5086, 0.5110],
            'WF AUC Std':       [0.0626, 0.0913, 0.0762],
            'WF Equity Return': ['+6.29%', '-4.28%', '-9.45%'],
            'Single-Split AUC': [0.5358, 0.5083, 0.4933],
            'Backtest Return':  ['+8.13%', '—', '—'],
            'Sharpe':           [1.51, '—', '—'],
            'Win Rate':         ['57.8%', '—', '—'],
        }), hide_index=True, use_container_width=True)
    st.caption("★ Primary model selected based on highest walk-forward stability. All metrics from completed experiments.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Interpretability":
    st.markdown("# 🔍 Feature Interpretability")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Permutation Importance", "SHAP Analysis", "Ablation Study"])

    with tab1:
        st.markdown("#### Permutation Importance — Logistic Regression")
        st.caption("Drop in AUC when each feature is randomly shuffled. Higher = more critical.")
        img = load_img("permutation_importance_lr.png")
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
            ("eu_core_cpi_surprise_pct", "EU Core CPI percentile rank — how this ECB surprise compares historically. Top feature overall."),
            ("nfp_day",                  "Binary flag: is today a Non-Farm Payrolls release day? Model uses event timing as a strong signal."),
            ("mom7",                     "7-day momentum (r7d minus r1d) — recent directional price pressure heading into the event."),
            ("eu_core_cpi_surprise",     "Raw EU Core CPI surprise value — actual minus forecast for the ECB core inflation print."),
            ("days_since_event",         "How many days since the last high-impact event. Market tends to drift back to trend after releases."),
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
            img = load_img("shap_summary_lr.png")
            if img:
                st.image(img, width=700)
            else:
                st.info("Run Section 9 in notebook.")
        with c2:
            st.markdown("#### SHAP Bar Chart")
            st.caption("Mean absolute SHAP value — overall importance ranking.")
            img = load_img("shap_bar_lr.png")
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
            <code>days_until_event</code> dominates both charts — the model is most sensitive to how close we are
            to the next high-impact release. This confirms that event timing, not just the surprise magnitude,
            is a critical driver of EUR/USD directional signal.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Ablation Study — Feature Group Contribution")
        st.caption("AUC drop when each feature group is removed. Positive = group adds value.")
        img = load_img("ablation_study_lr.png")
        if img:
            st.image(img, width=700)
        else:
            st.info("Save ablation_study_lr.png from Section 9b.")
        for group, drop, desc in [
            ("USD Events",  "+0.0266", "Largest contributor — CPI and NFP surprises drive the primary signal."),
            ("ECB Events",  "+0.0182", "Second largest — EUR-side surprises provide independent signal."),
            ("Momentum",    "+0.0106", "Lagged returns add moderate signal."),
            ("Time",        "+0.0035", "Days since/until event adds marginal value."),
            ("Regime",      "~0.0000", "200-day MA adds minimal marginal value over event features."),
            ("Technical",   "~0.0000", "RSI, MACD, Bollinger Band add noise rather than signal."),
            ("Volatility",  "-0.0043", "Removal slightly improves AUC — volatility features may add noise."),
        ]:
            st.markdown(f"""<div class="finding-box">
                <strong style="font-family:'IBM Plex Mono',monospace">{group}</strong>
                <span style="color:#1f6feb;font-family:'IBM Plex Mono',monospace;margin-left:8px">{drop}</span>
                <br>{desc}</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Results Summary":
    st.markdown("# 📋 Results Summary")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""
    **Economic Event-Driven Forex Trading Assistant using Machine Learning**  
    This project characterises the **predictability boundary** of EUR/USD around high-impact 
    macroeconomic events. The primary finding is that a **noise ceiling** — not model complexity 
    or hyperparameters — is the binding constraint. AUC converges to 0.51–0.54 regardless of model class.
    """)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### Key Findings")
    for f in [
        "Logistic Regression (C=0.001, L2, saga) — most stable AUC across 13 walk-forward folds.",
        "Walk-forward mean AUC 0.5383 (std 0.0626). 10/13 folds above random, 6/13 above 0.55.",
        "Two catastrophic folds align with sustained USD strength regimes — regime dependency confirmed.",
        "Single-split backtest: +8.13% return, Sharpe 1.51, win rate 57.8% across 84 trades.",
        "Walk-forward equity: LR +6.29% vs buy-and-hold +16.01% in bullish regime.",
        "USD event surprises (NFP, CPI, FOMC) are the primary signal source — ablation drop +0.0266 AUC.",
        "ECB event integration expanded features 33→42 and contributed +0.0182 AUC.",
        "Noise ceiling confirmed: AUC converges to 0.51–0.54 regardless of model complexity.",
    ]:
        st.markdown(f'<div class="finding-box">{f}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### Limitations")
    st.markdown("""
    - Daily EUR/USD direction has limited predictability from macro surprises alone
    - Model degrades in sustained USD strength / strong trending regimes
    - Z-score surprise proxy is imperfect — intraday data may improve signal
    """)
    st.markdown("### Future Work")
    st.markdown("""
    - Intraday models (1H/4H) for immediate post-release price action
    - Ensemble regime-switching: separate models per regime
    - MT5 live deployment with safety locks (demo only, max 0.1 lot, daily loss limit)
    """)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.caption("ICT3902C Capstone | Muhammad Mursyid Bin Hassan | 2300435 | Supervisor: Dr Li Xiaorong")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SIGNAL HISTORY (30-day lookback)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Signal History":
    st.markdown("# 📊 Signal History — Last 30 Days")
    st.caption("Retrospective model signals using live price data + trained model. Shows what the model would have recommended each day.")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("fyp_model.pkl not found. Cannot generate signal history.")
    else:
        with st.spinner("Computing 30-day signal history..."):
            try:
                # Fetch price data
                price_df = fetch_live_price()
                close = price_df['close']

                # Load surprise CSVs for percentile rank
                csv_map = {
                    'cpi':      ('cpi_surprise.csv',         'cpi_surprise'),
                    'nfp':      ('nfp_surprise.csv',          'nfp_surprise'),
                    'fomc':     ('fomc_surprise.csv',         'fomc_surprise'),
                    'ecb_rate': ('ecb_rate_surprise.csv',     'ecb_rate_surprise'),
                    'eu_cpi':   ('eu_cpi_surprise.csv',       'eu_cpi_surprise'),
                    'eu_core':  ('eu_core_cpi_surprise.csv',  'eu_core_cpi_surprise'),
                }
                csv_data = {}
                for key, (fname, col) in csv_map.items():
                    if os.path.exists(fname):
                        df_csv = pd.read_csv(fname, parse_dates=['date'])
                        csv_data[key] = df_csv[col].dropna()

                # Generate signals for last 30 trading days
                lookback = 30
                history_rows = []

                # We need at least 200 rows for 200MA warmup
                if len(price_df) < 220:
                    st.warning("Not enough price history to compute signals.")
                else:
                    eval_indices = list(range(len(price_df) - lookback - 1, len(price_df) - 1))

                    for idx in eval_indices:
                        row_date = price_df.index[idx]
                        # Slice price data up to and including this row
                        sub_df = price_df.iloc[:idx+1].copy()

                        try:
                            X_input, _, _ = compute_features_from_price(
                                sub_df,
                                event_surprises={},  # no live event on historical dates
                                days_since=3,
                                days_until=5,
                                any_event=False
                            )
                            X_scaled = scaler.transform(X_input)
                            prob = float(model.predict_proba(X_scaled)[0, 1])
                            signal = get_signal(prob, 0.57, 0.43)

                            # Actual next-day return
                            if idx + 1 < len(price_df):
                                next_close = float(price_df['close'].iloc[idx + 1])
                                curr_close = float(price_df['close'].iloc[idx])
                                actual_return = (next_close - curr_close) / curr_close * 100
                                actual_direction = "UP" if actual_return > 0 else "DOWN"
                            else:
                                actual_return = None
                                actual_direction = "—"

                            # Hit/miss
                            if signal == "HOLD" or actual_direction == "—":
                                hit = "—"
                            elif (signal == "BUY" and actual_direction == "UP") or \
                                 (signal == "SELL" and actual_direction == "DOWN"):
                                hit = "✅"
                            else:
                                hit = "❌"

                            history_rows.append({
                                "Date": row_date.strftime("%a %d %b %Y"),
                                "Signal": signal,
                                "P(UP)": f"{prob:.3f}",
                                "Next Day Return": f"{actual_return:+.2f}%" if actual_return is not None else "—",
                                "Actual Direction": actual_direction,
                                "Hit/Miss": hit,
                            })
                        except Exception:
                            pass

                if history_rows:
                    hist_df = pd.DataFrame(history_rows[::-1])  # most recent first

                    # Summary stats
                    signals_only = [r for r in history_rows if r["Signal"] != "HOLD"]
                    hits = [r for r in signals_only if r["Hit/Miss"] == "✅"]
                    misses = [r for r in signals_only if r["Hit/Miss"] == "❌"]
                    holds = [r for r in history_rows if r["Signal"] == "HOLD"]

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Signals Generated", len(signals_only), f"of {len(history_rows)} days")
                    c2.metric("Hit Rate", f"{len(hits)/len(signals_only)*100:.0f}%" if signals_only else "—",
                              f"{len(hits)} correct")
                    c3.metric("Misses", len(misses))
                    c4.metric("HOLD Days", len(holds), "below threshold")

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                    # Colour code the table
                    def style_signal(val):
                        if val == "BUY":   return "color: #2ea043; font-weight: bold"
                        if val == "SELL":  return "color: #f85149; font-weight: bold"
                        if val == "HOLD":  return "color: #8b949e"
                        return ""
                    def style_hit(val):
                        if val == "✅": return "color: #2ea043"
                        if val == "❌": return "color: #f85149"
                        return "color: #8b949e"

                    styled = hist_df.style\
                        .map(style_signal, subset=["Signal"])\
                        .map(style_hit, subset=["Hit/Miss"])

                    st.dataframe(styled, hide_index=True, use_container_width=True)

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.caption("⚠️ Signal history uses no event surprise data for historical dates — signals reflect price/regime features only. Live Signal page incorporates today's released events for the current signal.")
                else:
                    st.warning("Could not compute signal history. Check that fyp_model.pkl and price data are available.")

            except Exception as e:
                st.error(f"Error computing signal history: {e}")
