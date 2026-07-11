"""
Standalone ForexFactory calendar scraper.
Run on a schedule (via GitHub Actions) to keep ff_calendar.json fresh,
independent of the Streamlit Cloud deployment (which is blocked by
ForexFactory's anti-bot protection on its shared IP ranges).
"""
import json
from datetime import datetime, date
import cloudscraper
from bs4 import BeautifulSoup

TARGET_KEYWORDS = [
    'CPI m/m', 'Non-Farm Employment Change', 'Federal Funds Rate',
    'Main Refinancing Rate', 'CPI Flash Estimate', 'Core CPI Flash Estimate'
]
TARGET_CURRENCIES = {'USD', 'EUR'}


def parse_events(soup, today):
    events = []
    current_date = None
    rows = soup.find_all('tr', class_='calendar__row')
    for row in rows:
        date_cell = row.find('td', class_='calendar__date')
        if date_cell and date_cell.text.strip():
            try:
                parsed = datetime.strptime(
                    date_cell.text.strip().replace('\n', ' ').strip()[:10], '%a %b %d'
                ).replace(year=today.year)
                current_date = parsed.date()
            except Exception:
                pass
        currency_cell = row.find('td', class_='calendar__currency')
        if not currency_cell or currency_cell.text.strip() not in TARGET_CURRENCIES:
            continue
        event_cell = row.find('td', class_='calendar__event')
        if not event_cell:
            continue
        event_name = event_cell.text.strip()
        if not any(k in event_name for k in TARGET_KEYWORDS):
            continue
        actual = row.find('td', class_='calendar__actual')
        forecast = row.find('td', class_='calendar__forecast')
        previous = row.find('td', class_='calendar__previous')
        time_cell = row.find('td', class_='calendar__time')
        events.append({
            'date': str(current_date) if current_date else None,
            'time': time_cell.text.strip() if time_cell else '—',
            'currency': currency_cell.text.strip(),
            'event': event_name,
            'forecast': forecast.text.strip() if forecast else '—',
            'previous': previous.text.strip() if previous else '—',
            'actual': actual.text.strip() if actual else '—',
            'is_today': (current_date == today) if current_date else False,
            'released': bool(actual and actual.text.strip() not in ('', '—')),
        })
    return events


def main():
    today = date.today()
    url = "https://www.forexfactory.com/calendar?week=this"

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    resp = scraper.get(url, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, 'html.parser')
    events = parse_events(soup, today)

    if not events:
        print("WARNING: scrape succeeded but found 0 matching events. "
              "Not overwriting existing ff_calendar.json.")
        return

    with open('ff_calendar.json', 'w') as f:
        json.dump({
            'scraped_at': str(datetime.now()),
            'source': 'github_actions',
            'events': events
        }, f, indent=2)

    print(f"Wrote {len(events)} events to ff_calendar.json at {datetime.now()}")


if __name__ == "__main__":
    main()
