"""Rebuild the derived historical BTC daily aggregate from public CrypTop12 files.

Run from repository root: python scripts/build_historical_daily.py
This downloads source posts for the specified archival window; it never uploads
local data or uses any authentication token.
"""
import argparse
import csv
import json
import math
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cwt.model import score  # noqa: E402


SOURCE_COMMIT = 'f05691aeee35349c8adb02f17940b91a1531b399'
BASE = f'https://raw.githubusercontent.com/am15h/CrypTop12/{SOURCE_COMMIT}'
START = date(2017, 9, 21)
END = date(2018, 3, 19)


def aggregate_row(row, seen, totals):
    body = ' '.join(row['text']) if isinstance(row.get('text'), list) else str(row.get('text') or '')
    if not re.search(r'(bitcoin|#\s*btc|\$\s*btc|\bbtc\b)', body, re.I):
        return
    key = str(row.get('id_str') or (str(row.get('created_at')) + body))
    if key in seen:
        return
    seen.add(key)
    raw_time = str(row.get('created_at') or '').replace(' IST', '+05:30').replace(' ', 'T')
    timestamp = pd.to_datetime(raw_time, utc=True, errors='coerce')
    if pd.isna(timestamp):
        return
    def positive(name):
        try:
            return max(0, float(row.get(name) or 0))
        except (ValueError, TypeError):
            return 0
    weight = min(10, max(1, math.log1p(positive('likes_count') +
                     2*positive('retweets_count') + positive('Follower_count')/1000)))
    item = totals.setdefault(timestamp.strftime('%Y-%m-%d'), [0, 0., 0.])
    item[0] += 1
    item[1] += weight
    item[2] += score(body)*weight


def download(session, url):
    response = session.get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


def build(output):
    with requests.Session() as session:
        raw_prices = download(session, BASE+'/price/raw/btc.csv')
        if not raw_prices:
            raise RuntimeError('Source BTC price CSV is unavailable.')
        prices = {row['Date']: float(row['Close']) for row in csv.DictReader(raw_prices.splitlines())
                  if row.get('Date') and row.get('Close')}
        seen, totals, missing = set(), {}, []
        day = START
        while day <= END:
            name = day.isoformat()
            text = download(session, BASE+'/tweet/preprocessed/btc/'+name)
            if text is None:
                missing.append(name)
            else:
                for line in text.splitlines():
                    if line.strip():
                        aggregate_row(json.loads(line), seen, totals)
            day += timedelta(days=1)
    first, last = date(2017,10,2), END
    rows = []
    day = first
    while day <= last:
        name = day.isoformat()
        if name not in prices:
            raise RuntimeError(f'Missing BTC close for {name}; refusing to fill price.')
        n, weight, weighted_sum = totals.get(name, [0, 0., 0.])
        rows.append((name, prices[name], weighted_sum/weight if weight else 0., n))
        day += timedelta(days=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['date','close','sentiment','post_volume'])
        writer.writerows(rows)
    print(f'{len(rows)} price bars, {sum(x[3] for x in rows)} matched posts; '
          f'{len(missing)} unavailable source-day files: {missing}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('data/historical_daily.csv'))
    build(parser.parse_args().out)
