import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests


def load_posts(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        rows = pd.read_csv(path).to_dict('records')
    else:
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get('items', [])
    out = []
    for row in rows:
        ts = next((row[k] for k in ('createdAt', 'created_at', 'date', 'timestamp', 'time') if row.get(k)), None)
        body = next((row[k] for k in ('fullText', 'text', 'tweetText', 'content') if row.get(k)), None)
        if not ts or not body:
            continue
        body = str(body)
        if not re.search(r'(bitcoin|#\s*btc|\$\s*btc|\bbtc\b)', body, re.I):
            continue
        dt = pd.to_datetime(ts, utc=True, errors='coerce')
        if pd.isna(dt):
            continue
        author = row.get('author') or row.get('user') or {}
        if not isinstance(author, dict):
            author = {}
        def num(*keys):
            for key in keys:
                val = row.get(key, author.get(key, 0))
                try:
                    return max(0, int(float(val or 0)))
                except (TypeError, ValueError):
                    pass
            return 0
        identifier = str(row.get('id') or row.get('tweetId') or hashlib.sha256((str(dt)+body).encode()).hexdigest())
        out.append(dict(id=identifier, timestamp=dt, text=body, likes=num('likeCount','likes','favoriteCount'),
                        reposts=num('retweetCount','reposts','retweets'), views=num('viewCount','views'),
                        followers=num('followersCount','followers','followers_count')))
    if not out:
        raise ValueError('No usable Bitcoin posts: check the actor dataset, date and text fields.')
    return pd.DataFrame(out).drop_duplicates('id').sort_values('timestamp').reset_index(drop=True)


def load_prices(path=None, asset='BTC-USD', start=None, end=None):
    if path:
        frame = pd.read_csv(path)
        names = {c.lower(): c for c in frame.columns}
        frame = frame.rename(columns={names['date']: 'date', names['close']: 'close'})[['date','close']]
    else:
        params = {'period1': int(pd.Timestamp(start, tz='UTC').timestamp()),
                  'period2': int((pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)).timestamp()), 'interval': '1d'}
        response = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{asset}', params=params, timeout=30)
        response.raise_for_status()
        result = response.json()['chart']['result'][0]
        frame = pd.DataFrame({'date': pd.to_datetime(result['timestamp'], unit='s', utc=True).strftime('%Y-%m-%d'),
                              'close': result['indicators']['quote'][0]['close']})
    frame['date'] = pd.to_datetime(frame['date'], utc=True).dt.strftime('%Y-%m-%d')
    frame['close'] = pd.to_numeric(frame['close'], errors='coerce')
    frame = frame.dropna().query('close > 0').drop_duplicates('date', keep='last').sort_values('date')
    return frame[(frame.date >= start) & (frame.date <= end)].reset_index(drop=True)


def collect_apify(actor_input, actor_id, out_path):
    token = os.environ.get('APIFY_TOKEN')
    if not token:
        raise ValueError('Set APIFY_TOKEN in the environment.')
    endpoint = f'https://api.apify.com/v2/acts/{actor_id.replace("/", "~")}/run-sync-get-dataset-items'
    response = requests.post(endpoint, params={'timeout': 300},
                             headers={'Authorization': f'Bearer {token}'}, json=actor_input, timeout=330)
    response.raise_for_status()
    items = response.json()
    if not isinstance(items, list) or not items:
        raise ValueError('Apify returned no items; review the actor input and run log.')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(items, indent=2))
    return len(items)


def collect_news(query):
    token = os.environ.get('TAVILY_API_KEY')
    if not token:
        return []
    response = requests.post('https://api.tavily.com/search', json={'api_key': token, 'query': query,
                              'topic': 'news', 'time_range': 'week', 'max_results': 5}, timeout=30)
    response.raise_for_status()
    return [{k: row.get(k) for k in ('title','url','published_date')} for row in response.json().get('results', [])]


def store(db_path, posts, prices, news):
    with sqlite3.connect(db_path) as conn:
        p = posts.copy()
        p['timestamp'] = p.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        p.to_sql('posts', conn, if_exists='replace', index=False)
        prices.to_sql('prices', conn, if_exists='replace', index=False)
        pd.DataFrame(news, columns=['title','url','published_date']).to_sql('news', conn, if_exists='replace', index=False)
