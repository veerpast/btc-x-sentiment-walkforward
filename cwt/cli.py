import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .ingest import collect_apify, collect_news, load_posts, load_prices, store
from .model import current_forecast, evaluate, features, walk_forward


def chart(bars, path):
    width, height = 1100, 450
    sample = bars.tail(180).reset_index(drop=True)
    if len(sample) < 2:
        return
    def series(values, ymin, ymax, top, bottom):
        span = ymax-ymin or 1
        return ' '.join(f'{40+i*(width-80)/(len(values)-1):.1f},{bottom-(v-ymin)/span*(bottom-top):.1f}' for i,v in enumerate(values))
    prices = sample.close.to_numpy()
    sentiments = sample.sentiment.to_numpy()
    p1 = series(prices, min(prices), max(prices), 45, 185)
    p2 = series(sentiments, -1, 1, 245, 390)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font:15px Arial;fill:#333}} .axis{{stroke:#bbb}} </style>
<text x="40" y="25">BTC daily close (USD)</text><text x="40" y="225">X weighted sentiment (-1 to +1)</text>
<path class="axis" d="M40 190H1060 M40 395H1060"/>
<polyline points="{p1}" fill="none" stroke="#174ea6" stroke-width="2"/>
<polyline points="{p2}" fill="none" stroke="#a74925" stroke-width="2"/>
<text x="40" y="430">{sample.date.iloc[0]}</text><text x="960" y="430">{sample.date.iloc[-1]}</text>
</svg>'''
    path.write_text(svg)


def report(out, bars, posts, walk, metrics, news, source):
    fmt = lambda x: 'undefined' if x is None else f'{x:.4f}'
    lines = ['# BTC X sentiment: evaluation', '',
        f'Data: `{source}`; UTC dates {bars.date.iloc[0]} through {bars.date.iloc[-1]}. '
        f'{len(posts)} filtered, deduplicated X posts; {len(bars)} daily price bars. '
        f'{int((bars.post_volume>0).sum())} days with posts. These are historical observations, not a claim of current 180-day coverage.', '',
        'Issue time is just after the UTC daily close. `n+0` is the observed close; no predictive metric is assigned. '
        'Horizons 1–5 are direct return forecasts. Each test point uses a refitted expanding Ridge model; '
        'the training-label cutoff is at least the horizon before the issue date. No test-period tuning was performed.', '',
        '| Horizon | OOS n | MAE (USD) | RMSE (USD) | Naive MAE (USD) | Direction accuracy |',
        '|---:|---:|---:|---:|---:|---:|']
    for h in range(1,6):
        m = metrics[str(h)]
        lines.append(f"| {h} | {m['n']} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['naive_mae']:.2f} | {m['direction_accuracy']:.1%} |")
    m = metrics['strategy']
    lines += ['', '## Idealized one-day signal diagnostic', '',
        f"Long BTC if the one-day forecast return is positive; otherwise cash. Assumes an idealized fill at the issue close; "
        f"{m['fee_bps']} bps per position change, with no additional slippage or financing. "
        f"Annualization uses 365 crypto trading days. OOS observations: {m['n']}; "
        f"Sharpe {fmt(m['sharpe'])}; Sortino {fmt(m['sortino'])}; "
        f"maximum drawdown {m['max_drawdown']:.2%}; total return {m['total_return']:.2%}; "
        f"turnover {m['turnover']:.0f}.", '',
        'Sharpe is mean daily net return divided by sample standard deviation times √365. '
        'Sortino uses the root mean square of returns below a zero target. Maximum drawdown is the minimum '
        'equity-to-running-peak return, including starting equity of 1. The signal uses that completed close, '
        'so an exact fill there is generally unavailable. These are signal diagnostics, not executable or live returns.', '',
        '## Caveats', '',
        'Post counts and engagement can be biased by scrape coverage. A simple lexicon can misread sarcasm or context. '
        'A live strategy needs a feasible post-close fill rule and executable prices. '
        'Yahoo daily close timing and post timestamp definitions should be checked before live use. '
        'No hyperparameter selection or statistical significance testing was done. Recent Tavily news is excluded from training.', '',
        '![Historical price and sentiment](sentiment_price.svg)']
    (out/'evaluation.md').write_text('\n'.join(lines)+'\n')


def outlook(out, bars, posts, news):
    latest = bars.iloc[-1]
    trailing = bars.tail(7)
    recent_posts = posts[posts.date.isin(trailing.date)]
    direction = 'positive' if latest.sentiment > .1 else 'negative' if latest.sentiment < -.1 else 'mixed or neutral'
    lines = ['# BTC sentiment outlook', '', f'As of {latest.date} (UTC daily close): {direction}.', '',
      f"Latest weighted X score: {latest.sentiment:+.3f} on [-1,1], from {int(latest.post_volume)} posts. "
      f"Seven-day mean: {trailing.sentiment.mean():+.3f}; seven-day post volume: {int(trailing.post_volume.sum())}. "
      f"BTC close: ${latest.close:,.2f}; seven-day close change: "
      f"{(latest.close/trailing.close.iloc[0]-1):+.2%}.", '',
      f'Coverage: {len(recent_posts)} collected posts in the last seven price bars. '
      'This is a corpus-relative reading, not necessarily a current market view. Engagement-weighted scores '
      'can overrepresent large accounts; the lexicon is imperfect.', '', '## Recent news context (not model input)', '']
    lines += [f"- {n.get('title') or 'Untitled'} ({n.get('published_date') or 'date unavailable'}): {n.get('url') or ''}" for n in news] or ['No Tavily news fetched.']
    lines += ['', 'The news headlines are contextual and do not alter the forecasts. This is research, not investment advice.']
    (out/'outlook.md').write_text('\n'.join(lines)+'\n')


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description='CWT daily X sentiment experiment')
    parser.add_argument('--posts', required=True, help='Apify JSON or historical post CSV')
    parser.add_argument('--prices', help='Optional CSV with date,close')
    parser.add_argument('--asset', default='BTC-USD')
    parser.add_argument('--out', default='results')
    parser.add_argument('--start', help='UTC inclusive start; default first post date')
    parser.add_argument('--end', help='UTC inclusive end; default last post date')
    parser.add_argument('--news', action='store_true', help='Fetch Tavily news context')
    parser.add_argument('--collect-apify', action='store_true')
    parser.add_argument('--actor-id', default=os.environ.get('APIFY_ACTOR_ID','apidojo/twitter-scraper-lite'))
    parser.add_argument('--actor-input', help='Actor-specific JSON input; defaults to simple Bitcoin searches')
    args = parser.parse_args()
    if args.collect_apify:
        payload = json.loads(Path(args.actor_input).read_text()) if args.actor_input else {
          'searchTerms': ['$BTC', '#Bitcoin', 'Bitcoin'], 'maxItems': 1000}
        print(f'Collected {collect_apify(payload, args.actor_id, args.posts)} Apify items')
    posts = load_posts(args.posts)
    start = args.start or posts.timestamp.min().strftime('%Y-%m-%d')
    end = args.end or posts.timestamp.max().strftime('%Y-%m-%d')
    posts = posts[(posts.timestamp.dt.strftime('%Y-%m-%d')>=start) &
                  (posts.timestamp.dt.strftime('%Y-%m-%d')<=end)].copy()
    if posts.empty:
        raise ValueError('No posts in the selected date window.')
    prices = load_prices(args.prices, args.asset, start, end)
    if prices.empty:
        raise ValueError('No matching price bars.')
    expected = pd.date_range(prices.date.iloc[0], prices.date.iloc[-1], freq='D')
    if len(prices) != len(expected):
        raise ValueError('Daily BTC prices have missing dates; horizons require consecutive bars.')
    news = collect_news('Bitcoin BTC latest market news') if args.news else []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store(out/'data.sqlite', posts, prices, news)
    bars, scored = features(posts, prices)
    walk = walk_forward(bars)
    metrics = evaluate(walk)
    forecast = current_forecast(bars)
    walk.to_csv(out/'walk_forward.csv', index=False)
    forecast.to_csv(out/'forecasts.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    chart(bars, out/'sentiment_price.svg')
    report(out, bars, scored, walk, metrics, news, args.posts)
    outlook(out, bars, scored, news)
    print(f'Wrote verified historical evaluation to {out}; OOS n+1 observations: {metrics["1"]["n"]}')


if __name__ == '__main__':
    main()
