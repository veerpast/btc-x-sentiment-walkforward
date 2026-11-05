"""Reproduce the archival CrypTop12 evaluation from a derived daily aggregate."""
import json
from pathlib import Path

import pandas as pd

from .cli import chart
from .model import PRICE_FEATURES, current_forecast, evaluate, feature_bars, walk_forward


def main():
    source = Path('data/historical_daily.csv')
    raw = pd.read_csv(source)
    if len(raw) < 66 or raw.date.duplicated().any():
        raise ValueError('Historical daily data has insufficient or duplicate dates.')
    # Include all available price days. Zero post days mean no matching collected posts.
    dates = pd.date_range(raw.date.min(), raw.date.max(), freq='D').strftime('%Y-%m-%d')
    if len(dates) != len(raw):
        raise ValueError('Historical aggregate lacks matching price days; do not silently fill prices.')
    bars = feature_bars(raw)
    walk = walk_forward(bars)
    metrics = evaluate(walk)
    price_only = evaluate(walk_forward(bars, feature_columns=PRICE_FEATURES))
    metrics['price_only'] = {h: price_only[h] for h in ('1','2','3','4','5')}
    out = Path('results/historical')
    out.mkdir(parents=True, exist_ok=True)
    walk.to_csv(out/'walk_forward.csv', index=False)
    current_forecast(bars).to_csv(out/'forecasts_at_historical_end.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    chart(bars, out/'sentiment_price.svg')
    lines = ['# Historical BTC X sentiment evaluation', '',
        '## Provenance and scope', '',
        'Source: [CrypTop12](https://github.com/am15h/CrypTop12), commit '
        '`f05691aeee35349c8adb02f17940b91a1531b399`, `tweet/preprocessed/btc/YYYY-MM-DD` and '
        '`price/raw/btc.csv`. We parsed each JSON-lines post, filtered Bitcoin/BTC terms, deduplicated by post ID, '
        'converted the source IST timestamps to UTC dates, scored text with the transparent lexicon in `cwt/model.py`, '
        'and capped engagement weights. `data/historical_daily.csv` holds the derived daily score, count and matching close. '
        'No raw tweets or credentials are redistributed. The source price series has matching bars from '
        f"{raw.date.iloc[0]} through {raw.date.iloc[-1]}; {int(raw.post_volume.sum())} matching posts "
        f"across {len(raw)} daily bars. This is an archival experiment, **not** a recent 180-day scrape or current outlook.", '',
        '## Timing and validation', '',
        'Forecasts are issued after each UTC daily close. `n+0` is the observed close and is not scored as a forecast. '
        'For horizon h, a training label is used only if its closing date is no later than the issue date. '
        'Expanding Ridge models use price returns and past/current daily X sentiment and post volume only. '
        'The naive baseline predicts no price change. No test-set tuning was performed. '
        'The original price file does not document its exact close timestamp; this daily alignment is an assumption '
        'that needs source verification before intraday or live claims.', '',
        '| Horizon | OOS n | X+price MAE | Price-only MAE | No-change MAE | X+price RMSE | Direction accuracy |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for h in range(1,6):
        m=metrics[str(h)]
        p=price_only[str(h)]
        lines.append(f"| {h} | {m['n']} | {m['mae']:.2f} | {p['mae']:.2f} | {m['naive_mae']:.2f} | {m['rmse']:.2f} | {m['direction_accuracy']:.1%} |")
    lines += ['', 'In this window, adding the simple X sentiment features increased MAE versus the price-only model '
              'at every horizon. Both fitted models lost to the no-change forecast. This is negative evidence for '
              'the current feature design, and a reason to improve data coverage and sentiment quality before '
              'using the model for a live decision.']
    s=metrics['strategy']
    lines += ['', '## One-day idealized long/cash signal diagnostic', '',
        f"The position is long BTC when the one-day predicted return is positive and otherwise cash. "
        f"The calculation assumes execution at the issue close and {s['fee_bps']} bps per position change. "
        'Because the signal uses that completed close, exact execution there is generally unavailable. '
        'These metrics diagnose the signal; they are not an executable backtest. '
        f"For {s['n']} out-of-sample days: Sharpe {s['sharpe']:.3f}, Sortino "
        f"{s['sortino']:.3f} if defined, maximum drawdown {s['max_drawdown']:.2%}, "
        f"total return {s['total_return']:.2%}, turnover {s['turnover']:.0f}. "
        'Sharpe uses sample daily standard deviation and √365; Sortino uses the root mean square of returns below zero. '
        'Drawdown is measured from a starting equity of 1. No slippage, financing or market impact is modeled.', '',
        '## Historical sentiment outlook at the dataset end', '',
        f"On {raw.date.iloc[-1]} the weighted daily score was {raw.sentiment.iloc[-1]:+.3f} "
        f"from {int(raw.post_volume.iloc[-1])} posts. The trailing seven-day mean was "
        f"{raw.sentiment.tail(7).mean():+.3f}, across {int(raw.post_volume.tail(7).sum())} posts. "
        f"The BTC close was ${raw.close.iloc[-1]:,.2f}. This statement describes March 2018 only. "
        'The current sentiment requires a fresh scrape and news retrieval.', '',
        '## Limitations', '',
        'Selection and scraping coverage, bot posts, lexicon ambiguity, timezone/bar conventions and the absence of '
        'slippage can materially alter conclusions. Price and sentiment may be contemporaneously correlated. '
        'A live strategy needs a feasible post-close fill rule and matching executable prices. '
        'The same corpus is not proof of present performance.', '',
        '![Historical BTC close and X sentiment](sentiment_price.svg)']
    (out/'evaluation.md').write_text('\n'.join(lines)+'\n')
    print('Historical evaluation:', out, 'one-day OOS n=', metrics['1']['n'])


if __name__ == '__main__':
    main()
