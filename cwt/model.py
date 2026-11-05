import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

POS = {'bullish','gain','gains','up','rally','rise','rising','buy','breakout','strong','surge','profit','positive','adoption','approve'}
NEG = {'bearish','loss','losses','down','crash','fall','falling','sell','weak','drop','dump','negative','ban','fraud'}


def score(text):
    import re
    words = re.findall(r'[a-z]+', text.lower())
    return (sum(w in POS for w in words) - sum(w in NEG for w in words)) / max(1, sum(w in POS or w in NEG for w in words))


FEATURES = ['return_1','return_2','return_5','sentiment','sentiment_lag1','sentiment_mean3',
            'post_volume','post_volume_lag1','sentiment_delta']
PRICE_FEATURES = ['return_1','return_2','return_5']


def features(posts, prices):
    p = posts.copy()
    p['date'] = p.timestamp.dt.strftime('%Y-%m-%d')
    p['ny_time'] = p.timestamp.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d %H:%M:%S%z')
    p['sentiment_score'] = p.text.map(score)
    p['weight'] = np.clip(np.log1p(p.likes + 2*p.reposts + p.views/100 + p.followers/1000), 1, 10)
    daily = p.groupby('date').apply(lambda x: pd.Series({'post_volume': len(x),
          'sentiment': np.average(x.sentiment_score, weights=x.weight)}), include_groups=False).reset_index()
    bars = prices.merge(daily, on='date', how='left').sort_values('date').reset_index(drop=True)
    return feature_bars(bars), p


def feature_bars(bars):
    bars = bars.copy().sort_values('date').reset_index(drop=True)
    bars[['post_volume','sentiment']] = bars[['post_volume','sentiment']].fillna(0)
    for lag in (1,2,5):
        bars[f'return_{lag}'] = bars.close.pct_change(lag, fill_method=None)
    bars['sentiment_lag1'] = bars.sentiment.shift(1)
    bars['sentiment_mean3'] = bars.sentiment.rolling(3).mean()
    bars['post_volume_lag1'] = bars.post_volume.shift(1)
    bars['sentiment_delta'] = bars.sentiment.diff()
    for h in range(1,6):
        bars[f'target_{h}'] = bars.close.shift(-h)/bars.close - 1
    return bars


def walk_forward(bars, min_train=60, feature_columns=FEATURES):
    if len(bars) < min_train + 6:
        raise ValueError(f'Need at least {min_train + 6} aligned daily price bars; got {len(bars)}.')
    rows = []
    x = bars[feature_columns]
    for h in range(1,6):
        target = bars[f'target_{h}']
        for i in range(min_train, len(bars)-h):
            # At issue close i, label j is observable iff j+h <= i.
            train_ids = [j for j in range(i-h+1) if x.iloc[j].notna().all() and pd.notna(target.iloc[j])]
            if len(train_ids) < min_train-5 or x.iloc[i].isna().any():
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(x.iloc[train_ids], target.iloc[train_ids])
            predicted = float(model.predict(x.iloc[[i]])[0])
            rows.append({'date': bars.date.iloc[i], 'horizon': h, 'close': bars.close.iloc[i],
                         'actual_return': target.iloc[i], 'pred_return': predicted,
                         'actual_price': bars.close.iloc[i+h], 'pred_price': bars.close.iloc[i]*(1+predicted),
                         'train_count': len(train_ids), 'last_label_date': bars.date.iloc[train_ids[-1]+h]})
    if not rows:
        raise ValueError('No valid walk-forward predictions; check history length and missing features.')
    return pd.DataFrame(rows)


def current_forecast(bars):
    last = bars.iloc[-1]
    out = [{'horizon': 0, 'date': last.date, 'price': float(last.close), 'type': 'observed close'}]
    for h in range(1,6):
        train = bars.iloc[:-h].dropna(subset=FEATURES+[f'target_{h}'])
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(train[FEATURES], train[f'target_{h}'])
        ret = float(model.predict(bars[FEATURES].iloc[[-1]])[0])
        out.append({'horizon': h, 'date': (pd.Timestamp(last.date)+pd.Timedelta(days=h)).strftime('%Y-%m-%d'),
                    'price': float(last.close*(1+ret)), 'type': 'forecast'})
    return pd.DataFrame(out)


def evaluate(walk, fee_bps=10):
    metrics = {}
    for h, group in walk.groupby('horizon'):
        error = group.pred_price - group.actual_price
        baseline = group.close - group.actual_price
        metrics[str(h)] = {'n': len(group), 'mae': float(error.abs().mean()), 'rmse': float(np.sqrt(np.mean(error**2))),
                           'naive_mae': float(baseline.abs().mean()), 'direction_accuracy': float(np.mean(np.sign(group.pred_return)==np.sign(group.actual_return)))}
    one = walk[walk.horizon==1].sort_values('date').copy()
    one['position'] = (one.pred_return > 0).astype(int)
    one['turnover'] = one.position.diff().abs().fillna(one.position)
    one['strategy_return'] = one.position*one.actual_return - one.turnover*fee_bps/10000
    r = one.strategy_return.to_numpy()
    downside = np.sqrt(np.mean(np.minimum(r, 0)**2))
    equity = np.cumprod(1+r)
    metrics['strategy'] = {'n': len(r), 'fee_bps': fee_bps,
        'sharpe': float(np.sqrt(365)*np.mean(r)/np.std(r, ddof=1)) if len(r)>1 and np.std(r,ddof=1)>0 else None,
        'sortino': float(np.sqrt(365)*np.mean(r)/downside) if downside>0 else None,
        'max_drawdown': float(np.min(equity/np.maximum.accumulate(np.r_[1,equity])[1:]-1)),
        'total_return': float(equity[-1]-1), 'turnover': float(one.turnover.sum())}
    return metrics
