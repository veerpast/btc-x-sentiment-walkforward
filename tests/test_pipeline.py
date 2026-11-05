import unittest
import numpy as np
import pandas as pd
from cwt.model import features, walk_forward, evaluate, score
from scripts.build_historical_daily import aggregate_row


class PipelineTests(unittest.TestCase):
    def test_archival_parser_utc_and_deduplication(self):
        row = {'id_str': '1', 'created_at': '2017-10-02 00:30:00 IST',
               'text': ['Bitcoin', 'bullish'], 'likes_count': 0,
               'retweets_count': 0, 'Follower_count': 0}
        seen, totals = set(), {}
        aggregate_row(row, seen, totals)
        aggregate_row(row, seen, totals)
        self.assertEqual(totals['2017-10-01'][0], 1)
        self.assertEqual(totals['2017-10-01'][2], 1)

    def test_sentiment_and_time_cutoff(self):
        self.assertGreater(score('Bitcoin bullish rally'), 0)
        self.assertLess(score('Bitcoin bearish crash'), 0)
        dates = pd.date_range('2025-01-01', periods=85, tz='UTC')
        prices = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'),
                               'close': 30000+np.arange(85)*10+100*np.sin(np.arange(85)/3)})
        posts = pd.DataFrame({'id': [str(i) for i in range(85)], 'timestamp': dates,
                              'text': ['Bitcoin bullish' if i%2 else 'Bitcoin bearish' for i in range(85)],
                              'likes': [2]*85, 'reposts': [1]*85, 'views': [30]*85, 'followers': [100]*85})
        bars, _ = features(posts, prices)
        walk = walk_forward(bars)
        self.assertTrue((walk.last_label_date <= walk.date).all())
        self.assertTrue((walk.pred_price > 0).all())
        self.assertIn('max_drawdown', evaluate(walk)['strategy'])
        changed = prices.copy()
        changed.loc[changed.index[-1], 'close'] *= 3
        altered, _ = features(posts, changed)
        self.assertEqual(bars.loc[70, 'return_1'], altered.loc[70, 'return_1'])


if __name__ == '__main__':
    unittest.main()
