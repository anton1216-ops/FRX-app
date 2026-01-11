import unittest
import pandas as pd
from app import analyze_signal, run_backtest

class TestAppLogic(unittest.TestCase):
    def test_analyze_signal(self):
        row = {'ema9': 1.2, 'ema21': 1.0, 'close': 1.5, 'sma200': 1.0}
        buy, sell, trend = analyze_signal(row)
        self.assertTrue(buy)
        self.assertFalse(sell)
        self.assertTrue(trend)

    def test_run_backtest(self):
        idx = pd.date_range("2023-01-01", periods=250, freq="H")
        df = pd.DataFrame({
            'ema9': [1.1]*250,
            'ema21': [1.0]*250,
            'close': [1.2]*250,
            'sma200': [1.0]*250,
            'atr': [0.01]*250,
            'high': [1.22]*250,
            'low': [1.18]*250,
        }, index=idx)
        balance, curve, wins, losses, total_p, total_l = run_backtest(df, 1000, 1, 2, 2, False)
        self.assertTrue(balance >= 1000)
        self.assertTrue(len(curve) > 1)

if __name__ == "__main__":
    unittest.main()