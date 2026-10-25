import unittest

import pandas as pd

from app.data.market_data import MarketDataService


class TestWeeklyMissingDataFill(unittest.TestCase):
    def test_missing_week_repeats_previous_ohlcv_and_not_corporate_actions(self):
        daily = pd.DataFrame(
            {
                "Open": [10.0, 12.0],
                "High": [11.0, 13.0],
                "Low": [9.0, 11.0],
                "Close": [10.5, 12.5],
                "Adj Close": [10.5, 12.5],
                "Volume": [1000, 2000],
                "Dividends": [0.5, 0.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.to_datetime(["2024-01-05", "2024-01-19"]),
        )

        weekly = MarketDataService().convert_to_weekly(daily)

        self.assertEqual(
            list(weekly.index.strftime("%Y-%m-%d")),
            ["2024-01-05", "2024-01-12", "2024-01-19"],
        )
        # The empty week repeats the prior week's OHLCV, including volume,
        # as requested for continuity in weekly VFI calculations.
        for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
            self.assertEqual(weekly.loc["2024-01-12", column], weekly.loc["2024-01-05", column])
        self.assertEqual(weekly.loc["2024-01-12", "Dividends"], 0.0)
        self.assertEqual(weekly.loc["2024-01-12", "Stock Splits"], 0.0)


if __name__ == "__main__":
    unittest.main()
